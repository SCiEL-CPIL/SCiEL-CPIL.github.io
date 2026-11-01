"""Combined program: Strassen blocks + structured search for 2x2x3 rank-10."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 140  # 10s safety margin

    # Build rank-53 baseline (7 Strassen blocks + 1 schoolbook)
    U53, V53, W53 = _build_strassen_53(n, m, p)
    
    if not _verify_fast(U53, V53, W53, n, m, p):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    best_rank = 53
    best_U, best_V, best_W = U53, V53, W53

    # Try to find rank-10 decomposition for 2x2x3
    if time.time() < deadline - 5:
        U10, V10, W10 = _find_2x2x3_rank10(deadline)
        if U10 is not None:
            U52, V52, W52 = _build_combined_52(n, m, p, U10, V10, W10)
            if _verify_fast(U52, V52, W52, n, m, p):
                best_rank = 52
                best_U, best_V, best_W = U52, V52, W52

    return {"U": best_U, "V": best_V, "W": best_W}


def _build_strassen_53(n, m, p):
    """Build rank-53: 7 Strassen blocks (rank 7 each) + 1 schoolbook (rank 4)."""
    n, m, p = 2, 2, 15
    blocks = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,)]
    
    all_U, all_V, all_W = [], [], []

    for block in blocks:
        if len(block) == 2:
            k0, k1 = block
            terms = _strassen_terms(k0, k1)
        else:
            k = block[0]
            terms = _schoolbook_terms(k)
        
        for (u_dict, v_dict, w_dict) in terms:
            u_entries = [(i*m+j, c) for (i,j), c in u_dict.items()]
            v_entries = [(j*p+k_idx, c) for (j,k_idx), c in v_dict.items()]
            w_entries = [(k_idx*n+i, c) for (k_idx,i), c in w_dict.items()]
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)

    rank = len(all_U)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for c in range(rank):
        for row, val in all_U[c]:
            U[row][c] = val
        for row, val in all_V[c]:
            V[row][c] = val
        for row, val in all_W[c]:
            W[row][c] = val

    return U, V, W


def _strassen_terms(k0, k1):
    """Strassen rank-7 terms for 2x2x2 block with k indices k0, k1."""
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    return [
        # M1
        ({(0,0):1, (1,1):1}, {(0,k0):1, (1,k1):1}, {(k0,0):1, (k1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,k0):1}, {(k0,1):1, (k1,1):-1}),
        # M3
        ({(0,0):1}, {(0,k1):1, (1,k1):-1}, {(k1,0):1, (k1,1):1}),
        # M4
        ({(1,1):1}, {(1,k0):1, (0,k0):-1}, {(k0,0):1, (k0,1):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,k1):1}, {(k0,0):-1, (k1,0):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,k0):1, (0,k1):1}, {(k1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,k0):1, (1,k1):1}, {(k0,0):1}),
    ]


def _schoolbook_terms(k):
    """Schoolbook rank-4 terms for 2x2x1 block."""
    return [
        ({(0,0):1}, {(0,k):1}, {(k,0):1}),
        ({(0,1):1}, {(1,k):1}, {(k,0):1}),
        ({(1,0):1}, {(0,k):1}, {(k,1):1}),
        ({(1,1):1}, {(1,k):1}, {(k,1):1}),
    ]


def _build_schoolbook(n, m, p):
    """Build schoolbook rank-60 decomposition."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _verify_fast(U, V, W, n, m, p):
    """Verify exact decomposition using tensor contraction (fast)."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Tensor contraction: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    
    # For each (i,j,k): sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # This is a 3-way contraction
    # Efficient: for each r, compute outer product contributions
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    
    # Use einsum
    # U3: (n, m, r), V3: (m, p, r), W3: (p, n, r)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    T = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3)
    
    # Check all entries are 1
    return np.allclose(T, 1.0, atol=1e-9)


def _build_combined_52(n, m, p, U10, V10, W10):
    """Build rank-52: 6 Strassen blocks + rank-10 2x2x3 block."""
    n, m, p = 2, 2, 15
    blocks = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11)]
    
    all_U, all_V, all_W = [], [], []

    for block in blocks:
        k0, k1 = block
        terms = _strassen_terms(k0, k1)
        for (u_dict, v_dict, w_dict) in terms:
            u_entries = [(i*m+j, c) for (i,j), c in u_dict.items()]
            v_entries = [(j*p+k_idx, c) for (j,k_idx), c in v_dict.items()]
            w_entries = [(k_idx*n+i, c) for (k_idx,i), c in w_dict.items()]
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)

    # Add rank-10 terms for columns 12, 13, 14
    # U10: (4, 10) rows are i*m+j
    # V10: (6, 10) rows are j*3+k_local
    # W10: (6, 10) rows are k_local*n+i
    for r in range(10):
        u_entries = []
        v_entries = []
        w_entries = []
        
        for i in range(n):
            for j in range(m):
                val = U10[i*m+j, r]
                if val != 0:
                    u_entries.append((i*m+j, val))
        
        for j in range(m):
            for k_local in range(3):
                val = V10[j*3 + k_local, r]
                if val != 0:
                    k_global = 12 + k_local
                    v_entries.append((j*p + k_global, val))
        
        for k_local in range(3):
            for i in range(n):
                val = W10[k_local*n + i, r]
                if val != 0:
                    k_global = 12 + k_local
                    w_entries.append((k_global*n + i, val))
        
        all_U.append(u_entries)
        all_V.append(v_entries)
        all_W.append(w_entries)

    rank = len(all_U)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for c in range(rank):
        for row, val in all_U[c]:
            U[row][c] = val
        for row, val in all_V[c]:
            V[row][c] = val
        for row, val in all_W[c]:
            W[row][c] = val

    return U, V, W


def _find_2x2x3_rank10(deadline):
    """
    Search for rank-10 decomposition of 2x2x3 matrix multiplication.
    Uses structured initialization from Strassen + ALS refinement.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    def _verify2x2x3(U, V, W, tol=1e-9):
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=tol)
    
    def _als_step(U, V, W, max_iter=200):
        """ALS refinement for 2x2x3 tensor decomposition."""
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    B_mat = np.zeros((rank, p))
                    for r in range(rank):
                        for k in range(p):
                            B_mat[r, k] = V[j*p + k, r] * W[k*n + i, r]
                    try:
                        U[i*m + j, :] = np.linalg.lstsq(B_mat, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    C_mat = np.zeros((rank, n*m))
                    for r in range(rank):
                        for i in range(n):
                            C_mat[r, i*m + j] = U[i*m + j, r] * W[k*n + i, r]
                    try:
                        V[j*p + k, :] = np.linalg.lstsq(C_mat, np.ones(n*m), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    D_mat = np.zeros((rank, m))
                    for r in range(rank):
                        for j in range(m):
                            D_mat[r, j] = U[i*m + j, r] * V[j*p + k, r]
                    try:
                        W[k*n + i, :] = np.linalg.lstsq(D_mat, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check error
            U3 = U.reshape(n, m, -1)
            V3 = V.reshape(m, p, -1)
            W3 = W.reshape(p, n, -1)
            T = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3)
            error = np.sum((T - 1.0) ** 2)
            
            if error < 1e-14:
                break
        
        return U, V, W
    
    def _try_round_and_verify(U, V, W):
        """Try rounding to integers or half-integers and verify."""
        for mult in [1, 2, 4]:
            U_r = np.round(U * mult) / mult
            V_r = np.round(V * mult) / mult
            W_r = np.round(W * mult) / mult
            if _verify2x2x3(U_r, V_r, W_r):
                return U_r, V_r, W_r
        return None, None, None
    
    # Strategy 1: Start from Strassen 7-term solution for k=0,1 + 3 random terms for k=2
    # Strassen for 2x2x2 (k=0,1) embedded in 2x2x3:
    # The 7 Strassen terms only involve B[j,0] and B[j,1], not B[j,2]
    
    # Build initial U, V, W with Strassen in first 7 columns
    U = np.zeros((4, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))
    
    # Strassen terms for k=0,1 (from _strassen_terms)
    strassen = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(strassen):
        for (i,j), c in u_dict.items():
            U[i*m+j, r] = c
        for (j,k), c in v_dict.items():
            V[j*p+k, r] = c
        for (k,i), c in w_dict.items():
            W[k*n+i, r] = c
    
    # Try multiple initializations for the last 3 terms
    best_solution = None
    best_error = float('inf')
    
    np.random.seed(42)
    
    for restart in range(30):
        if time.time() > deadline - 5:
            break
        
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        
        # Randomize the last 3 terms
        for r in range(7, rank):
            U_curr[:, r] = np.random.randn(4) * 0.5
            V_curr[:, r] = np.random.randn(6) * 0.5
            W_curr[:, r] = np.random.randn(6) * 0.5
        
        # ALS refinement
        U_curr, V_curr, W_curr = _als_step(U_curr, V_curr, W_curr, max_iter=150)
        
        # Check error
        U3 = U_curr.reshape(n, m, -1)
        V3 = V_curr.reshape(m, p, -1)
        W3 = W_curr.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3)
        error = np.sum((T - 1.0) ** 2)
        
        if error < best_error:
            best_error = error
            best_solution = (U_curr.copy(), V_curr.copy(), W_curr.copy())
        
        # Try to round
        if error < 1e-8:
            result = _try_round_and_verify(U_curr, V_curr, W_curr)
            if result[0] is not None:
                return result
    
    # Strategy 2: Try a different structured initialization
    # Use schoolbook for k=0 (4 terms) + Strassen-like for k=1,2 (6 terms)
    U2 = np.zeros((4, rank))
    V2 = np.zeros((6, rank))
    W2 = np.zeros((6, rank))
    
    # 4 schoolbook terms for k=0
    for idx, (i, j) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
        r = idx
        U2[i*m+j, r] = 1
        V2[j*p+0, r] = 1
        W2[0*n+i, r] = 1
    
    # 6 terms for k=1,2 using a modified Strassen-like approach
    # Try to cover C[0,1], C[0,2], C[1,1], C[1,2] with 6 terms
    # This is a 2x2x2 problem (rows i=0,1; cols k=1,2) but with shared A rows
    
    # Let's just randomize and ALS
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        U_curr = U2.copy()
        V_curr = V2.copy()
        W_curr = W2.copy()
        
        for r in range(4, rank):
            U_curr[:, r] = np.random.randn(4) * 0.5
            V_curr[:, r] = np.random.randn(6) * 0.5
            W_curr[:, r] = np.random.randn(6) * 0.5
        
        U_curr, V_curr, W_curr = _als_step(U_curr, V_curr, W_curr, max_iter=150)
        
        U3 = U_curr.reshape(n, m, -1)
        V3 = V_curr.reshape(m, p, -1)
        W3 = W_curr.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3)
        error = np.sum((T - 1.0) ** 2)
        
        if error < best_error:
            best_error = error
            best_solution = (U_curr.copy(), V_curr.copy(), W_curr.copy())
        
        if error < 1e-8:
            result = _try_round_and_verify(U_curr, V_curr, W_curr)
            if result[0] is not None:
                return result
    
    # If we found a good floating-point solution, try harder to round
    if best_solution is not None and best_error < 1e-6:
        U_b, V_b, W_b = best_solution
        result = _try_round_and_verify(U_b, V_b, W_b)
        if result[0] is not None:
            return result
    
    return None, None, None


def _verify_2x2x3(U, V, W, tol=1e-9):
    """Verify 2x2x3 decomposition using tensor contraction."""
    n, m, p = 2, 2, 3
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=tol)
# EVOLVE-BLOCK-END
