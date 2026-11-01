"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import sys

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np

def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Left block (cols 0,1) - Strassen
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right block (cols 2,3) - Strassen
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = 2, 2, 4
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    if U.shape != (n * m, r):
        return False
    if V.shape != (m * p, r):
        return False
    if W.shape != (p * n, r):
        return False
    
    # Check all 16 required entries equal 1
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = 0.0
                for rr in range(r):
                    val += U[i * m + k, rr] * V[k * p + j, rr] * W[j * n + i, rr]
                if abs(val - 1.0) > tol:
                    return False
    
    # Check all other entries are 0
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                a = i // m
                b = i % m
                bp = j // p
                c = j % p
                cp = k // n
                ap = k % n
                if a == ap and b == bp:
                    continue
                val = 0.0
                for rr in range(r):
                    val += U[i, rr] * V[j, rr] * W[k, rr]
                if abs(val) > tol:
                    return False
    
    return True


def _tensor_residual(U, V, W):
    """Compute full tensor residual: T[i,j,k] - delta(i,j,k)."""
    n, m, p = 2, 2, 4
    r = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for rr in range(r):
        T += np.einsum('ir,jr,kr->ijk', U[:, rr], V[:, rr], W[:, rr])
    # Target: T[i,j,k] = 1 iff i,j,k form a valid matmul term
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                target[i * m + k, k * p + j, j * n + i] = 1.0
    return T - target


def _als_rank13(U, V, W, max_iter=100, tol=1e-10, deadline=None):
    """Fast ALS for rank-13. Returns (U,V,W, residual_norm)."""
    n, m, p = 2, 2, 4
    r = U.shape[1]
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    
    # Precompute index arrays for vectorized updates
    # For updating U[:,rr]: U[i*m+k, rr] = sum_j target[i,j,k] * V[k*p+j, rr] * W[j*n+i, rr]
    # where target[i*m+k, k*p+j, j*n+i] = 1
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        for rr in range(r):
            # Compute residual without term rr
            T_other = np.zeros((n * m, m * p, p * n))
            for rr2 in range(r):
                if rr2 == rr:
                    continue
                T_other += np.einsum('ir,jr,kr->ijk', U[:, rr2], V[:, rr2], W[:, rr2])
            target = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        target[i * m + k, k * p + j, j * n + i] = 1.0
            residual = target - T_other
            
            # U[i*m+k, rr] = sum_j residual[i*m+k, k*p+j, j*n+i] * V[k*p+j, rr] * W[j*n+i, rr]
            for i in range(n):
                for k in range(m):
                    val = 0.0
                    for j in range(p):
                        val += residual[i * m + k, k * p + j, j * n + i] * V[k * p + j, rr] * W[j * n + i, rr]
                    U[i * m + k, rr] = val
        
        # Update V
        for rr in range(r):
            T_other = np.zeros((n * m, m * p, p * n))
            for rr2 in range(r):
                if rr2 == rr:
                    continue
                T_other += np.einsum('ir,jr,kr->ijk', U[:, rr2], V[:, rr2], W[:, rr2])
            target = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        target[i * m + k, k * p + j, j * n + i] = 1.0
            residual = target - T_other
            
            for k in range(m):
                for j in range(p):
                    val = 0.0
                    for i in range(n):
                        val += residual[i * m + k, k * p + j, j * n + i] * U[i * m + k, rr] * W[j * n + i, rr]
                    V[k * p + j, rr] = val
        
        # Update W
        for rr in range(r):
            T_other = np.zeros((n * m, m * p, p * n))
            for rr2 in range(r):
                if rr2 == rr:
                    continue
                T_other += np.einsum('ir,jr,kr->ijk', U[:, rr2], V[:, rr2], W[:, rr2])
            target = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        target[i * m + k, k * p + j, j * n + i] = 1.0
            residual = target - T_other
            
            for j in range(p):
                for i in range(n):
                    val = 0.0
                    for k in range(m):
                        val += residual[i * m + k, k * p + j, j * n + i] * U[i * m + k, rr] * V[k * p + j, rr]
                    W[j * n + i, rr] = val
        
        # Check convergence
        T = np.zeros((n * m, m * p, p * n))
        for rr in range(r):
            T += np.einsum('ir,jr,kr->ijk', U[:, rr], V[:, rr], W[:, rr])
        target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    target[i * m + k, k * p + j, j * n + i] = 1.0
        res = T - target
        res_norm = np.linalg.norm(res)
        if res_norm < tol:
            break
    
    return U, V, W, res_norm


def _search_rank13(deadline):
    """Search for rank-13 decomposition using multiple strategies."""
    n, m, p = 2, 2, 4
    target_rank = 13
    
    base = _build_rank14()
    U14 = np.array(base['U'], dtype=float)
    V14 = np.array(base['V'], dtype=float)
    W14 = np.array(base['W'], dtype=float)
    
    best_result = None
    best_residual = np.inf
    
    # Strategy 1: Merge pairs from rank-14
    left_terms = list(range(7))
    right_terms = list(range(7, 14))
    
    # Try all cross-block merges (7*7=49) and same-block merges
    merge_pairs = []
    for li in left_terms:
        for ri in right_terms:
            merge_pairs.append((li, ri))
    for block in [left_terms, right_terms]:
        for a in range(len(block)):
            for b in range(a + 1, len(block)):
                merge_pairs.append((block[a], block[b]))
    
    np.random.seed(42)
    # Shuffle and limit to fit in time
    random.shuffle(merge_pairs)
    
    for li, ri in merge_pairs:
        if time.time() > deadline - 15:
            break
        
        U13 = np.zeros((n * m, target_rank))
        V13 = np.zeros((m * p, target_rank))
        W13 = np.zeros((p * n, target_rank))
        
        kept = [t for t in range(14) if t != li and t != ri]
        for new_idx, old_idx in enumerate(kept):
            U13[:, new_idx] = U14[:, old_idx]
            V13[:, new_idx] = V14[:, old_idx]
            W13[:, new_idx] = W14[:, old_idx]
        
        merge_idx = target_rank - 1
        U13[:, merge_idx] = 0.5 * (U14[:, li] + U14[:, ri])
        V13[:, merge_idx] = 0.5 * (V14[:, li] + V14[:, ri])
        W13[:, merge_idx] = 0.5 * (W14[:, li] + W14[:, ri])
        
        # Small perturbation
        U13 += 0.005 * np.random.randn(*U13.shape)
        V13 += 0.005 * np.random.randn(*V13.shape)
        W13 += 0.005 * np.random.randn(*W13.shape)
        
        U13, V13, W13, res_norm = _als_rank13(U13, V13, W13, max_iter=50, deadline=deadline - 10)
        
        if res_norm < best_residual:
            best_residual = res_norm
            if res_norm < 1e-8:
                U13 = np.round(U13, 10)
                V13 = np.round(V13, 10)
                W13 = np.round(W13, 10)
                U_list = [[float(x) for x in row] for row in U13]
                V_list = [[float(x) for x in row] for row in V13]
                W_list = [[float(x) for x in row] for row in W13]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                return best_result
    
    # Strategy 2: Random integer-valued search
    if time.time() < deadline - 20:
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(1000 + attempt)
            # Sparse random initialization
            U13 = np.zeros((n * m, target_rank))
            V13 = np.zeros((m * p, target_rank))
            W13 = np.zeros((p * n, target_rank))
            
            for rr in range(target_rank):
                # Each term has a few non-zero entries
                u_nonzero = np.random.choice(n * m, size=np.random.randint(2, 5), replace=False)
                for idx in u_nonzero:
                    U13[idx, rr] = np.random.choice([-1, 1])
                
                v_nonzero = np.random.choice(m * p, size=np.random.randint(2, 5), replace=False)
                for idx in v_nonzero:
                    V13[idx, rr] = np.random.choice([-1, 1])
                
                w_nonzero = np.random.choice(p * n, size=np.random.randint(2, 5), replace=False)
                for idx in w_nonzero:
                    W13[idx, rr] = np.random.choice([-1, 1])
            
            U13, V13, W13, res_norm = _als_rank13(U13, V13, W13, max_iter=30, deadline=deadline - 10)
            
            if res_norm < best_residual:
                best_residual = res_norm
                if res_norm < 1e-8:
                    U13 = np.round(U13, 10)
                    V13 = np.round(V13, 10)
                    W13 = np.round(W13, 10)
                    U_list = [[float(x) for x in row] for row in U13]
                    V_list = [[float(x) for x in row] for row in V13]
                    W_list = [[float(x) for x in row] for row in W13]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    return best_result
    
    return best_result


def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    # Try to find rank-13
    result = _search_rank13(deadline)
    
    if result is not None:
        if _verify_decomposition(result['U'], result['V'], result['W']):
            return result
    
    # Fall back to rank-14
    return _build_rank14()
# EVOLVE-BLOCK-END
