import numpy as np
import time
import random

DIMENSIONS = [2, 2, 4]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Indexing:
    U[i*m+j, r]
    V[j*p+k, r]
    W[k*n+i, r]
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _construct_trivial_rank16():
    """
    Construct the trivial rank-16 decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank14_strassen():
    """
    Construct the known rank-14 decomposition using two independent Strassen 2x2x2 blocks.
    """
    n, m, p = 2, 2, 4
    rank = 14
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen 2x2x2 coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B10+B11)
    
    # U rows: 0=A00, 1=A01, 2=A10, 3=A11
    # V rows: 0=B00, 1=B01, 2=B10, 3=B11
    # W rows: 0=C00, 1=C01, 2=C10, 3=C11 (Note: W index is k*n+i, so for 2x2, k=0->0,1; k=1->2,3)
    # Wait, standard Strassen output mapping:
    # C00 = M1 + M3 - M4 - M7  (Indices in W: k=0, i=0 -> W[0])
    # C01 = M4 + M5            (Indices in W: k=0, i=1 -> W[1])
    # C10 = M2 + M6            (Indices in W: k=1, i=0 -> W[2])
    # C11 = M1 - M2 + M5 + M7  (Indices in W: k=1, i=1 -> W[3])
    
    # Let's define the Strassen terms for 2x2
    # Term 0: M1
    # Term 1: M2
    # Term 2: M3
    # Term 3: M4
    # Term 4: M5
    # Term 5: M6
    # Term 6: M7
    
    def set_strassen_block(offset_u, offset_v, offset_w, block_v_offset, block_w_offset):
        # U is shared structure for A, but we fill specific columns
        # V and W depend on the B and C block position
        
        # M1
        U[0, offset_u] += 1; U[3, offset_u] += 1
        V[0, offset_u] += 1; V[3, offset_u] += 1
        W[0, offset_u] += 1  # C00
        W[3, offset_u] += 1  # C11
        W[1, offset_u] -= 1  # C01
        W[2, offset_u] += 1  # C10
        
        # M2
        U[2, offset_u+1] += 1; U[3, offset_u+1] += 1
        V[0, offset_u+1] += 1
        W[2, offset_u+1] += 1  # C10
        W[3, offset_u+1] += 1  # C11
        
        # M3
        U[0, offset_u+2] += 1
        V[2, offset_u+2] += 1; V[3, offset_u+2] -= 1
        W[0, offset_u+2] += 1  # C00
        W[2, offset_u+2] -= 1  # C10
        
        # M4
        U[3, offset_u+3] += 1
        V[1, offset_u+3] += 1; V[0, offset_u+3] -= 1
        W[1, offset_u+3] += 1  # C01
        W[3, offset_u+3] -= 1  # C11
        
        # M5
        U[0, offset_u+4] += 1; U[1, offset_u+4] += 1
        V[3, offset_u+4] += 1
        W[1, offset_u+4] += 1  # C01
        W[3, offset_u+4] += 1  # C11
        
        # M6
        U[2, offset_u+5] += 1; U[0, offset_u+5] -= 1
        V[0, offset_u+5] += 1; V[1, offset_u+5] += 1
        W[0, offset_u+5] += 1  # C00
        W[2, offset_u+5] += 1  # C10
        
        # M7
        U[1, offset_u+6] += 1; U[3, offset_u+6] -= 1
        V[2, offset_u+6] += 1; V[3, offset_u+6] += 1
        W[1, offset_u+6] += 1  # C01
        W[3, offset_u+6] += 1  # C11

    # Block 1: B cols 0-1, C cols 0-1
    # V indices for block 1: j*4 + k where k in 0,1 -> 0,1, 4,5
    # W indices for block 1: k*2 + i where k in 0,1 -> 0,1, 2,3
    
    # We need to map the local 2x2 Strassen indices to the global 2x4 indices.
    # Local V: 0->0, 1->1, 2->4, 3->5
    # Local W: 0->0, 1->1, 2->2, 3->3
    
    # Let's just build the 2x2 matrices and place them.
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1
    U_s[0,0]=1; U_s[3,0]=1
    V_s[0,0]=1; V_s[3,0]=1
    W_s[0,0]=1; W_s[3,0]=1; W_s[1,0]=-1; W_s[2,0]=1
    
    # M2
    U_s[2,1]=1; U_s[3,1]=1
    V_s[0,1]=1
    W_s[2,1]=1; W_s[3,1]=1
    
    # M3
    U_s[0,2]=1
    V_s[2,2]=1; V_s[3,2]=-1
    W_s[0,2]=1; W_s[2,2]=-1
    
    # M4
    U_s[3,3]=1
    V_s[1,3]=1; V_s[0,3]=-1
    W_s[1,3]=1; W_s[3,3]=-1
    
    # M5
    U_s[0,4]=1; U_s[1,4]=1
    V_s[3,4]=1
    W_s[1,4]=1; W_s[3,4]=1
    
    # M6
    U_s[2,5]=1; U_s[0,5]=-1
    V_s[0,5]=1; V_s[1,5]=1
    W_s[0,5]=1; W_s[2,5]=1
    
    # M7
    U_s[1,6]=1; U_s[3,6]=-1
    V_s[2,6]=1; V_s[3,6]=1
    W_s[1,6]=1; W_s[3,6]=1
    
    # Place Block 1 (cols 0-1 of B, cols 0-1 of C)
    U[:, 0:7] = U_s
    # V mapping: local 0->0, 1->1, 2->4, 3->5
    V[0, 0:7] = V_s[0, 0:7]
    V[1, 0:7] = V_s[1, 0:7]
    V[4, 0:7] = V_s[2, 0:7]
    V[5, 0:7] = V_s[3, 0:7]
    # W mapping: local 0->0, 1->1, 2->2, 3->3
    W[0, 0:7] = W_s[0, 0:7]
    W[1, 0:7] = W_s[1, 0:7]
    W[2, 0:7] = W_s[2, 0:7]
    W[3, 0:7] = W_s[3, 0:7]
    
    # Place Block 2 (cols 2-3 of B, cols 2-3 of C)
    U[:, 7:14] = U_s
    # V mapping: local 0->2, 1->3, 2->6, 3->7
    V[2, 7:14] = V_s[0, 0:7]
    V[3, 7:14] = V_s[1, 0:7]
    V[6, 7:14] = V_s[2, 0:7]
    V[7, 7:14] = V_s[3, 0:7]
    # W mapping: local 0->4, 1->5, 2->6, 3->7
    W[4, 7:14] = W_s[0, 0:7]
    W[5, 7:14] = W_s[1, 0:7]
    W[6, 7:14] = W_s[2, 0:7]
    W[7, 7:14] = W_s[3, 0:7]
    
    return U.tolist(), V.tolist(), W.tolist()

def _search_global(n, m, p, rank, deadline, seed=0):
    """
    Global search using random restarts and local refinement (Simulated Annealing style).
    """
    start_time = time.time()
    np.random.seed(seed)
    
    best_score = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Precompute target tensor indices
    # We want to minimize sum ( sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] - 1 )^2
    
    # Number of equations
    num_eq = n * m * p
    
    # Initialize with random sparse matrices
    # Heuristic: start near the trivial rank-16 solution but compressed?
    # Or just random.
    
    for restart in range(20): # 20 restarts
        if time.time() > deadline - 5:
            break
            
        # Initialize
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        # Add some structure: try to make them sparse
        mask_U = (np.random.rand(n*m, rank) < 0.3)
        mask_V = (np.random.rand(m*p, rank) < 0.3)
        mask_W = (np.random.rand(p*n, rank) < 0.3)
        U *= mask_U
        V *= mask_V
        W *= mask_W
        
        # If all zeros, add a bit
        if np.all(U == 0) or np.all(V == 0) or np.all(W == 0):
            U = np.ones((n*m, rank)) * 0.1
            V = np.ones((m*p, rank)) * 0.1
            W = np.ones((p*n, rank)) * 0.1
            
        current_score = _compute_error(U, V, W, n, m, p)
        
        # Local search / Annealing
        T = 1.0
        for step in range(200):
            if time.time() > deadline - 5:
                break
                
            # Perturb
            dU = np.random.randn(n*m, rank) * 0.1
            dV = np.random.randn(m*p, rank) * 0.1
            dW = np.random.randn(p*n, rank) * 0.1
            
            new_U = U + dU
            new_V = V + dV
            new_W = W + dW
            
            new_score = _compute_error(new_U, new_V, new_W, n, m, p)
            
            if new_score < current_score or np.random.rand() < np.exp((current_score - new_score) / T):
                U, V, W = new_U, new_V, new_W
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                    # Check if we found an exact solution (within tolerance)
                    if current_score < 1e-8:
                        # Try to snap to half-integers
                        snapped_U = np.round(U * 2) / 2
                        snapped_V = np.round(V * 2) / 2
                        snapped_W = np.round(W * 2) / 2
                        
                        if _verify_decomposition(snapped_U.tolist(), snapped_V.tolist(), snapped_W.tolist(), n, m, p):
                            return snapped_U.tolist(), snapped_V.tolist(), snapped_W.tolist()
                        
                        # Maybe integers?
                        snapped_U_int = np.round(U)
                        snapped_V_int = np.round(V)
                        snapped_W_int = np.round(W)
                        if _verify_decomposition(snapped_U_int.tolist(), snapped_V_int.tolist(), snapped_W_int.tolist(), n, m, p):
                            return snapped_U_int.tolist(), snapped_V_int.tolist(), snapped_W_int.tolist()
            
            T *= 0.99 # Cooling
            
            # Early break if perfect
            if current_score < 1e-10:
                break
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_score

def _compute_error(U, V, W, n, m, p):
    """
    Compute the squared error of the tensor decomposition.
    """
    rank = U.shape[1]
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                u_row = U[i*m+j]
                v_row = V[j*p+k]
                w_row = W[k*n+i]
                # Vectorized dot product for speed
                val = np.dot(np.dot(u_row, v_row), w_row)
                error += (val - 1.0) ** 2
    return error

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    # 1. Get the known Rank 14 solution
    U14, V14, W14 = _construct_rank14_strassen()
    if _verify_decomposition(U14, V14, W14, n, m, p):
        best_result = {"U": U14, "V": V14, "W": W14}
        best_rank = 14
    else:
        # Fallback to Rank 16
        best_result = _construct_trivial_rank16()
        best_rank = 16

    # 2. Try to find Rank 13 or 12 using Global Search
    # Rank 13
    if best_rank > 13 and time.time() < deadline - 20:
        try:
            U13, V13, W13, err = _search_global(n, m, p, 13, deadline - 20, seed=42)
            if U13 is not None and _verify_decomposition(U13, V13, W13, n, m, p):
                best_result = {"U": U13, "V": V13, "W": W13}
                best_rank = 13
        except Exception:
            pass

    # Rank 12
    if best_rank > 12 and time.time() < deadline - 10:
        try:
            U12, V12, W12, err = _search_global(n, m, p, 12, deadline - 10, seed=123)
            if U12 is not None and _verify_decomposition(U12, V12, W12, n, m, p):
                best_result = {"U": U12, "V": V12, "W": W12}
                best_rank = 12
        except Exception:
            pass
            
    # Rank 11 (Long shot)
    if best_rank > 11 and time.time() < deadline - 5:
        try:
            U11, V11, W11, err = _search_global(n, m, p, 11, deadline - 5, seed=777)
            if U11 is not None and _verify_decomposition(U11, V11, W11, n, m, p):
                best_result = {"U": U11, "V": V11, "W": W11}
                best_rank = 11
        except Exception:
            pass

    return best_result

if __name__ == "__main__":
    res = construct()
    print(res)
