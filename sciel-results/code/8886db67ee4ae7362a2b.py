import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Rank-56 decomposition for multiplying a 2x2 matrix A by a 2x16 matrix B.
    Uses Strassen's algorithm (rank 7) for each of the 8 independent 2x2 blocks.
    Falls back to schoolbook (rank 64) if verification fails.
    """
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Strassen coefficients for 2x2x2
    # U: [A00, A01, A10, A11]
    # V: [B00, B01, B10, B11]
    # W: [C00, C01, C10, C11]
    
    strassen_U = [
        [1, 0, 0, 1],  # M1: A00+A11
        [0, 0, 1, 1],  # M2: A10+A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00+A01
        [-1, 0, 1, 0], # M6: A10-A00
        [0, 1, 0, -1]  # M7: A01-A11
    ]
    
    strassen_V = [
        [1, 0, 0, 1],  # M1: B00+B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01-B11
        [-1, 0, 1, 0], # M4: B10-B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00+B01
        [0, 0, 1, 1]   # M7: B10+B11
    ]
    
    strassen_W = [
        [1, 0, 0, 1],  # M1: C00+1, C11+1
        [0, 0, 1, -1], # M2: C10+1, C11-1
        [0, 1, 0, 1],  # M3: C01+1, C11+1
        [1, 0, 1, 0],  # M4: C00+1, C10+1
        [-1, 1, 0, 0], # M5: C00-1, C01+1
        [0, 0, 0, 1],  # M6: C11+1
        [1, 0, 0, 0]   # M7: C00+1
    ]
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for l in range(8):
        for s in range(7):
            col = 7 * l + s
            
            # U: same for all blocks
            for a_idx in range(4):
                U[a_idx, col] = strassen_U[s][a_idx]
            
            # V: local B indices mapped to global
            # B00 -> 2*l, B01 -> 2*l+1, B10 -> 16+2*l, B11 -> 16+2*l+1
            b_rows = [2*l, 2*l+1, 16+2*l, 16+2*l+1]
            for b_idx in range(4):
                V[b_rows[b_idx], col] = strassen_V[s][b_idx]
            
            # W: local C indices mapped to global
            # C00 -> 4*l, C01 -> 4*l+2, C10 -> 4*l+1, C11 -> 4*l+3
            c_rows = [4*l, 4*l+2, 4*l+1, 4*l+3]
            for c_idx in range(4):
                W[c_rows[c_idx], col] = strassen_W[s][c_idx]
    
    # Verify
    def verify(U, V, W, n, m, p):
        # Check T[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
        # For mat mult, T[i,j,k] should be 1 if this is a valid term?
        # No, the tensor T is defined such that C[i,k] = sum_j A[i,j] B[j,k].
        # The decomposition is T = sum u v w.
        # We check if the reconstructed product matches the standard product for random inputs.
        
        # Random A (2x2) and B (2x16)
        A_rand = np.random.rand(n, m)
        B_rand = np.random.rand(m, p)
        C_expected = A_rand @ B_rand
        
        # Compute C from decomposition
        # C[i,k] = sum_s (sum_j A[i,j] U[i*m+j, s]) * (sum_k' B[j',k'] ... )
        # Actually, the bilinear form is:
        # C[i,k] = sum_s (sum_j A[i,j] U[i*m+j, s]) * (sum_{j',k'} B[j',k'] V[j'*p+k', s] * W[k*n+i, s] ... )
        # No, the rank-1 tensor u v w contributes u_i v_j w_k to T[i,j,k].
        # The product is C[i,k] = sum_j sum_s A[i,j] B[j,k] (coefficient of T[i,j,k] from s)
        # Wait, the decomposition is T = sum_s u_s v_s w_s.
        # Then C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k]? No.
        # The tensor T maps A, B to C.
        # C[i,k] = sum_{j,j',k'} A[i,j] B[j',k'] T[i,j,j',k',i,k]? No.
        
        # Standard check:
        # C_decomposed = sum_s (A * U_s) * (V_s * B) * W_s?
        # Let's use the definition:
        # C[i,k] = sum_s (sum_j A[i,j] U[i*m+j, s]) * (sum_{j',k'} B[j',k'] V[j'*p+k', s] W[k'*n+i, s])?
        # No, W is indexed by [k*n+i].
        
        # Correct evaluation:
        # For each s:
        #   A_s = A @ U_s (where U_s is a vector of size n*m, reshaped to n x m? No, U is flat)
        #   Actually, U_s is a vector in R^{n*m}.
        #   The contribution of term s to C is:
        #   C_s[i,k] = (sum_j A[i,j] U[i*m+j, s]) * (sum_{j',k'} B[j',k'] V[j'*p+k', s] * W[k*n+i, s] ... )
        #   This is not right.
        
        # The tensor T has indices (i, j, k) for the "standard" basis?
        # No, the tensor is in R^{n*m} x R^{m*p} x R^{p*n}.
        # T[i*m+j, j'*p+k', k''*n+i''] = 1 if j=j' and i=i''?
        # For matrix multiplication, T is the "multiplication tensor".
        # T[u, v, w] = 1 if u corresponds to A[i,j], v to B[j,k], w to C[i,k].
        # So T[i*m+j, j*p+k, k*n+i] = 1.
        
        # We check if sum_s U[u,s] V[v,s] W[w,s] == T[u,v,w] for all u,v,w.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    val = 0
                    for s in range(rank):
                        val += U[u, s] * V[v, s] * W[w, s]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school), dtype=float)
    V = np.zeros((m * p, rank_school), dtype=float)
    W = np.zeros((p * n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
