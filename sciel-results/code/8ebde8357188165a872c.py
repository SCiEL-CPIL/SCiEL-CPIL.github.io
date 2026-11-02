"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize best with schoolbook
    U0 = np.zeros((n*m, best_rank))
    V0 = np.zeros((m*p, best_rank))
    W0 = np.zeros((p*n, best_rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, c] = 1.0
                V0[j*p+k, c] = 1.0
                W0[k*n+i, c] = 1.0
                c += 1
    best_U = U0
    best_V = V0
    best_W = W0
    
    # Try to find a lower rank decomposition using random initialization + ALS
    # Known bounds: rank(3x3x8) is likely around 60-70. Schoolbook is 72.
    # Let's try ranks from 64 down to 60.
    
    np.random.seed(42)
    
    for rank in range(64, 59, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random inits for this rank
        found = False
        max_iters = 10000
        success = False
        
        for trial in range(10):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # ALS: Alternate solving for U, V, W.
            # The tensor T has entries T[i, j, k] = 1 if (i,j,k) is a valid multiplication term?
            # No, the tensor IS the multiplication map.
            # T is a 3x3x8 tensor where T[i,j,k] = 1 for all i,j,k?
            # Wait, matrix multiplication C = A * B where A is 3x3 and B is 3x8.
            # C[i,k] = sum_j A[i,j] B[j,k].
            # The bilinear form is C[i,k] = <A_flat, T_{i,k} B_flat>?
            # The tensor T_{i,j,k} is the structure constant.
            # In standard matrix multiplication, T_{i,j,k} = 1 for all i,j,k? No.
            # The tensor is T such that T_{i,k,j} = 1 if we view it as C_{i,k} = sum_j T_{i,k,j} A_{i,j} B_{j,k}.
            # So T_{i,k,j} = 1 for all i,k,j.
            # Let's check the interface:
            # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
            # Indices:
            # U row index: i*m + j (i in 0..n-1, j in 0..m-1)
            # V row index: j*p + k (j in 0..m-1, k in 0..p-1)
            # W row index: k*n + i (k in 0..p-1, i in 0..n-1)
            #
            # Decomposition:
            # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = T_{i,j,k}
            #
            # For matrix multiplication C[i,k] = sum_j A[i,j] B[j,k].
            # The "tensor" is 1 for all i,j,k?
            # Yes, because every A[i,j] multiplies every B[j,k] to contribute to C[i,k].
            # So T_{i,j,k} = 1 for all i, j, k.
            
            T = np.ones((n, m, p))
            
            # ALS Loop
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                    
                # Solve for W (fixed U, V)
                # Equation: sum_c (U[i*m+j, c] V[j*p+k, c]) W[k*n+i, c] = 1
                # This is linear in W.
                # Let's flatten the indices.
                # Variables: W_flat of size p*n*rank
                # Equations: n*m*p equations.
                # Matrix A_w of size (n*m*p) x (p*n*rank)
                # A_w[idx, (k*n+i)*rank + c] = U[i*m+j, c] * V[j*p+k, c]
                # where idx is flat index of (i,j,k)
                
                # This matrix is huge: 72 x (24*rank). For rank=64, 72 x 1536.
                # We can build it efficiently.
                
                # Actually, we can exploit structure.
                # Let's just build the full matrix for simplicity if it's not too slow.
                # 72 * 1536 ~ 110k floats. Building it takes time.
                
                # Alternative: Use vectorized numpy.
                # U is (9, R), V is (24, R), W is (24, R).
                
                # Let's construct A_w.
                # i: 0..2, j: 0..2, k: 0..7
                # Row index for (i,j,k): i*m*p + j*p + k ? No, let's use linear index in T.
                # Let's define a helper to get linear index of (i,j,k) in T.
                # T is (n, m, p). Index = i*m*p + j*p + k.
                
                # A_w row (i,j,k) has non-zeros at cols (k*n+i, c) for c in 0..R-1.
                # Value is U[i*m+j, c] * V[j*p+k, c].
                
                # Let's build A_w as a sparse matrix?
                # Or just dense since it's small.
                
                # Initialize A_w
                A_w = np.zeros((n*m*p, p*n*rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row_idx = i*m*p + j*p + k
                            # W index is k*n + i
                            w_base = (k*n + i) * rank
                            u_val = U[i*m + j, :]
                            v_val = V[j*p + k, :]
                            A_w[row_idx, w_base:w_base+rank] = u_val * v_val
                
                b_w = np.ones(n*m*p)
                
                # Solve min ||A_w x - b_w||^2
                # x is W flattened.
                try:
                    W_flat, residuals, rank_w, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
                    W = W_flat.reshape(p*n, rank)
                except:
                    continue
                
                # Solve for U (fixed V, W)
                # Equation: sum_c U[i*m+j, c] (V[j*p+k, c] W[k*n+i, c]) = 1
                # A_u row (i,j,k) non-zeros at (i*m+j, c).
                # Value is V[j*p+k, c] * W[k*n+i, c].
                
                A_u = np.zeros((n*m*p, n*m*rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row_idx = i*m*p + j*p + k
                            u_base = (i*m + j) * rank
                            v_val = V[j*p + k, :]
                            w_val = W[k*n + i, :]
                            A_u[row_idx, u_base:u_base+rank] = v_val * w_val
                
                b_u = np.ones(n*m*p)
                
                try:
                    U_flat, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                    U = U_flat.reshape(n*m, rank)
                except:
                    continue
                
                # Solve for V (fixed U, W)
                # Equation: sum_c (U[i*m+j, c] W[k*n+i, c]) V[j*p+k, c] = 1
                # A_v row (i,j,k) non-zeros at (j*p+k, c).
                # Value is U[i*m+j, c] * W[k*n+i, c].
                
                A_v = np.zeros((n*m*p, m*p*rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row_idx = i*m*p + j*p + k
                            v_base = (j*p + k) * rank
                            u_val = U[i*m + j, :]
                            w_val = W[k*n + i, :]
                            A_v[row_idx, v_base:v_base+rank] = u_val * w_val
                
                b_v = np.ones(n*m*p)
                
                try:
                    V_flat, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                    V = V_flat.reshape(m*p, rank)
                except:
                    continue
                
                # Check error
                # Compute tensor from U, V, W
                # T_comp[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                
                # Vectorize check
                # Reshape U, V, W
                # U is (n*m, R) -> (n, m, R)
                U_3d = U.reshape(n, m, rank)
                V_3d = V.reshape(m, p, rank)
                W_3d = W.reshape(p, n, rank)
                
                # T_comp = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                # This is a tensor contraction.
                # np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
                
                T_comp = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
                error = np.linalg.norm(T_comp - T)
                
                if error < 1e-8:
                    # Found a valid decomposition!
                    success = True
                    break
                
                if time.time() > deadline - 5:
                    break
        
        if success:
            # Convert to list of lists
            best_U = U
            best_V = V
            best_W = W
            best_rank = rank
            # We found a valid decomposition for this rank.
            # Can we go lower? The loop continues to lower ranks.
            # But we should return the best (lowest rank) we can find within time.
            # The loop iterates rank downwards, so if we break out of the trial loop with success,
            # we update best. Then the outer loop tries lower rank.
            # If we don't find lower, we return the best found.
    
    # If no better than schoolbook found, return schoolbook.
    # Otherwise return the best found.
    
    # Ensure return type is correct: dict with 'U', 'V', 'W' as lists?
    # The task says "factor matrices of shapes ... with integer/half-integer entries".
    # It doesn't explicitly say list of lists, but the seed program returns list of lists.
    # Let's check the seed: `U = [[0] * rank for _ in range(n * m)]`.
    # So it expects lists.
    
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

# EVOLVE-BLOCK-END
