import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    # A is n x m = 2x4, B is m x p = 4x4, C is n x p = 2x4
    # Output convention: W[k*n + i, r] where i is row of C, k is col of C
    
    def schoolbook_solution():
        rank = n * m * p  # 32
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_solution(U, V, W, max_samples=50):
        """
        Verify that the tensor decomposition exactly reproduces matrix multiplication.
        T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        We check this by evaluating the bilinear map on random matrices.
        """
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        np.random.seed(42)
        for _ in range(max_samples):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            
            # Compute C via tensor:
            # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r])
            # This can be computed as:
            # C = sum_r (A * U_r) * (B * V_r) * W_r  -- wait, this is not quite right dimensionally if we just multiply.
            # Let's stick to the explicit sum for correctness in verification, or use einsum.
            # T is a tensor of shape (n*m, m*p, p*n).
            # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
            # The reconstructed tensor T_rec[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # Note: The indices in U, V, W correspond to flattened (i,j), (j,k), (k,i).
            # Let's reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank).
            # Then T_rec[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
            
            U_3d = U_np.reshape(n, m, rank)
            V_3d = V_np.reshape(m, p, rank)
            W_3d = W_np.reshape(p, n, rank)
            
            # Compute C[i, k] = sum_j A[i, j] B[j, k] T_rec[i, j, k]
            # C[i, k] = sum_j A[i, j] B[j, k] sum_r U[i, j, r] V[j, k, r] W[k, i, r]
            # C[i, k] = sum_r W[k, i, r] (sum_j A[i, j] U[i, j, r]) (sum_j B[j, k] V[j, k, r])
            
            C_tensor = np.zeros((n, p))
            for r in range(rank):
                a_proj = np.zeros(n)
                for i in range(n):
                    for j in range(m):
                        a_proj[i] += A[i, j] * U_3d[i, j, r]
                b_proj = np.zeros(p)
                for k in range(p):
                    for j in range(m):
                        b_proj[k] += B[j, k] * V_3d[j, k, r]
                for i in range(n):
                    for k in range(p):
                        C_tensor[i, k] += W_3d[k, i, r] * a_proj[i] * b_proj[k]
            
            if not np.allclose(C_tensor, C_true, atol=1e-8, rtol=1e-8):
                return False
        return True

    def strassen_decomposition():
        """
        Rank-28 decomposition using Strassen for 2x2 blocks.
        """
        rank = 28
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        def add_strassen(col_offset, A_block, B_block, C_block_W_indices):
            a00_idx = A_block[(0,0)]
            a01_idx = A_block[(0,1)]
            a10_idx = A_block[(1,0)]
            a11_idx = A_block[(1,1)]
            
            b00_idx = B_block[(0,0)]
            b01_idx = B_block[(0,1)]
            b10_idx = B_block[(1,0)]
            b11_idx = B_block[(1,1)]
            
            c00_w = C_block_W_indices[(0,0)]
            c01_w = C_block_W_indices[(0,1)]
            c10_w = C_block_W_indices[(1,0)]
            c11_w = C_block_W_indices[(1,1)]
            
            # M1 = (A00+A11)*(B00+B11)
            U[a00_idx][col_offset] += 1.0
            U[a11_idx][col_offset] += 1.0
            V[b00_idx][col_offset] += 1.0
            V[b11_idx][col_offset] += 1.0
            W[c00_w][col_offset] += 1.0
            W[c11_w][col_offset] += 1.0
            
            # M2 = (A10+A11)*B00
            U[a10_idx][col_offset+1] += 1.0
            U[a11_idx][col_offset+1] += 1.0
            V[b00_idx][col_offset+1] += 1.0
            W[c10_w][col_offset+1] += 1.0
            
            # M3 = A00*(B10-B11)
            U[a00_idx][col_offset+2] += 1.0
            V[b10_idx][col_offset+2] += 1.0
            V[b11_idx][col_offset+2] += -1.0
            W[c01_w][col_offset+2] += 1.0
            W[c11_w][col_offset+2] += 1.0
            
            # M4 = A11*(B01-B00)
            U[a11_idx][col_offset+3] += 1.0
            V[b01_idx][col_offset+3] += 1.0
            V[b00_idx][col_offset+3] += -1.0
            W[c00_w][col_offset+3] += 1.0
            W[c10_w][col_offset+3] += 1.0
            
            # M5 = (A00+A01)*B11
            U[a00_idx][col_offset+4] += 1.0
            U[a01_idx][col_offset+4] += 1.0
            V[b11_idx][col_offset+4] += 1.0
            W[c00_w][col_offset+4] += -1.0
            W[c01_w][col_offset+4] += 1.0
            
            # M6 = (A10-A00)*(B00+B01)
            U[a10_idx][col_offset+5] += 1.0
            U[a00_idx][col_offset+5] += -1.0
            V[b00_idx][col_offset+5] += 1.0
            V[b01_idx][col_offset+5] += 1.0
            W[c11_w][col_offset+5] += 1.0
            
            # M7 = (A01-A11)*(B10+B11)
            U[a01_idx][col_offset+6] += 1.0
            U[a11_idx][col_offset+6] += -1.0
            V[b10_idx][col_offset+6] += 1.0
            V[b11_idx][col_offset+6] += 1.0
            W[c00_w][col_offset+6] += 1.0
            W[c11_w][col_offset+6] += -1.0
            
            return col_offset + 7
        
        A1 = {(0,0): 0, (0,1): 1, (1,0): 2, (1,1): 3}
        A2 = {(0,0): 4, (0,1): 5, (1,0): 6, (1,1): 7}
        
        B11 = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
        B12 = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
        B21 = {(0,0): 8, (0,1): 9, (1,0): 12, (1,1): 13}
        B22 = {(0,0): 10, (0,1): 11, (1,0): 14, (1,1): 15}
        
        C1_W = {(0,0): 0, (0,1): 2, (1,0): 1, (1,1): 3}
        C2_W = {(0,0): 4, (0,1): 6, (1,0): 5, (1,1): 7}
        
        col = 0
        col = add_strassen(col, A1, B11, C1_W)
        col = add_strassen(col, A2, B21, C1_W)
        col = add_strassen(col, A1, B12, C2_W)
        col = add_strassen(col, A2, B22, C2_W)
        
        return {"U": U, "V": V, "W": W}

    def numerical_search(rank):
        """
        Attempt to find a decomposition of the given rank using numerical optimization.
        We initialize with random vectors and minimize the error between the reconstructed
        tensor and the true matrix multiplication tensor.
        """
        if time.time() > deadline:
            return None
            
        # True tensor T of shape (n*m, m*p, p*n)
        # T[i*m+j, j*p+k, k*n+i] = 1 if i,j,k are valid indices for matmul?
        # No, the tensor for matrix multiplication M_{n,m,p} has entries:
        # M[i, j, k] = 1 if we are computing C[i,k] = sum_j A[i,j]B[j,k].
        # In the standard tensor decomposition context, the tensor is defined such that:
        # C[i,k] = sum_{j,r} A[i,j] B[j,k] T[i,j,k]
        # For standard matmul, T[i,j,k] = 1 for all valid i,j,k.
        # Wait, if T[i,j,k]=1, then C[i,k] = sum_j A[i,j]B[j,k] * 1. This is correct.
        # So the tensor to decompose is simply a tensor of all ones of shape (n*m, m*p, p*n)
        # BUT the indices are coupled.
        # Let's define the target tensor T_target with shape (n*m, m*p, p*n).
        # T_target[idx_u, idx_v, idx_w] = 1 if the indices correspond to a valid (i,j,k) triple.
        # idx_u = i*m + j
        # idx_v = j*p + k
        # idx_w = k*n + i
        # If such i,j,k exist, T_target = 1. Otherwise 0.
        # Note: For any i in [0,n), j in [0,m), k in [0,p), the triple is valid.
        # So T_target is 1 at positions (i*m+j, j*p+k, k*n+i) and 0 elsewhere.
        
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i*m + j
                    idx_v = j*p + k
                    idx_w = k*n + i
                    T_target[idx_u, idx_v, idx_w] = 1.0

        # We want U, V, W such that sum_r U[:,r] V[:,r] W[:,r] approx T_target
        # This is a tensor completion/rank minimization problem.
        # We can use a simple iterative method:
        # Initialize U, V, W randomly.
        # Iterate:
        # 1. Fix V, W, update U via least squares for each column r? No, U is shared across r.
        # Actually, for a fixed rank R, we can use Alternating Least Squares (ALS) for tensor decomposition.
        # Reshape T_target to a 3D tensor.
        # U (n*m, R), V (m*p, R), W (p*n, R).
        # T_rec = sum_r U[:,r] outer V[:,r] outer W[:,r]
        
        # Initialize
        np.random.seed(0)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # To speed up convergence and avoid local minima, we can use a "soft" rank minimization
        # or just try several random initializations.
        # Given the time limit, we will try a few iterations of ALS.
        
        best_U, best_V, best_W = None, None, None
        best_error = float('inf')
        
        for init in range(5):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(init)
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Normalize columns to prevent scale drift
            U = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-9)
            V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-9)
            W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-9)
            
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                    
                # Update U
                # T_target reshaped to (n*m, m*p*p*n)
                # For each r, we want U[:,r] to minimize || T_target - sum_s U_s V_s W_s ||
                # This is complex. Let's use a simpler gradient descent on the objective:
                # F(U,V,W) = || T_target - sum_r U[:,r] V[:,r] W[:,r] ||_F^2
                
                # Compute current error
                T_rec = np.zeros_like(T_target)
                for r in range(rank):
                    T_rec += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
                error = np.linalg.norm(T_target - T_rec, 'fro')
                
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if error < 1e-6:
                    break
                
                # Gradient Descent
                # dF/dU[:,r] = -2 * sum_s (V[:,s] outer W[:,s]) * (T_target - T_rec) ... contracted?
                # Let E = T_target - T_rec
                # dF/dU[i, r] = -2 * sum_{j,k} E[i,j,k] V[j,r] W[k,r]  -- NO.
                # T_rec[i,j,k] = sum_s U[i,s] V[j,s] W[k,s]
                # d T_rec[i,j,k] / d U[i,r] = V[j,r] W[k,r]
                # dF/dU[i,r] = -2 * sum_{j,k} E[i,j,k] V[j,r] W[k,r]
                
                E = T_target - T_rec
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for r in range(rank):
                    # grad_U[:, r]
                    # sum_{j,k} E[:,j,k] * V[j,r] * W[k,r]
                    # This is a contraction of E with V[:,r] and W[:,r]
                    # Let's compute this efficiently
                    # temp = np.einsum('ijk,j,k->i', E, V[:,r], W[:,r])
                    grad_U[:, r] = np.einsum('ijk,j,k->i', E, V[:,r], W[:,r])
                    grad_V[:, r] = np.einsum('ijk,i,k->j', E, U[:,r], W[:,r])
                    grad_W[:, r] = np.einsum('ijk,i,j->k', E, U[:,r], V[:,r])
                    
                # Simple gradient step
                lr = 0.01
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Renormalize to keep scale stable
                U = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-9)
                V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-9)
                W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-9)
                
                # Rescale to account for normalization changing the magnitude
                # This is a heuristic. Ideally we should optimize the scale as well.
                # Let's just hope the gradient handles it or the normalization is mild.
                # Actually, normalizing after gradient step can distort the direction.
                # Let's try without normalization for the gradient step, but monitor norm.
                
            # After gradient descent, we can try to "clean up" the solution by setting small values to 0
            # and checking if it's still valid. But for now, return the best found.
            
        if best_U is not None and best_error < 1e-5:
            # Round to nearest 0.5 if possible? The problem allows half-integers.
            # Let's just return the float values. The evaluator checks exactness.
            # If the error is < 1e-5, it might be close to integers.
            # Let's try rounding to nearest 0.5
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            # Verify rounded solution
            if verify_solution(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), max_samples=10):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            
            # If rounding fails, verify the unrounded one (might be exact enough for float comparison?)
            # The task says "exactly decompose". Usually implies rational/integer.
            # If we can't get exact, we might fail.
            if verify_solution(best_U.tolist(), best_V.tolist(), best_W.tolist(), max_samples=10):
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                
        return None

    # Strategy:
    # 1. Try known rank 28 (Strassen).
    # 2. Try numerical search for rank 24, 20, 16, etc.
    # 3. Return best found.
    
    best_sol = None
    best_rank = float('inf')
    
    # Try Strassen (rank 28)
    try:
        strassen_sol = strassen_decomposition()
        if verify_solution(strassen_sol["U"], strassen_sol["V"], strassen_sol["W"], max_samples=50):
            best_sol = strassen_sol
            best_rank = 28
    except Exception:
        pass
    
    # Try numerical search for lower ranks
    # Rank 24 is a candidate if the rectangular structure allows it.
    # Rank 20, 16 are more aggressive.
    
    for target_rank in [24, 20, 16]:
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            continue
            
        print(f"Attempting numerical search for rank {target_rank}...")
        try:
            sol = numerical_search(target_rank)
            if sol is not None:
                best_sol = sol
                best_rank = target_rank
                print(f"Found valid solution for rank {target_rank}")
                break # Found a better one, stop searching for even lower?
                # Or continue to see if we can go lower?
                # Given time constraints, if we find 24, it's a big win.
                # Let's try one more step down if time permits.
        except Exception as e:
            print(f"Error in numerical search for rank {target_rank}: {e}")
    
    if best_sol is None:
        # Fallback to schoolbook
        return schoolbook_solution()
        
    return best_sol
# EVOLVE-BLOCK-END
