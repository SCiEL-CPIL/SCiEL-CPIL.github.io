"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    We use a structured search approach:
    1. Start with the schoolbook rank 54 decomposition
    2. Try to find lower rank decompositions using algebraic constraints
    3. The key insight is that for small matrices, we can search over possible 
       rank-r decompositions by solving systems of polynomial equations
    
    For 3x3 * 3x6, the theoretical lower bound is max(n*m, m*p, p*n) = max(9, 18, 18) = 18,
    but the actual minimum is known to be higher. We'll search systematically.
    
    Strategy: Use the fact that the multiplication tensor T has components T[i,j,k] = 1 
    if the output index (k*n+i) depends on input indices (i*m+j) and (j*p+k).
    
    For a rank-r decomposition, we need vectors u_c, v_c, w_c such that:
    T = sum_c u_c ⊗ v_c ⊗ w_c
    
    This means: for all (i,j,k) where the output element (C)_{i,k} = sum_j A_{i,j} * B_{j,k},
    we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{correct}
    
    where delta_{correct} = 1 if (i,j,k) corresponds to a valid multiplication term, else 0.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    def schoolbook_decomposition():
        """Return the schoolbook rank n*m*p decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def validate(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0]) if U else 0
        # Verify all required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # This term should contribute 1 to C[i,k]
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def try_rank_r_decomposition(r, max_iter=500, timeout=30):
        """
        Try to find a rank-r decomposition using random initialization and gradient descent
        on the constraint that the tensor is exactly decomposed.
        
        We formulate this as: minimize ||T - sum_c u_c ⊗ v_c ⊗ w_c||^2
        subject to the constraint that the solution is exact (residual = 0).
        """
        start_time = time.time()
        
        # Precompute indices for the tensor
        # We need to map (i, j, k) to indices in U, V, W
        # U index: i * m + j
        # V index: j * p + k
        # W index: k * n + i
        
        # Create arrays for vectorized operations
        # Residual is sum over valid (i,j,k) of (sum_c U[a,c] V[b,c] W[c_idx,c] - 1)^2
        
        # We will use NumPy for speed
        # Dimensions:
        # U: (n*m, r)
        # V: (m*p, r)
        # W: (p*n, r)
        
        # List of valid (i,j,k) triples
        triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        # Precompute indices
        idx_U = [i * m + j for i, j, k in triples]
        idx_V = [j * p + k for i, j, k in triples]
        idx_W = [k * n + i for i, j, k in triples]
        
        # These are arrays of length n*m*p
        idx_U = np.array(idx_U)
        idx_V = np.array(idx_V)
        idx_W = np.array(idx_W)
        
        # We want to minimize E = sum_t (sum_c U[idx_U[t], c] V[idx_V[t], c] W[idx_W[t], c] - 1)^2
        # Let S_t = sum_c U[idx_U[t], c] V[idx_V[t], c] W[idx_W[t], c]
        
        # Gradient descent with NumPy
        # U, V, W are (dim, r) matrices
        
        for restart in range(10):
            if time.time() - start_time > timeout:
                break
                
            # Initialize
            # Use a structured initialization: start with random but small values
            # Or try to initialize with a "sparse" structure
            U = np.random.uniform(-0.5, 0.5, (n * m, r))
            V = np.random.uniform(-0.5, 0.5, (m * p, r))
            W = np.random.uniform(-0.5, 0.5, (p * n, r))
            
            lr = 0.01
            for it in range(max_iter):
                if time.time() - start_time > timeout:
                    break
                
                # Compute S_t for all t
                # U_sel = U[idx_U, :] -> (T, r)
                # V_sel = V[idx_V, :] -> (T, r)
                # W_sel = W[idx_W, :] -> (T, r)
                
                U_sel = U[idx_U, :]
                V_sel = V[idx_V, :]
                W_sel = W[idx_W, :]
                
                # S_t = sum_c U_sel[t,c] * V_sel[t,c] * W_sel[t,c]
                S = np.sum(U_sel * V_sel * W_sel, axis=1)
                
                # Residual
                diff = S - 1.0
                error = np.sum(diff * diff)
                
                if error < 1e-10:
                    # Check exactness with rounding
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Convert to lists for validate
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    
                    if validate(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    
                    # If rounding didn't work, maybe we are close but not exact.
                    # Try a local polish? Or accept if error is extremely small?
                    # The task requires exact integer/half-integer entries.
                    # If the float solution is very close, rounding should work.
                    # If not, this restart failed.
                    continue

                # Compute Gradients
                # dE/dU[a,c] = sum_{t: idx_U[t]==a} 2 * diff[t] * V[idx_V[t],c] * W[idx_W[t],c]
                
                # Vectorized gradient computation
                # grad_U = (2 * diff[:, None] * V_sel * W_sel).T @ (one-hot of idx_U)
                # Actually, simpler:
                # term_U = 2 * diff[:, None] * V_sel * W_sel  # (T, r)
                # We need to sum term_U[t] for all t where idx_U[t] == a
                
                term_U = 2.0 * diff[:, None] * V_sel * W_sel
                term_V = 2.0 * diff[:, None] * U_sel * W_sel
                term_W = 2.0 * diff[:, None] * U_sel * V_sel
                
                # Accumulate gradients
                grad_U = np.zeros((n * m, r))
                grad_V = np.zeros((m * p, r))
                grad_W = np.zeros((p * n, r))
                
                # np.add.at for unbuffered in-place addition
                np.add.at(grad_U, idx_U, term_U)
                np.add.at(grad_V, idx_V, term_V)
                np.add.at(grad_W, idx_W, term_W)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Decay learning rate
                lr *= 0.995
                
                # Optional: Normalize or constrain?
                # The problem is homogeneous in scaling of U, V, W? No, it's cubic.
                # But we can fix the scale of one factor to avoid diverging to 0 or inf.
                # For example, keep ||U||_F = 1?
                # Or just let it be.
        
        return None
    
    # First, try to find a decomposition with rank < 54
    # The minimum rank for 3x3 * 3x6 is not well-known, but let's search
    
    # Try ranks from a reasonable lower bound up
    # Lower bound: max(n*m, m*p, n*p) = max(9, 18, 18) = 18
    # But this is a weak bound. Let's try from 18 upwards
    
    best = None
    # We have 120s total. Use 110s for search.
    deadline = time.time() + 110
    
    for r in range(18, 55):
        if time.time() > deadline:
            break
        result = try_rank_r_decomposition(r, max_iter=200, timeout=10)
        if result is not None:
            best = result
            break
    
    if best is not None:
        return best
    
    # Fall back to schoolbook
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
