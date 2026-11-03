"""Find rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 12
    dim_w = p * n  # 18
    
    # Build the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition (rank = n*m*p = 72)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin (evaluator kills at 150s)
    
    def verify(U, V, W, tol=1e-10):
        """Check if the decomposition is exact."""
        pred = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(pred, T, atol=tol, rtol=0)
    
    def clean_matrix(M):
        """Round to nearest half-integer."""
        M = np.round(M * 2) / 2
        return M.tolist()
    
    # --- Deterministic Strassen-style block construction ---
    # For 3x4 * 4x6, we can think of it as blocks.
    # Let's try a structured approach: split the 4 middle dimension into 2+2
    # and the 3 rows into 1+1+1, and 6 cols into 2+2+2.
    # 
    # Actually, let's try a simpler structured decomposition first:
    # Use Strassen for 2x2 blocks where possible.
    #
    # A (3x4) * B (4x6) = C (3x6)
    # Split A into rows: A1 (1x4), A2 (1x4), A3 (1x4)
    # Split B into cols: B1 (4x2), B2 (4x2), B3 (4x2)
    # Split B into rows: B_top (2x6), B_bot (2x6)
    # Split A into cols: A_left (3x2), A_right (3x2)
    #
    # C = A1*B1 + A2*B2 + A3*B3 (where each is 1x2 * 2x2 = 1x2)
    #
    # Let's use a rank-66 construction based on known results.
    # The theoretical lower bound is around 63, and known constructions give 66.
    
    # Try a structured rank-66 decomposition:
    # We'll use the fact that 3*4*6 = 72, and we can save 6 operations.
    # One approach: use Strassen's algorithm on 2x2 sub-blocks.
    
    def structured_rank66():
        """
        Construct a rank-66 decomposition using block structure.
        
        Split the 4 middle dimension as 2+2.
        A (3x4) = [A_left (3x2) | A_right (3x2)]
        B (4x6) = [B_top (2x6); B_bot (2x6)]
        
        C = A_left * B_top + A_right * B_bot
        
        Now A_left (3x2) * B_top (2x6) = 3*2*6 = 36 operations
        A_right (3x2) * B_bot (2x6) = 3*2*6 = 36 operations
        Total = 72 (no savings yet)
        
        To save, we need to use Strassen on the 3x2 * 2x6 products.
        But Strassen works on 2x2. Let's think differently.
        
        Actually, let's just use a known rank-66 construction.
        The key insight is that we can share some computations.
        """
        rank = 66
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        
        # We'll build this by combining schoolbook for most terms
        # and using some shared structure for 6 terms.
        #
        # For now, let's just return the schoolbook and try to improve.
        # A proper rank-66 construction requires careful design.
        
        # Let's try a different approach: use the fact that
        # we can write some products as sums of fewer terms.
        
        # Actually, let's just do the schoolbook and then try to
        # find a lower-rank decomposition via optimization.
        return None  # Skip for now, use optimization
    
    # --- Optimization-based search for lower rank ---
    # For each rank r, try to find a decomposition.
    # We'll use a smarter approach: initialize with schoolbook,
    # then try to zero out some entries via optimization.
    
    for r in range(71, 60, -1):  # Try ranks 71 down to 61
        if time.time() > deadline - 5:
            break
        
        found = False
        max_restarts = 10
        max_iter = 100
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize: start with schoolbook and perturb
            U = np.zeros((dim_u, r))
            V = np.zeros((dim_v, r))
            W = np.zeros((dim_w, r))
            
            # Fill first r columns of schoolbook
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= r:
                            break
                        U[i*m+j, col] = 1.0
                        V[j*p+k, col] = 1.0
                        W[k*n+i, col] = 1.0
                        col += 1
                        if col >= r:
                            break
                    if col >= r:
                        break
                if col >= r:
                    break
            
            # Add small random perturbation to break symmetry
            U += 0.01 * np.random.randn(dim_u, r)
            V += 0.01 * np.random.randn(dim_v, r)
            W += 0.01 * np.random.randn(dim_w, r)
            
            # Optimize using gradient descent with adaptive learning rate
            lr = 0.01
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                pred = np.einsum('ar,br,cr->abc', U, V, W)
                diff = pred - T
                
                # Check convergence
                err = np.linalg.norm(diff)
                if err < 1e-12:
                    if verify(U, V, W, tol=1e-12):
                        found = True
                        best_rank = r
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        break
                
                # Compute gradients
                # dU[a,r] = sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
                # dV[b,r] = sum_{a,c} diff[a,b,c] * U[a,r] * W[c,r]
                # dW[c,r] = sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
                
                dU = np.einsum('abc,br,cr->ar', diff, V, W)
                dV = np.einsum('abc,ar,cr->br', diff, U, W)
                dW = np.einsum('abc,abr->cr', diff, U[:, np.newaxis, :] * V[np.newaxis, :, :])
                
                # Adaptive learning rate based on gradient magnitude
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm > 1e-15:
                    step = lr / (1 + 0.01 * it)
                    U -= step * dU
                    V -= step * dV
                    W -= step * dW
                
                # Periodically check and adjust learning rate
                if it % 10 == 0:
                    new_err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)
                    if new_err > err:
                        lr *= 0.5
                    else:
                        lr *= 1.1
        
        if found:
            break
    
    # Final verification and cleanup
    U_final = np.array(best_U, dtype=np.float64)
    V_final = np.array(best_V, dtype=np.float64)
    W_final = np.array(best_W, dtype=np.float64)
    
    if not verify(U_final, V_final, W_final, tol=1e-10):
        # Fall back to schoolbook
        U_final, V_final, W_final = U_sb, V_sb, W_sb
        best_rank = n * m * p
    
    U_list = clean_matrix(U_final)
    V_list = clean_matrix(V_final)
    W_list = clean_matrix(W_final)
    
    # Final verification after cleaning
    U_np = np.array(U_list, dtype=np.float64)
    V_np = np.array(V_list, dtype=np.float64)
    W_np = np.array(W_list, dtype=np.float64)
    
    if not verify(U_np, V_np, W_np, tol=1e-10):
        # Fall back to schoolbook
        U_list = U_sb.tolist()
        V_list = V_sb.tolist()
        W_list = W_sb.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
