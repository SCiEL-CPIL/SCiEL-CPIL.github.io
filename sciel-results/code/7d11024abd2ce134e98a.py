import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    # Target tensor T[i, j, k] = 1
    # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    num_eqs = n * m * p
    
    # Pre-compute indices for linear systems
    # For U: unknowns are U[i*m+j, c]. Index in vector: c*size_u + i*m + j
    # Equation index: i*m*p + j*p + k
    # Coefficient for U[i*m+j, c] in eq(i,j,k) is V[j*p+k, c] * W[k*n+i, c]
    
    # We will construct the A matrices for each factor type dynamically based on the other two factors.
    
    def get_A_U(V, W, R):
        # V is (size_v, R), W is (size_w, R)
        # A is (num_eqs, R * size_u)
        A = np.zeros((num_eqs, R * size_u))
        for c in range(R):
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    u_idx = c * size_u + i * m + j
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        # coeff is V_c[j, k] * W_c[k, i]
                        A[eq_idx, u_idx] = V_c[j, k] * W_c[k, i]
        return A

    def get_A_V(U, W, R):
        # U is (size_u, R), W is (size_w, R)
        # A is (num_eqs, R * size_v)
        A = np.zeros((num_eqs, R * size_v))
        for c in range(R):
            U_c = U[:, c].reshape(n, m)
            W_c = W[:, c].reshape(p, n)
            for j in range(m):
                for k in range(p):
                    v_idx = c * size_v + j * p + k
                    for i in range(n):
                        eq_idx = i * m * p + j * p + k
                        # coeff is U_c[i, j] * W_c[k, i]
                        A[eq_idx, v_idx] = U_c[i, j] * W_c[k, i]
        return A

    def get_A_W(U, V, R):
        # U is (size_u, R), V is (size_v, R)
        # A is (num_eqs, R * size_w)
        A = np.zeros((num_eqs, R * size_w))
        for c in range(R):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            for k in range(p):
                for i in range(n):
                    w_idx = c * size_w + k * n + i
                    for j in range(m):
                        eq_idx = i * m * p + j * p + k
                        # coeff is U_c[i, j] * V_c[j, k]
                        A[eq_idx, w_idx] = U_c[i, j] * V_c[j, k]
        return A

    def compute_residual(U, V, W):
        res = np.zeros((n, m, p))
        R = U.shape[1]
        for c in range(R):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            # res[i,j,k] += U_c[i,j] * V_c[j,k] * W_c[k,i]
            # Use broadcasting
            # U_c: (n, m, 1)
            # V_c: (1, m, p)
            # W_c: (p, n) -> need (n, 1, p) where [i, 0, k] = W_c[k, i]
            # W_c.T is (n, p). So W_c.T[i, k] = W_c[k, i].
            # So we use W_c.T reshaped to (n, 1, p).
            U_part = U_c[:, :, np.newaxis]
            V_part = V_c[np.newaxis, :, :]
            W_part = W_c.T[:, np.newaxis, :]
            res += U_part * V_part * W_part
        return np.sum((res - 1.0) ** 2)

    def snap_to_half_integers(M):
        return np.round(M * 2) / 2.0

    def is_valid(U, V, W):
        # Check if U, V, W are half-integers
        if not np.allclose(U * 2, np.round(U * 2)): return False
        if not np.allclose(V * 2, np.round(V * 2)): return False
        if not np.allclose(W * 2, np.round(W * 2)): return False
        
        res = np.zeros((n, m, p))
        R = U.shape[1]
        for c in range(R):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            U_part = U_c[:, :, np.newaxis]
            V_part = V_c[np.newaxis, :, :]
            W_part = W_c.T[:, np.newaxis, :]
            res += U_part * V_part * W_part
        return np.allclose(res, 1.0, atol=1e-9)

    def schoolbook_solution():
        R = n * m * p
        U = np.zeros((size_u, R))
        V = np.zeros((size_v, R))
        W = np.zeros((size_w, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook_solution()
    best_rank = n * m * p

    # Try to find lower ranks
    # We iterate R from best_rank-1 down to 1
    # To save time, we limit restarts and iterations based on remaining time.
    
    for R in range(best_rank - 1, 0, -1):
        if time.time() > deadline - 10.0:
            break
            
        # Determine number of restarts based on time
        # If we have lots of time, do more restarts.
        max_restarts = 50
        max_iters = 200
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Random initialization
            # Scale matters. 1.0 is usually good.
            U = np.random.randn(size_u, R)
            V = np.random.randn(size_v, R)
            W = np.random.randn(size_w, R)
            
            # Normalize initial columns to avoid huge/small values
            for c in range(R):
                if np.linalg.norm(U[:, c]) > 1e-6: U[:, c] /= np.linalg.norm(U[:, c])
                if np.linalg.norm(V[:, c]) > 1e-6: V[:, c] /= np.linalg.norm(V[:, c])
                if np.linalg.norm(W[:, c]) > 1e-6: W[:, c] /= np.linalg.norm(W[:, c])
            
            prev_res = np.inf
            for it in range(max_iters):
                if time.time() > deadline - 5.0:
                    break
                    
                # Solve for U
                A_U = get_A_U(V, W, R)
                b = np.ones(num_eqs)
                U_flat, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                U = U_flat.reshape(size_u, R)
                
                # Solve for V
                A_V = get_A_V(U, W, R)
                V_flat, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                V = V_flat.reshape(size_v, R)
                
                # Solve for W
                A_W = get_A_W(U, V, R)
                W_flat, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                W = W_flat.reshape(size_w, R)
                
                res = compute_residual(U, V, W)
                
                if res < 1e-8:
                    # Snap and verify
                    U_snap = snap_to_half_integers(U)
                    V_snap = snap_to_half_integers(V)
                    W_snap = snap_to_half_integers(W)
                    
                    if is_valid(U_snap, V_snap, W_snap):
                        best_U = U_snap
                        best_V = V_snap
                        best_W = W_snap
                        best_rank = R
                        break # Break ALS loop, try next restart? No, break restart loop to try lower R?
                        # Actually, if we found a valid solution for R, we should break out of the restart loop
                        # and continue to the next lower R in the outer loop.
                        # But wait, we are in the restart loop. We break the restart loop.
                        # Then the outer loop continues to R-1.
                        break
                elif res > prev_res:
                    # Diverging? Maybe reset? Or just break.
                    # ALS usually converges. If it diverges, this restart failed.
                    break
                prev_res = res
            else:
                # If we finished ALS without breaking (i.e., didn't find res < 1e-8)
                # Check if we got close?
                pass
                
            # If we broke the inner for loop because of success, we need to break the restart loop too
            # The 'break' inside the 'if res < 1e-8' block breaks the 'for it' loop.
            # We need to check if we found a solution to break the 'for restart' loop.
            # Let's use a flag.
            
    # Re-structure the loop logic to be cleaner
    
    # Reset best to schoolbook
    best_U, best_V, best_W = schoolbook_solution()
    best_rank = n * m * p
    
    for R in range(best_rank - 1, 0, -1):
        if time.time() > deadline - 10.0:
            break
            
        found = False
        max_restarts = 100
        max_iters = 300
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
                
            U = np.random.randn(size_u, R)
            V = np.random.randn(size_v, R)
            W = np.random.randn(size_w, R)
            
            for it in range(max_iters):
                if time.time() > deadline - 5.0:
                    break
                    
                A_U = get_A_U(V, W, R)
                b = np.ones(num_eqs)
                U_flat, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                U = U_flat.reshape(size_u, R)
                
                A_V = get_A_V(U, W, R)
                V_flat, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                V = V_flat.reshape(size_v, R)
                
                A_W = get_A_W(U, V, R)
                W_flat, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                W = W_flat.reshape(size_w, R)
                
                res = compute_residual(U, V, W)
                
                if res < 1e-8:
                    U_snap = snap_to_half_integers(U)
                    V_snap = snap_to_half_integers(V)
                    W_snap = snap_to_half_integers(W)
                    
                    if is_valid(U_snap, V_snap, W_snap):
                        best_U = U_snap
                        best_V = V_snap
                        best_W = W_snap
                        best_rank = R
                        found = True
                        break
            if found:
                break
        
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
