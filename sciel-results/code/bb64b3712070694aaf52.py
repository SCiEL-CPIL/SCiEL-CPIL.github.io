import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s limit

def construct():
    n, m, p = DIMENSIONS
    # Dimensions
    dim_u = n * m  # 6
    dim_v = m * p  # 12
    dim_w = p * n  # 8
    
    # Target tensor T_target[a, b, c]
    # a = i*m + j (index in A)
    # b = j*p + k (index in B)
    # c = k*n + i (index in C, trace convention)
    T_target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1.0

    def check_solution(U, V, W, rank):
        """
        Checks if U, V, W (floats) can be snapped to half-integers
        and exactly decompose T_target.
        """
        # Snap to half-integers
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        
        # Check if snap was close to original (optional, but helps debug)
        # If not close, it might still work if the exact solution is at a half-integer
        # and the optimizer got close enough.
        
        # Reconstruct T
        # T_rec[a, b, c] = sum_r U_s[a, r] * V_s[b, r] * W_s[c, r]
        T_rec = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
        
        # Check exactness
        # Entries should be 0 or 1.
        # Due to floating point, check if close to integers.
        if not np.allclose(T_rec, np.round(T_rec), atol=1e-9):
            return None
        
        T_rec_int = np.round(T_rec)
        
        # Check if it matches target
        if not np.array_equal(T_rec_int, T_target):
            return None
        
        return U_s, V_s, W_s

    def objective(params, rank):
        """
        Computes sum of squared errors.
        """
        size_u = dim_u * rank
        size_v = dim_v * rank
        size_w = dim_w * rank
        
        U = params[:size_u].reshape(dim_u, rank)
        V = params[size_u:size_u+size_v].reshape(dim_v, rank)
        W = params[size_u+size_v:].reshape(dim_w, rank)
        
        # Compute T_rec
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Error
        err = np.sum((T_rec - T_target)**2)
        return err

    def local_search(U, V, W, rank, time_budget=10.0):
        """
        Performs a local search / refinement starting from U, V, W.
        Uses gradient descent with adaptive step size.
        """
        start_t = time.time()
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        best_err = objective(U.flatten().tolist()[:dim_u*rank] + V.flatten().tolist()[:dim_v*rank] + W.flatten().tolist()[:dim_w*rank], rank)
        
        # We can compute gradients analytically or numerically.
        # Let's use numerical gradients for simplicity and robustness, 
        # but that's slow. 
        # Analytical gradient:
        # E = sum (T_rec - T_target)^2
        # dE/dU[a, r] = 2 * sum_{b,c} (T_rec[a,b,c] - T_target[a,b,c]) * V[b,r] * W[c,r]
        # This is efficient.
        
        step_size = 0.01
        min_step = 1e-8
        max_iter = 1000
        no_improve_count = 0
        patience = 50
        
        for it in range(max_iter):
            if time.time() - start_t > time_budget:
                break
            
            # Compute T_rec
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_rec - T_target
            err = np.sum(diff**2)
            
            if err < best_err:
                best_err = err
                no_improve_count = 0
                # Decay step size slightly if improving
                step_size = max(min_step, step_size * 0.95)
            else:
                no_improve_count += 1
                step_size = max(min_step, step_size * 0.8)
                
            if no_improve_count > patience:
                break
                
            if err < 1e-12:
                break

            # Compute Gradients
            # dE/dU[a, r] = 2 * sum_{b,c} diff[a, b, c] * V[b, r] * W[c, r]
            # This is equivalent to 2 * (diff * (V[:, r] * W[:, r]) sum over b,c)
            # More efficiently:
            # G_U[a, r] = 2 * sum_b sum_c diff[a, b, c] V[b, r] W[c, r]
            
            # We can compute this via einsum
            # Let M_b_c_r = V[b, r] * W[c, r]
            # G_U[a, r] = 2 * sum_{b, c} diff[a, b, c] * V[b, r] * W[c, r]
            
            # To avoid large intermediate arrays, we can do:
            # Temp1[b, r, c] = V[b, r] * W[c, r] ? No, that's size dim_v * rank * dim_w.
            # dim_v=12, rank=12, dim_w=8 -> 1152. Small.
            # diff is 6x12x8.
            
            # Let's just use einsum for simplicity, it's fast enough for small dims.
            # G_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
            # G_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
            # G_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            G_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
            G_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
            G_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Normalize gradients to prevent huge steps
            norm_G = np.sqrt(np.sum(G_U**2) + np.sum(G_V**2) + np.sum(G_W**2))
            if norm_G > 1e-6:
                G_U /= norm_G
                G_V /= norm_G
                G_W /= norm_G
            else:
                break
                
            # Update
            U -= step_size * G_U
            V -= step_size * G_V
            W -= step_size * G_W
            
            # Optional: Snap to half-integers if very close?
            # This can help jump over local minima if we are near a valid solution.
            # But it's non-smooth. Let's stick to smooth descent first.
            
        return U, V, W

    # Schoolbook solution for fallback
    rank_school = n * m * p
    U_school = np.zeros((dim_u, rank_school))
    V_school = np.zeros((dim_v, rank_school))
    W_school = np.zeros((dim_w, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1

    best_solution = None
    best_rank = rank_school

    # Try ranks from low to high
    # Known lower bound for 2x3x3 is 10. For 2x3x4, it's likely higher.
    # Let's try 12, 14, 16, 18, 20, 22, 24.
    # We have time for a few optimizations per rank.
    
    ranks_to_try = [12, 14, 16, 18, 20, 22, 24]
    
    for rank in ranks_to_try:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # If we found a solution with lower rank, we can stop? 
        # No, we want the MINIMAL rank. 
        # But if we find rank 12, we should check if 11 is possible? 
        # The task asks for rank-minimal. 
        # However, proving minimality is hard. 
        # Usually, finding a low rank solution is the goal. 
        # If we find rank 12, and we know LB is 10, we might be stuck.
        # Let's just find the lowest rank we can construct.
        
        found_solution = False
        
        # Multiple restarts for this rank
        num_restarts = 5
        # Reduce restarts if time is running out
        if time.time() - START_TIME > TIME_LIMIT - 20:
            num_restarts = 1
            
        for r in range(num_restarts):
            if time.time() - START_TIME > TIME_LIMIT:
                break
                
            np.random.seed(42 + rank * 100 + r)
            
            # Initialize
            # Scale: If rank is R, and we want sum of products to be 1.
            # If entries are O(1/sqrt(R)), product is O(1/R), sum is O(1).
            scale = 1.0 / np.sqrt(rank)
            params = np.random.randn(rank * (dim_u + dim_v + dim_w)) * scale
            
            U = params[:dim_u*rank].reshape(dim_u, rank)
            V = params[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
            W = params[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
            
            # Local search
            # Allocate time based on remaining budget
            time_left = TIME_LIMIT - (time.time() - START_TIME)
            # Reserve 5s for final checks and fallback
            alloc_time = max(2.0, (time_left - 5.0) / (num_restarts - r))
            # Cap at 15s per restart to avoid getting stuck
            alloc_time = min(alloc_time, 15.0)
            
            U, V, W = local_search(U, V, W, rank, time_budget=alloc_time)
            
            # Check solution
            sol = check_solution(U, V, W, rank)
            if sol is not None:
                U_s, V_s, W_s = sol
                # Convert to list of lists for return
                # Ensure they are half-integers (they are by construction of check_solution)
                return {
                    "U": U_s.tolist(),
                    "V": V_s.tolist(),
                    "W": W_s.tolist()
                }
                
            # If not found, try a "snap and polish" approach
            # Sometimes the optimizer gets close to half-integers but not quite.
            # Snap to half-integers, then do a few steps of gradient descent
            # on the "distance to target" but with the constraint that variables
            # stay near the snapped values? No, that's complex.
            # Instead, just check if snapping the current U,V,W works.
            # (Already done in check_solution)
            
            # Alternative: Randomize a few columns and retry?
            # Skip for now to save time.

        # If no solution found for this rank, continue to next (higher) rank.
        # We don't save the "best" non-integer solution because it's invalid.
        # We only return valid half-integer solutions.

    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

# Keep the entry point
# The task says: Interface: construct() -> {'U', 'V', 'W'}
# So we just need to ensure construct is defined.

if __name__ == "__main__":
    # Test
    res = construct()
    print("Constructed solution.")
    # Verify
    n, m, p = DIMENSIONS
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    rank = U.shape[1]
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1.0
    print("Max error:", np.max(np.abs(T_rec - T_target)))
    print("Rank:", rank)
