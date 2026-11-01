"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    n, m, p = DIMENSIONS

    # Target tensor T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def time_left():
        return deadline - time.time()

    def verify(U, V, W):
        # Check if U, V, W decompose T
        # T_rec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For speed, we can compute this via matrix multiplication tricks or just einsum
        # Since dimensions are small, einsum is fine.
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_rec - T)
        return err < 1e-9

    def random_init(rank, seed):
        np.random.seed(seed)
        # Initialize with small integers or zeros, then fill with random +/-1
        # Or start with schoolbook-like sparse structure?
        # Let's start with random sparse vectors with integer entries.
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # A better initialization: start from the rank-42 Strassen solution?
        # No, that's fixed. Let's start from a random perturbation of a lower rank guess.
        # Or just random integers in {-1, 0, 1}.
        
        # To help the search, let's bias towards non-zero entries.
        for r in range(rank):
            # Randomly select a few non-zero entries for each factor
            # This keeps the matrix sparse-ish, which might help convergence?
            # Actually, dense random integers might get stuck.
            # Let's try dense random integers in {-1, 1}.
            U[:, r] = np.random.choice([-1, 1], size=n * m)
            V[:, r] = np.random.choice([-1, 1], size=m * p)
            W[:, r] = np.random.choice([-1, 1], size=p * n)
            
        return U, V, W

    def search(rank, max_iter, seed):
        U, V, W = random_init(rank, seed)
        
        # We want to minimize || U V^T W^T - T ||? No, it's a tensor.
        # Objective: f(U,V,W) = sum_{a,b,c} (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
        
        # Gradient descent is hard with integer constraints.
        # Let's use a simple local search / simulated annealing style approach.
        # Or coordinate descent: pick one entry, change it by +/-1 or set to 0, accept if error decreases.
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Precompute T for fast error calculation?
        # Error calculation:
        # E = || T_rec - T ||^2
        # T_rec = einsum('ar,br,cr->abc', U, V, W)
        
        # To speed up, we can compute the gradient of the error w.r.t. each entry.
        # dE/dU[a,r] = 2 * sum_{b,c} (T_rec[a,b,c] - T[a,b,c]) V[b,r] W[c,r]
        # This allows us to find the "best" move for each coordinate.
        
        def compute_grads(U, V, W):
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_rec - T
            # Grad U: shape (n*m, rank)
            # Gu[a, r] = 2 * sum_{b,c} diff[a,b,c] V[b,r] W[c,r]
            # This is 2 * (diff @ (V[:,r] outer W[:,r]))? No.
            # sum_{b,c} diff[a,b,c] V[b,r] W[c,r]
            # Let M_r[b,c] = V[b,r] W[c,r]
            # Then Gu[a,r] = 2 * sum_{b,c} diff[a,b,c] M_r[b,c]
            # We can compute this by looping over r? Or using einsum.
            # Gu = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
            # Gv = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
            # Gw = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            Gu = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
            Gv = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
            Gw = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
            return Gu, Gv, Gw

        # Local search:
        # At each step, pick a random coordinate, try changing it by -1, 0, +1.
        # If error decreases, accept.
        # To be faster, use gradient to determine direction.
        
        current_err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)**2
        if current_err < best_err:
            best_err = current_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        for it in range(max_iter):
            if time_left() < 5:
                break
                
            # Compute gradients
            Gu, Gv, Gw = compute_grads(U, V, W)
            
            # Pick a random coordinate from U, V, or W
            # Let's cycle or random pick.
            # Random pick:
            coord_type = np.random.choice(3)
            if coord_type == 0:
                idx_a = np.random.randint(n * m)
                idx_r = np.random.randint(rank)
                # Try changing U[idx_a, idx_r] by +/-1
                # Gradient Gu[idx_a, idx_r] indicates direction.
                # If Gu > 0, decreasing U reduces error. If Gu < 0, increasing U reduces error.
                # But we are constrained to integers.
                # Let's try the move that most reduces error locally.
                # E(new) - E(old) approx grad * delta
                # Exact check is better.
                
                old_val = U[idx_a, idx_r]
                # Try delta = -1, 0, 1
                # Actually, just try the sign opposite to gradient.
                if Gu[idx_a, idx_r] > 0:
                    delta = -1
                elif Gu[idx_a, idx_r] < 0:
                    delta = 1
                else:
                    delta = np.random.choice([-1, 1])
                    
                U[idx_a, idx_r] += delta
                new_err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)**2
                
                if new_err < current_err:
                    current_err = new_err
                    if current_err < best_err:
                        best_err = current_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    U[idx_a, idx_r] -= delta # revert
                    
            elif coord_type == 1:
                idx_b = np.random.randint(m * p)
                idx_r = np.random.randint(rank)
                if Gv[idx_b, idx_r] > 0:
                    delta = -1
                elif Gv[idx_b, idx_r] < 0:
                    delta = 1
                else:
                    delta = np.random.choice([-1, 1])
                    
                V[idx_b, idx_r] += delta
                new_err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)**2
                
                if new_err < current_err:
                    current_err = new_err
                    if current_err < best_err:
                        best_err = current_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    V[idx_b, idx_r] -= delta
            else:
                idx_c = np.random.randint(p * n)
                idx_r = np.random.randint(rank)
                if Gw[idx_c, idx_r] > 0:
                    delta = -1
                elif Gw[idx_c, idx_r] < 0:
                    delta = 1
                else:
                    delta = np.random.choice([-1, 1])
                    
                W[idx_c, idx_r] += delta
                new_err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)**2
                
                if new_err < current_err:
                    current_err = new_err
                    if current_err < best_err:
                        best_err = current_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    W[idx_c, idx_r] -= delta
                    
        return best_U, best_V, best_W, best_err

    # Strategy:
    # 1. Start with rank 42 (known good) as a baseline? Or start lower?
    # 2. Try rank 40, 38, 36... down to a lower bound.
    # 3. For each rank, run multiple short searches with different seeds.
    # 4. If we find a valid decomposition, return it.
    
    # Lower bound for [2,2,12] is likely higher than 24.
    # Let's try ranks: 40, 38, 36, 34, 32, 30.
    # If we fail, we return the best found (which might be invalid, so we need fallback).
    # Actually, if we don't find a valid one, we must return the rank 42 solution to ensure validity.
    # But the task says "score is the rank". If we return rank 42, score is 42 (or 1.143 relative to 48).
    # If we find rank 40, score is 40 (or 1.2 relative to 48).
    
    # Let's first generate the rank 42 solution to have a fallback.
    def get_strassen_42():
        rank = 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Re-use the logic from the seed program, but cleaner.
        # We define the 7 Strassen products for a 2x2x2 block.
        # Then replicate for 6 blocks.
        
        # Strassen 2x2x2 factors (u, v, w) for local indices (i,j), (j,k), (k,i)
        # Note: local k is 0,1. Global k = 2t + local_k.
        
        # M1: (a00+a11)(b00+b11) -> c00+=1, c11+=1
        # M2: (a10+a11)b00 -> c10+=1
        # M3: a00(b01-b11) -> c01+=1
        # M4: a11(b10-b00) -> c00+=1, c10+=1
        # M5: (a00+a01)b11 -> c00-=1, c01+=1
        # M6: (a10-a00)(b00+b01) -> c11+=1
        # M7: (a01-a11)(b10+b11) -> c00-=1, c11+=1
        
        # Let's hardcode the non-zero entries for each of the 7 products.
        # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        prods = [
            # M1
            ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
            # M2
            ([(2,1), (3,1)], [(0,1)], [(2,1)]), # c10 is k=0, i=1 -> idx 2
            # M3
            ([(0,1)], [(1,1), (3,-1)], [(1,1)]), # c01 is k=1, i=0 -> idx 1
            # M4
            ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]), # c00=idx0, c10=idx2
            # M5
            ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]), # c00=idx0, c01=idx1
            # M6
            ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]), # c11=idx3
            # M7
            ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,-1), (3,1)]) # c00=idx0, c11=idx3
        ]
        
        for t in range(6):
            for s in range(7):
                col = 7 * t + s
                u_entries, v_entries, w_entries = prods[s]
                
                for idx, coeff in u_entries:
                    U[idx, col] = coeff
                    
                for idx, coeff in v_entries:
                    j = idx // 2
                    k_loc = idx % 2
                    k = 2 * t + k_loc
                    V[j * p + k, col] = coeff
                    
                for idx, coeff in w_entries:
                    k_loc = idx // 2
                    i = idx % 2
                    k = 2 * t + k_loc
                    W[k * n + i, col] = coeff
                    
        return U, V, W

    U_42, V_42, W_42 = get_strassen_42()
    
    # Verify 42
    if not verify(U_42, V_42, W_42):
        # Fallback to schoolbook if Strassen logic is broken
        rank_fb = 48
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_fb[a, col] = 1
                    V_fb[b, col] = 1
                    W_fb[c, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Now try to improve
    best_rank = 42
    best_U, best_V, best_W = U_42, V_42, W_42
    
    # Try ranks lower than 42
    # We have ~100s.
    # Each search takes some time. Let's limit iterations per rank.
    
    # We can also try to "prune" the rank 42 solution.
    # If we set some columns to 0, does the error stay 0?
    # This is a specific local search: start with rank 42, try to zero out columns.
    
    def prune_search(U, V, W, max_iter):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        # Try to remove columns one by one.
        # If removing column r keeps error 0, remove it (shift columns or just mark as used).
        # Since we need to return dense matrices, we can just zero them out.
        # But rank is defined by number of non-zero columns?
        # The evaluator likely checks if the matrix decomposes T.
        # If a column is all zeros, it contributes nothing.
        # Does the evaluator count the shape or the rank?
        # "factor matrices of shapes (n*m, rank)... score is the rank"
        # Usually, if you return a matrix with shape (4, 40) but 2 columns are zero,
        # does it count as rank 40 or rank 38?
        # The prompt says "score is the rank". It likely infers rank from the matrix shape or non-zero columns.
        # To be safe, if we find a valid decomposition with K non-zero columns out of R,
        # we should return matrices of shape (n*m, K) containing only the non-zero columns.
        
        active_cols = list(range(rank))
        
        for it in range(max_iter):
            if time_left() < 5:
                break
            
            improved = False
            # Try removing each active column
            for idx, col in enumerate(active_cols):
                # Check if col is essential
                # Temporarily zero it
                U[:, col] = 0
                V[:, col] = 0
                W[:, col] = 0
                
                if verify(U, V, W):
                    # It's redundant! Remove it from active list.
                    active_cols.pop(idx)
                    improved = True
                    # Don't break, try to remove more?
                    # But we modified U,V,W.
                    # We can continue checking others.
                else:
                    # Restore
                    U[:, col] = U_42[:, col] # This is wrong, U_42 is original.
                    # We need to keep track of the current state.
                    # Let's restore from a copy or just undo the zeroing.
                    # This is inefficient.
                    pass
                    
            if not improved:
                break
                
        # Extract active columns
        if len(active_cols) < rank:
            U_new = U[:, active_cols]
            V_new = V[:, active_cols]
            W_new = W[:, active_cols]
            return U_new, V_new, W_new, len(active_cols)
        else:
            return U, V, W, rank

    # The prune search above has a bug in restoration. Let's do a cleaner version.
    def prune_search_clean(U, V, W, max_iter):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        active_cols = list(range(rank))
        
        # To make removal safe, we don't zero out in place permanently until we confirm.
        # But verify is expensive.
        
        # Strategy:
        # 1. Start with all columns active.
        # 2. For each column c, try to remove it.
        #    To check if c is removable, we need to see if the tensor T can be formed by the other columns.
        #    This is equivalent to checking if the column vector (U[:,c], V[:,c], W[:,c]) is in the span of the others?
        #    No, tensor rank is not linear span.
        #    We just have to recompute the tensor sum without column c.
        
        for c in active_cols[:]: # iterate over a copy
            if time_left() < 5:
                break
            
            # Try removing c
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            U_test[:, c] = 0
            V_test[:, c] = 0
            W_test[:, c] = 0
            
            if verify(U_test, V_test, W_test):
                # Remove c
                active_cols.remove(c)
                # Update U,V,W to only have active columns?
                # Or keep them and just track active_cols.
                # At the end, we slice.
                # But if we remove c, the indices of other columns don't change.
                # So we can just keep the full matrices and slice at the end.
                
        # Slice
        if len(active_cols) < rank:
            return U[:, active_cols], V[:, active_cols], W[:, active_cols], len(active_cols)
        else:
            return U, V, W, rank

    # Run prune search on the rank 42 solution
    U_p, V_p, W_p, rank_p = prune_search_clean(U_42, V_42, W_42, 100)
    
    if rank_p < best_rank:
        best_rank = rank_p
        best_U, best_V, best_W = U_p, V_p, W_p
        
    # Now try to find even lower ranks with random search?
    # If prune didn't work, maybe a random search for rank 40?
    # But random search is very slow.
    
    # Let's try a "mutation" search from the rank 42 solution.
    # Take rank 42, try to merge two columns into one?
    # If U[:,c1] + U[:,c2] = U_new, etc., and the sum of outer products matches?
    # (u1 v1 w1) + (u2 v2 w2) = u_new v_new w_new?
    # This is only possible if the two rank-1 tensors are proportional.
    # So if col 1 and col 2 are proportional, we can merge them.
    
    def merge_search(U, V, W):
        rank = U.shape[1]
        active = list(range(rank))
        
        for c1 in active[:]:
            if time_left() < 5:
                break
            for c2 in active[:]:
                if c1 == c2 or c2 < c1: # avoid duplicates
                    continue
                
                # Check if col c1 and c2 are proportional
                # u1 = alpha * u2, v1 = alpha * v2, w1 = alpha * w2
                # If so, we can replace them with a single column u1+u2, etc?
                # No, (u1 v1 w1) + (u2 v2 w2) = u2(alpha v2 w2 + v2 w2) = u2 (alpha+1) v2 w2.
                # So yes, we can merge them into a single column with scaled factors.
                
                u1, u2 = U[:, c1], U[:, c2]
                v1, v2 = V[:, c1], V[:, c2]
                w1, w2 = W[:, c1], W[:, c2]
                
                # Find alpha such that u1 = alpha u2
                # If u2 is zero, skip.
                if np.allclose(u2, 0) or np.allclose(u1, 0):
                    continue
                    
                # Estimate alpha
                # Use non-zero elements
                mask = u2 != 0
                if np.sum(mask) == 0:
                    continue
                
                alphas = u1[mask] / u2[mask]
                alpha = alphas[0]
                
                if not np.allclose(alphas, alpha):
                    continue
                    
                # Check v and w
                if not np.allclose(v1, alpha * v2):
                    continue
                if not np.allclose(w1, alpha * w2):
                    continue
                
                # Merge: replace c2 with c1+c2, remove c1?
                # Or just keep c2 and scale it?
                # The term is u1 v1 w1 + u2 v2 w2 = u2(alpha+1) v2 w2.
                # So we can just update column c2 to be (alpha+1)*u2, (alpha+1)*v2, (alpha+1)*w2.
                # And zero out c1.
                
                U[:, c2] = (alpha + 1) * u2
                V[:, c2] = (alpha + 1) * v2
                W[:, c2] = (alpha + 1) * w2
                U[:, c1] = 0
                V[:, c1] = 0
                W[:, c1] = 0
                
                if c1 in active:
                    active.remove(c1)
                    
        # Slice
        if len(active) < rank:
            return U[:, active], V[:, active], W[:, active], len(active)
        else:
            return U, V, W, rank

    U_m, V_m, W_m, rank_m = merge_search(U_p, V_p, W_p)
    
    if rank_m < best_rank:
        best_rank = rank_m
        best_U, best_V, best_W = U_m, V_m, W_m

    # If we still have time, try random search for lower ranks?
    # Probably not enough time.
    
    # Return best found
    if verify(best_U, best_V, best_W):
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback
        return {
            "U": U_42.tolist(),
            "V": V_42.tolist(),
            "W": W_42.tolist()
        }
# EVOLVE-BLOCK-END
