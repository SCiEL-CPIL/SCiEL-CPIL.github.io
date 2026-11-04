"""Matrix multiplication 2x3 x 3x4 rank-minimal exact tensor decomposition.

Strategy:
  1. Define the matrix multiplication tensor T.
  2. Use Alternating Least Squares (ALS) to get a continuous approximation.
  3. Project to half-integers.
  4. Use a discrete Local Search (Simulated Annealing-like or Greedy) 
     to refine the solution on the half-integer grid.
     This handles the non-convexity of the discrete problem better than pure snapping.
  5. Multi-start with different random initializations.
  6. Verify exactness.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build the target tensor as a set of linear equations
    # T[i, j, k] = 1 if j is shared, else 0.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i, 1.0))
    
    u_pos = np.array([t[0] for t in triples], dtype=np.int64)
    v_pos = np.array([t[1] for t in triples], dtype=np.int64)
    w_pos = np.array([t[2] for t in triples], dtype=np.int64)
    target = np.array([t[3] for t in triples], dtype=np.float64)
    N = len(triples)
    
    def snap_half(x):
        return np.round(x * 2.0) / 2.0

    def compute_residuals(U, V, W):
        vals = U[u_pos, :] * V[v_pos, :] * W[w_pos, :]
        return np.sum(vals, axis=1) - target

    def max_err_from_residuals(res):
        return float(np.max(np.abs(res)))

    def local_search(U, V, W, residuals, time_budget, rng):
        """
        Greedy local search on the half-integer grid.
        Updates residuals incrementally.
        """
        current_err = max_err_from_residuals(residuals)
        if current_err < 1e-9:
            return U, V, W, residuals, current_err
        
        # Pre-compute indices for fast updates
        # For each unique u_idx, which t indices are affected?
        u_to_t = {}
        v_to_t = {}
        w_to_t = [] # w_pos is 1-to-1 with t? No, w_pos[t] is k*n+i.
        # u_pos[t] = i*m+j. Multiple t can have same u_pos.
        # v_pos[t] = j*p+k. Multiple t can have same v_pos.
        # w_pos[t] = k*n+i. Multiple t can have same w_pos.
        
        for t in range(N):
            u_to_t.setdefault(u_pos[t], []).append(t)
            v_to_t.setdefault(v_pos[t], []).append(t)
            w_to_t.setdefault(w_pos[t], []).append(t)

        # We will perform random moves until time runs out or we find a solution
        # To avoid getting stuck, we can allow small worsening moves (Simulated Annealing)
        # but for now, let's stick to Greedy with Random Restarts if stuck.
        
        stuck_counter = 0
        max_stuck = 5000
        
        while time.time() < time_budget:
            if current_err < 1e-9:
                return U, V, W, residuals, current_err
            
            if stuck_counter > max_stuck:
                # Randomize a bit to escape local minimum? 
                # Or just break and rely on outer restarts.
                break
            
            # Pick a random matrix
            mat_idx = rng.randint(3)
            
            if mat_idx == 0:
                # Update U
                u_idx = rng.randint(nm)
                r = rng.randint(U.shape[1])
                old_val = U[u_idx, r]
                
                # Try +0.5 and -0.5
                for delta in [0.5, -0.5]:
                    new_val = old_val + delta
                    if new_val != old_val: # Ensure we are on grid? 
                        # Actually, we just move by 0.5.
                        # Calculate change in residuals
                        # Affected t's: u_to_t.get(u_idx, [])
                        affected_ts = u_to_t.get(u_idx, [])
                        if not affected_ts:
                            continue
                        
                        # Check if this move improves max error
                        # We need to check max over ALL residuals, not just affected ones.
                        # However, if current_err is determined by an unaffected residual, 
                        # and the move doesn't change it, the max_err stays same or decreases?
                        # No, if we reduce an error that was the max, max_err might drop.
                        # If we increase an error, max_err might increase.
                        
                        # Efficient check:
                        # 1. Find the max error among unaffected residuals (call it max_unaffected).
                        # 2. Find the max error among affected residuals after update.
                        # 3. new_max_err = max(max_unaffected, max_affected_new).
                        
                        # To do this fast, we can keep a histogram or just recompute max for affected 
                        # and compare with global max if global max was unaffected.
                        # Given N=24, recomputing max for all N is O(24) which is trivial.
                        # Let's just recompute the residuals for affected t's and find new global max.
                        
                        # Save old residuals for affected t's
                        old_res_affected = residuals[affected_ts].copy()
                        
                        # Apply delta
                        U[u_idx, r] = new_val
                        
                        # Update residuals
                        for t in affected_ts:
                            # residual[t] += delta * V[v_pos[t], r] * W[w_pos[t], r]
                            residuals[t] += delta * V[v_pos[t], r] * W[w_pos[t], r]
                        
                        new_err = max_err_from_residuals(residuals)
                        
                        if new_err < current_err:
                            current_err = new_err
                            stuck_counter = 0
                            break # Accept move, go to next random move
                        else:
                            # Reject move, revert
                            U[u_idx, r] = old_val
                            residuals[affected_ts] = old_res_affected
                            stuck_counter += 1
                            
            elif mat_idx == 1:
                # Update V
                v_idx = rng.randint(mp)
                r = rng.randint(V.shape[1])
                old_val = V[v_idx, r]
                
                for delta in [0.5, -0.5]:
                    new_val = old_val + delta
                    affected_ts = v_to_t.get(v_idx, [])
                    if not affected_ts:
                        continue
                        
                    old_res_affected = residuals[affected_ts].copy()
                    
                    V[v_idx, r] = new_val
                    for t in affected_ts:
                        residuals[t] += delta * U[u_pos[t], r] * W[w_pos[t], r]
                    
                    new_err = max_err_from_residuals(residuals)
                    
                    if new_err < current_err:
                        current_err = new_err
                        stuck_counter = 0
                        break
                    else:
                        V[v_idx, r] = old_val
                        residuals[affected_ts] = old_res_affected
                        stuck_counter += 1
            else:
                # Update W
                w_idx = rng.randint(pn)
                r = rng.randint(W.shape[1])
                old_val = W[w_idx, r]
                
                for delta in [0.5, -0.5]:
                    new_val = old_val + delta
                    affected_ts = w_to_t.get(w_idx, [])
                    if not affected_ts:
                        continue
                        
                    old_res_affected = residuals[affected_ts].copy()
                    
                    W[w_idx, r] = new_val
                    for t in affected_ts:
                        residuals[t] += delta * U[u_pos[t], r] * V[v_pos[t], r]
                    
                    new_err = max_err_from_residuals(residuals)
                    
                    if new_err < current_err:
                        current_err = new_err
                        stuck_counter = 0
                        break
                    else:
                        W[w_idx, r] = old_val
                        residuals[affected_ts] = old_res_affected
                        stuck_counter += 1
                        
        return U, V, W, residuals, current_err

    def solve_rank(R, time_budget, seed):
        rng = np.random.RandomState(seed)
        best_err = float('inf')
        best_sol = None
        
        # Time per restart
        # We want to spend most of the time in local search.
        # ALS is fast.
        
        for restart in range(200):
            if time.time() > start_time + time_budget:
                break
            
            # Initialize with random half-integers
            # Scale: random integers in [-2, 2] -> /2
            U = snap_half(rng.randint(-4, 5, size=(nm, R)))
            V = snap_half(rng.randint(-4, 5, size=(mp, R)))
            W = snap_half(rng.randint(-4, 5, size=(pn, R)))
            
            # ALS loop to get close
            for it in range(50):
                # Solve for U
                U_new = U.copy()
                A_u = V[v_pos, :] * W[w_pos, :] # N x R
                for t in range(N):
                    a_t = A_u[t, :]
                    norm_sq = np.dot(a_t, a_t)
                    if norm_sq < 1e-12:
                        continue
                    current_val = np.dot(a_t, U[u_pos[t], :])
                    target_val = target[t]
                    lam = (target_val - current_val) / norm_sq
                    U_new[u_pos[t], :] = U[u_pos[t], :] + lam * a_t
                U = snap_half(U_new)
                
                # Solve for V
                V_new = V.copy()
                A_v = U[u_pos, :] * W[w_pos, :] # N x R
                for t in range(N):
                    a_t = A_v[t, :]
                    norm_sq = np.dot(a_t, a_t)
                    if norm_sq < 1e-12:
                        continue
                    current_val = np.dot(a_t, V[v_pos[t], :])
                    target_val = target[t]
                    lam = (target_val - current_val) / norm_sq
                    V_new[v_pos[t], :] = V[v_pos[t], :] + lam * a_t
                V = snap_half(V_new)
                
                # Solve for W
                W_new = W.copy()
                A_w = U[u_pos, :] * V[v_pos, :] # N x R
                for t in range(N):
                    a_t = A_w[t, :]
                    norm_sq = np.dot(a_t, a_t)
                    if norm_sq < 1e-12:
                        continue
                    current_val = np.dot(a_t, W[w_pos[t], :])
                    target_val = target[t]
                    lam = (target_val - current_val) / norm_sq
                    W_new[w_pos[t], :] = W[w_pos[t], :] + lam * a_t
                W = snap_half(W_new)
                
                # Check error
                residuals = compute_residuals(U, V, W)
                err = max_err_from_residuals(residuals)
                if err < best_err:
                    best_err = err
                    best_sol = (U.copy(), V.copy(), W.copy())
                if best_err < 1e-9:
                    return best_sol, best_err

            # Local Search Phase
            residuals = compute_residuals(U, V, W)
            err = max_err_from_residuals(residuals)
            if err < best_err:
                best_err = err
                best_sol = (U.copy(), V.copy(), W.copy())
            
            # Allocate remaining time for this restart to local search
            # We can just run local search until global deadline or a small fixed time per restart
            ls_budget = min(start_time + time_budget, time.time() + 2.0)
            U, V, W, residuals, err = local_search(U, V, W, residuals, ls_budget, rng)
            
            if err < best_err:
                best_err = err
                best_sol = (U.copy(), V.copy(), W.copy())
            
            if best_err < 1e-9:
                return best_sol, best_err
                
        return best_sol, best_err

    def to_list(arr):
        out = []
        for row in arr:
            rl = []
            for val in row:
                rv = round(float(val) * 2) / 2.0
                rl.append(rv)
            out.append(rl)
        return out

    # Try ranks from smallest known potential to naive
    # Known lower bound for 2x3x4 is 19.
    candidate_ranks = [19, 20, 21, 22, 23, 24]
    
    for R in candidate_ranks:
        if time.time() > deadline - 10:
            break
            
        # Allocate time for this rank
        # Rank 19 is the most important. Give it more time.
        if R == 19:
            time_budget = 60.0
        else:
            time_budget = 20.0
            
        sol, err = solve_rank(R, time_budget, seed=R)
        
        if sol is not None and err < 1e-9:
            U, V, W = sol
            # Final verification
            residuals = compute_residuals(U, V, W)
            if max_err_from_residuals(residuals) < 1e-9:
                return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
        
    # Fallback: Naive decomposition
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
