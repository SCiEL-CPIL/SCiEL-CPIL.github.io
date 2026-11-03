"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Ensure int64 for exact arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    # Precompute row sums
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)

    # 1. Initial Solution: Rank Matching
    heavy_idx = np.argsort(-a_sums, kind="stable")
    central_idx = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_idx] = central_idx

    # Helper to compute cost
    def calc_cost(perm):
        # sum_{i,j} a[i,j] * b[perm[i], perm[j]]
        return int(np.sum(a * b[perm][:, perm]))

    best_cost = calc_cost(p)
    best_p = p.copy()

    # Precompute Bp = b[:, p] for fast delta lookup
    # Bp[k, j] = b[k, p[j]]
    Bp = b[:, p]

    # Helper to update Bp after a swap of positions i and j
    def update_Bp(i, j):
        # p[i] and p[j] are swapped
        # Old: Bp[:, i] = b[:, p[i]], Bp[:, j] = b[:, p[j]]
        # New: Bp[:, i] = b[:, p[j]], Bp[:, j] = b[:, p[i]]
        # Since we swap p[i] and p[j] in-place, we need to be careful.
        # Let's just recompute the two columns
        Bp[:, i] = b[:, p[i]]
        Bp[:, j] = b[:, p[j]]

    # Tabu Search Parameters
    tabu_tenure = 30
    tabu_list = {}  # (i, j) -> step_expires
    current_step = 0
    no_improve_count = 0
    max_no_improve = 1500  # Restart threshold

    # Main Loop
    while time.time() < deadline:
        # Check for restart
        if no_improve_count > max_no_improve:
            # Perturb: random 3-swap or more
            # Simple 3-swap: i, j, k -> j, k, i
            idx = random.sample(range(n), 3)
            i, j, k = idx
            p[i], p[j], p[k] = p[j], p[k], p[i]
            update_Bp(i, j)
            update_Bp(j, k)
            update_Bp(i, k) # Ensure all affected columns are updated
            
            # Actually, update_Bp is called 3 times which is redundant but safe.
            # Let's just recompute the 3 columns to be safe and fast.
            Bp[:, i] = b[:, p[i]]
            Bp[:, j] = b[:, p[j]]
            Bp[:, k] = b[:, p[k]]

            curr_cost = calc_cost(p)
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_p = p.copy()
            
            no_improve_count = 0
            tabu_list.clear()
            current_step = 0
            continue

        # Find best non-tabu move
        best_delta = 0
        best_i = -1
        best_j = -1
        best_delta_val = 0

        # Vectorized-ish delta evaluation
        # delta(i, j) = dot(a[i] - a[j], Bp[p[j]] - Bp[p[i]])
        # Note: Bp is b[:, p]. So Bp[p[j]] is b[p[j], p] and Bp[p[i]] is b[p[i], p].
        # Wait, Bp[k, m] = b[k, p[m]].
        # We want sum_k (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]])
        # = sum_k (a[i,k] - a[j,k]) * (Bp[p[j], k] - Bp[p[i], k])
        # = dot(a[i] - a[j], Bp[p[j]] - Bp[p[i]])
        
        for i in range(n):
            if time.time() > deadline:
                break
            ai = a[i]
            pi = p[i]
            
            for j in range(i + 1, n):
                # Check tabu
                key = (i, j)
                if key in tabu_list and tabu_list[key] > current_step:
                    continue
                
                pj = p[j]
                
                # Compute delta
                # diff_a = ai - a[j]
                # diff_b = Bp[pj] - Bp[pi]
                # delta = dot(diff_a, diff_b)
                
                # Optimization: Only compute if likely to be improving?
                # Hard to know without computing.
                
                diff_a = ai - a[j]
                diff_b = Bp[pj] - Bp[pi]
                delta = int(np.dot(diff_a, diff_b))
                
                # Aspiration criterion: allow if it improves global best
                # (We don't track global best cost in loop easily, so skip for now or use current_cost)
                # If we had current_cost, we could check if current_cost + delta < best_cost
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
                    best_delta_val = delta

        if best_i == -1:
            # No improving move found
            no_improve_count += 1
            # If stuck, force a move or restart sooner
            if no_improve_count > 100:
                # Force a random move to escape
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                while j == i:
                    j = random.randint(0, n - 1)
                
                p[i], p[j] = p[j], p[i]
                Bp[:, i] = b[:, p[i]]
                Bp[:, j] = b[:, p[j]]
                
                curr_cost = calc_cost(p)
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_p = p.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 100 # Accelerate restart
                    
                tabu_list[(i, j)] = current_step + tabu_tenure
                tabu_list[(j, i)] = current_step + tabu_tenure
                current_step += 1
                continue
            else:
                # Just wait? No, we need to do something.
                # If no improving move, we are at a local optimum.
                # Increment no_improve and let the restart logic handle it.
                # But we need to consume time or do something.
                # Actually, if best_delta >= 0, we accept no move this iteration.
                # To avoid infinite loop, we increment no_improve.
                no_improve_count += 1
                current_step += 1
                continue

        # Perform the best move
        i, j = best_i, best_j
        p[i], p[j] = p[j], p[i]
        
        # Update Bp
        Bp[:, i] = b[:, p[i]]
        Bp[:, j] = b[:, p[j]]
        
        # Update cost
        # We don't maintain current_cost variable strictly, but we can update best_cost
        # Actually, we should maintain current_cost to check aspiration.
        # Let's assume current_cost is tracked.
        # For now, just check if this move improves best_cost.
        # We need current_cost. Let's add it.
        # Refactoring: Add current_cost tracking.
        
        # Since we didn't track current_cost in the loop above, let's calculate it or track it.
        # Tracking is O(1) if we have delta.
        # We need to initialize current_cost.
        # Let's restart the loop logic with current_cost.

        # --- RESTART LOGIC WITH CURRENT COST TRACKING ---
        # The code above is missing current_cost. Let's fix it in the final version.
        
        # For now, let's just break out and use a cleaner implementation below.
        break

    # --- CLEANER IMPLEMENTATION ---
    # The above loop had a flaw (missing current_cost). Let's rewrite the main loop properly.
    
    # Reset state
    p = best_p.copy() # Start from best found so far (which is initial)
    Bp = b[:, p]
    current_cost = best_cost
    best_cost = current_cost
    best_p = p.copy()
    
    tabu_list = {}
    current_step = 0
    no_improve_count = 0
    max_no_improve = 1500

    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Perturb
            idx = random.sample(range(n), 3)
            i, j, k = idx
            p[i], p[j], p[k] = p[j], p[k], p[i]
            Bp[:, i] = b[:, p[i]]
            Bp[:, j] = b[:, p[j]]
            Bp[:, k] = b[:, p[k]]
            
            current_cost = calc_cost(p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            
            no_improve_count = 0
            tabu_list.clear()
            current_step = 0
            continue

        best_delta = 0
        best_i = -1
        best_j = -1

        for i in range(n):
            if time.time() > deadline:
                break
            ai = a[i]
            pi = p[i]
            
            for j in range(i + 1, n):
                key = (i, j)
                if key in tabu_list and tabu_list[key] > current_step:
                    continue
                
                pj = p[j]
                diff_a = ai - a[j]
                diff_b = Bp[pj] - Bp[pi]
                delta = int(np.dot(diff_a, diff_b))
                
                # Aspiration: if delta < 0 and current_cost + delta < best_cost, allow even if tabu
                # But we already skipped tabu. So only non-tabu moves are considered.
                # We can add aspiration to override tabu if it improves best.
                # For simplicity, we stick to non-tabu improving moves.
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j

        if best_i == -1:
            # No improving non-tabu move
            no_improve_count += 1
            current_step += 1
            # If stuck for a while, clear tabu to allow moves
            if no_improve_count % 50 == 0:
                tabu_list.clear()
            continue

        # Perform move
        i, j = best_i, best_j
        p[i], p[j] = p[j], p[i]
        Bp[:, i] = b[:, p[i]]
        Bp[:, j] = b[:, p[j]]
        
        current_cost += best_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Update tabu
        tabu_list[(i, j)] = current_step + tabu_tenure
        tabu_list[(j, i)] = current_step + tabu_tenure
        
        # Cleanup old tabu entries
        if current_step % 100 == 0:
            keys_to_remove = [k for k, v in tabu_list.items() if v <= current_step]
            for k in keys_to_remove:
                del tabu_list[k]
                
        current_step += 1

    return best_p.tolist()

# EVOLVE-BLOCK-END
