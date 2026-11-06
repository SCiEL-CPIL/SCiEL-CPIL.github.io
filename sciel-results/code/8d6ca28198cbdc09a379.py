import time
import random
import numpy as np

def construct():
    deadline = time.time() + 145.0

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if not S:
            return True
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_instance(vals, time_limit=40.0):
        n = len(vals)
        start_time = time.time()
        
        # Map values to indices for bitset operations
        val_to_idx = {v: i for i, v in enumerate(vals)}
        vals_arr = np.array(vals)
        
        # Precompute sum indices: sum_idx[i, j] = index of vals[i] + vals[j] if in vals, else -1
        # We need a fast way to check if a value is in S.
        # We'll maintain two boolean numpy arrays: in_S and in_F (sums of elements in S).
        
        def get_sum_indices():
            # Returns a 2D array of indices, or -1 if sum not in vals
            # This is O(N^2) memory. For N=240, 240*240 = 57600 ints. Fine.
            # But we only need it for updates.
            # Actually, we can just compute on the fly or precompute a dictionary of sums.
            pass

        # Precompute: for each pair (i, j), what is the index of vals[i] + vals[j]?
        sum_map = {}
        for i in range(n):
            vi = vals[i]
            for j in range(i, n): # Since a+b=b+a, and a=a+b implies 2a
                s = vi + vals[j]
                if s in val_to_idx:
                    sum_map[(i, j)] = val_to_idx[s]
        
        # To speed up "add x", we need to know which existing elements s in S create sums that land in S.
        # And which sums land on x.
        
        # State: in_S (bool array), in_F (bool array)
        # in_F[k] is True if vals[k] = vals[a] + vals[b] for some a,b in S.
        
        def init_state(S_init):
            in_S = np.zeros(n, dtype=bool)
            for x in S_init:
                in_S[val_to_idx[x]] = True
            in_F = np.zeros(n, dtype=bool)
            # Compute initial F
            S_indices = np.where(in_S)[0]
            for i in S_indices:
                for j in S_indices:
                    s_idx = sum_map.get((i, j), sum_map.get((j, i), -1))
                    if s_idx != -1:
                        in_F[s_idx] = True
            return in_S, in_F

        def check_valid(in_S, in_F):
            # Valid if no element in S is in F
            return not np.any(in_S & in_F)

        def try_add(in_S, in_F, idx):
            # Returns (new_in_S, new_in_F, valid)
            if in_S[idx]:
                return in_S, in_F, False
            if in_F[idx]:
                # x is a sum of two elements in S, so adding x creates a+b=x with x in S. Invalid.
                return in_S, in_F, False
            
            # If we add x, new sums are x + s for all s in S, and 2x.
            # We need to check if any of these new sums land in S (including x itself? No, x is not sum of x+x unless 2x=x => x=0, but x!=0).
            # Wait, if x+s = y where y in S, then adding x creates x+s=y with x,s,y in S. Invalid.
            # So we need: (x + S) ∩ S == empty.
            # And we need to update F.
            
            S_indices = np.where(in_S)[0]
            if len(S_indices) == 0:
                # Can always add if S is empty
                new_in_S = in_S.copy()
                new_in_S[idx] = True
                # New F: just {2x} if 2x in vals
                new_in_F = in_F.copy()
                two_x_idx = sum_map.get((idx, idx), -1)
                if two_x_idx != -1:
                    new_in_F[two_x_idx] = True
                return new_in_S, new_in_F, True
            
            # Check conflicts: does x + s land in S for any s in S?
            # We can check this by iterating S_indices.
            for s_idx in S_indices:
                # Check x + vals[s_idx]
                s_val = vals[s_idx]
                target = vals[idx] + s_val
                if target in val_to_idx:
                    t_idx = val_to_idx[target]
                    if in_S[t_idx]:
                        return in_S, in_F, False
                # Also check 2x?
                # If 2x is in S, then x+x=2x with x,2x in S.
                # This is covered if s_idx == idx? No, s_idx is from current S, x is not in S yet.
                # So 2x conflict is separate.
            two_x_idx = sum_map.get((idx, idx), -1)
            if two_x_idx != -1 and in_S[two_x_idx]:
                return in_S, in_F, False
                
            # Move is valid. Update F.
            new_in_S = in_S.copy()
            new_in_S[idx] = True
            new_in_F = in_F.copy()
            
            # Add new sums to F
            for s_idx in S_indices:
                target = vals[idx] + vals[s_idx]
                if target in val_to_idx:
                    new_in_F[val_to_idx[target]] = True
            if two_x_idx != -1:
                new_in_F[two_x_idx] = True
                
            return new_in_S, new_in_F, True

        def try_remove(in_S, in_F, idx):
            # Removing an element always makes the set "more sum-free" in terms of constraints?
            # No, it reduces size. We only remove if it allows adding more?
            # Actually, removing an element never creates a new sum-free violation.
            # It only removes sums from F.
            # So removal is always "safe" but reduces objective.
            # We use removal only in swaps or if stuck.
            new_in_S = in_S.copy()
            new_in_S[idx] = False
            # Recalculating F from scratch is O(N^2).
            # For now, we'll just recalculate F if we need to remove.
            S_indices = np.where(new_in_S)[0]
            new_in_F = np.zeros(n, dtype=bool)
            for i in S_indices:
                for j in S_indices:
                    s_idx = sum_map.get((i, j), sum_map.get((j, i), -1))
                    if s_idx != -1:
                        new_in_F[s_idx] = True
            return new_in_S, new_in_F

        # Simulated Annealing / Random Restart Local Search
        
        best_S = set()
        best_size = 0
        
        # Generate initial candidates
        candidates = []
        # 1. Odds
        odds = [x for x in vals if x % 2 != 0]
        candidates.append(set(odds))
        # 2. Positives
        pos = [x for x in vals if x > 0]
        candidates.append(set(pos))
        # 3. Negatives
        neg = [x for x in vals if x < 0]
        candidates.append(set(neg))
        # 4. Greedy by absolute value
        sorted_abs = sorted(vals, key=lambda x: -abs(x))
        S_greedy = set()
        for x in sorted_abs:
            # Quick check
            valid = True
            for y in S_greedy:
                if x + y in S_greedy or y + x in S_greedy or 2*x in S_greedy:
                    valid = False
                    break
                if x - y in S_greedy or y - x in S_greedy: # Wait, x-y in S means x = y + (x-y). 
                    # If x, y, x-y are in S, then y + (x-y) = x. Violation.
                    valid = False
                    break
            if valid:
                S_greedy.add(x)
        candidates.append(S_greedy)
        
        # 5. Random samples
        rng = random.Random(123)
        for _ in range(20):
            sample = rng.sample(vals, n//2 + 1)
            candidates.append(set(sample))
            
        for S_init in candidates:
            if time.time() > start_time + time_limit:
                break
            if not is_sum_free(S_init):
                continue
                
            in_S, in_F = init_state(S_init)
            if not check_valid(in_S, in_F):
                continue
                
            current_size = np.sum(in_S)
            if current_size > best_size:
                best_size = int(current_size)
                best_S = {vals[i] for i in range(n) if in_S[i]}
                
            # Local Search: Try to add random elements, if fail, try swap?
            # Since add is O(N), we can try all possible adds.
            # If no add works, we are in a local optimum for "add" moves.
            # To escape, we need "swap" (remove one, add another) or "2-swap".
            # Given time constraints, let's stick to aggressive "Add" with random restarts.
            # Or, implement a simple "Remove and Add" (1-swap).
            
            # Optimize: Precompute list of non-in-S indices
            # We will do a loop of iterations
            improved = True
            while improved and (time.time() - start_time) < time_limit:
                improved = False
                # Try to add any element
                non_in_S = np.where(~in_S)[0]
                # Shuffle to break symmetry
                rng.shuffle(non_in_S)
                for idx in non_in_S:
                    new_in_S, new_in_F, valid = try_add(in_S, in_F, idx)
                    if valid:
                        in_S, in_F = new_in_S, new_in_F
                        current_size += 1
                        improved = True
                        if current_size > best_size:
                            best_size = int(current_size)
                            best_S = {vals[i] for i in range(n) if in_S[i]}
                        # Break to restart the loop with new state? 
                        # Yes, because adding one might enable others or block others.
                        break
                
                if not improved:
                    # Local optimum reached for pure additions.
                    # Try a swap: remove a random element, then try to add others.
                    # This is expensive. Let's just do a few random restarts of the whole process instead?
                    # Or, implement a limited swap.
                    
                    # Let's try one swap step:
                    in_S_indices = np.where(in_S)[0]
                    if len(in_S_indices) > 0:
                        # Pick a random element to remove
                        rem_idx = rng.choice(in_S_indices)
                        # Try removing it and then adding greedily
                        trial_in_S, trial_in_F = try_remove(in_S, in_F, rem_idx)
                        trial_size = np.sum(trial_in_S)
                        
                        # Try to add back others
                        trial_improved = True
                        while trial_improved:
                            trial_improved = False
                            trial_non_in_S = np.where(~trial_in_S)[0]
                            rng.shuffle(trial_non_in_S)
                            for add_idx in trial_non_in_S:
                                new_trial_S, new_trial_F, valid = try_add(trial_in_S, trial_in_F, add_idx)
                                if valid:
                                    trial_in_S, trial_in_F = new_trial_S, new_trial_F
                                    trial_size += 1
                                    trial_improved = True
                                    break
                            
                        if trial_size > current_size:
                            in_S, in_F = trial_in_S, trial_in_F
                            current_size = trial_size
                            improved = True
                            if current_size > best_size:
                                best_size = int(current_size)
                                best_S = {vals[i] for i in range(n) if in_S[i]}
                            # If we improved, continue the outer while loop to try more adds
                        else:
                            # Swap didn't help, revert
                            pass
                    else:
                        break

        # Final check
        if not is_sum_free(best_S):
            # Fallback to odds
            best_S = set(x for x in vals if x % 2 != 0)
            
        return best_S

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        
        if size == 60:
            t_lim = 25.0
        elif size == 120:
            t_lim = 45.0
        else:
            t_lim = 65.0
            
        S = solve_instance(vals, t_lim)
        
        if not is_sum_free(S):
            S = set(x for x in vals if x % 2 != 0)
        
        results.append(sorted(S))
        
        if time.time() > deadline:
            break

    while len(results) < 3:
        results.append([])

    return results[:3]
