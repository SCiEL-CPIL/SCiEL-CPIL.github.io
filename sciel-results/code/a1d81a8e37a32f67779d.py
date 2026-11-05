"""
Combined solver for B2[2] sets in 1..1000.

Strategy:
- Use the structural idea from the current program (greedy baseline + local search with swaps)
- Incorporate the refinement machinery from inspiration 1 (vectorized NumPy feasibility checks,
  phased local improvement with "remove-one-add-two" swaps)
- Use vectorized NumPy for fast counting of pair sums
- Phased local search: (1) pure adds, (2) remove-one-add-one swaps, (3) remove-one-add-two swaps
- Deadline guard to ensure we return within the time limit
"""

import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s limit
    N = 1000
    MAX_REP = 2

    # Phase 1: Fast greedy using NumPy
    arr = []
    counts = np.zeros(2 * N + 1, dtype=np.int32)
    
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        
        # Check if x can be added
        if arr:
            arr_np = np.array(arr)
            new_sums = np.concatenate([arr_np + x, [2 * x]])
            if (counts[new_sums] + 1).max() > MAX_REP:
                continue
        else:
            new_sums = np.array([2 * x])
        
        # Add x
        arr.append(x)
        if arr_np.size > 0:
            counts[arr_np + x] += 1
        counts[2 * x] += 1

    # Phase 2: Local search with phased improvements
    arr_set = set(arr)
    missing = [x for x in range(1, N + 1) if x not in arr_set]
    
    rng = random.Random(42)  # deterministic
    
    def rebuild_counts(ch):
        """Rebuild counts array from a list of chosen elements."""
        cc = np.array(ch, dtype=np.int64)
        nn = len(cc)
        if nn == 0:
            return np.zeros(2 * N + 1, dtype=np.int32)
        ii, jj = np.triu_indices(nn)
        return np.bincount(cc[ii] + cc[jj], minlength=2 * N + 1).astype(np.int32)
    
    def try_add_vectorized(x, cur_arr, cur_counts):
        """Check if x can be added. Returns (ok, new_sums)."""
        if len(cur_arr) == 0:
            ns = np.array([2 * x], dtype=np.int64)
        else:
            ns = np.concatenate([np.array(cur_arr) + x, np.array([2 * x], dtype=np.int64)])
        if (cur_counts[ns] + 1).max() <= MAX_REP:
            return True, ns
        return False, ns
    
    improved = True
    max_rounds = 1000
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        if not missing:
            break
        
        improved = False
        arr_np = np.array(arr, dtype=np.int64)
        n = len(arr)
        
        # Phase 2a: Try pure adds (vectorized)
        if missing:
            oa = np.array(missing, dtype=np.int64)
            if n == 0:
                new_sums_all = 2 * oa[:, None]
            else:
                cross = oa[:, None] + arr_np[None, :]
                new_sums_all = np.concatenate([cross, 2 * oa[:, None]], axis=1)
            
            gathered = counts[new_sums_all] + 1
            cand_max = gathered.max(axis=1)
            feasible = np.where(cand_max <= MAX_REP)[0]
            
            if feasible.size > 0:
                # Pick the one that leaves most room (minimize max count)
                tight = gathered[feasible].max(axis=1)
                bi = int(feasible[int(np.argmin(tight))])
                x = int(oa[bi])
                ns = new_sums_all[bi]
                
                arr.append(x)
                arr_set.add(x)
                missing.remove(x)
                counts[ns] += 1
                improved = True
                continue
        
        # Phase 2b: Try swaps - remove one, add one
        if missing and n > 0:
            batch_size = min(n, 20)
            found_swap = False
            
            for _ in range(batch_size):
                if time.time() > deadline:
                    break
                
                # Pick a random element to remove
                rem_idx = rng.randrange(n)
                rem_val = arr[rem_idx]
                
                # Build trial list without the removed element
                trial = [arr[j] for j in range(n) if j != rem_idx]
                trial_np = np.array(trial, dtype=np.int64)
                cnt_t = rebuild_counts(trial)
                
                # Check if we can add any missing element
                oa = np.array(missing, dtype=np.int64)
                if len(trial) == 0:
                    new_sums2 = 2 * oa[:, None]
                else:
                    cross2 = oa[:, None] + trial_np[None, :]
                    new_sums2 = np.concatenate([cross2, 2 * oa[:, None]], axis=1)
                
                gathered2 = cnt_t[new_sums2] + 1
                cand_max2 = gathered2.max(axis=1)
                feas2 = np.where(cand_max2 <= MAX_REP)[0]
                
                if feas2.size > 0:
                    # Pick the best candidate
                    tight2 = gathered2[feas2].max(axis=1)
                    bi2 = int(feas2[int(np.argmin(tight2))])
                    x = int(oa[bi2])
                    ns2 = new_sums2[bi2]
                    
                    # Perform the swap
                    arr = trial
                    arr.append(x)
                    arr_set = set(arr)
                    missing.remove(x)
                    counts = cnt_t
                    counts[ns2] += 1
                    n = len(arr)
                    found_swap = True
                    improved = True
                    break
            
            if found_swap:
                continue
        
        # Phase 2c: Try remove-one-add-two (net +1)
        if len(missing) >= 2 and n > 0:
            batch_size = min(n, 10)
            found_double = False
            
            for _ in range(batch_size):
                if time.time() > deadline:
                    break
                
                # Pick a random element to remove
                rem_idx = rng.randrange(n)
                rem_val = arr[rem_idx]
                
                # Build trial list without the removed element
                trial = [arr[j] for j in range(n) if j != rem_idx]
                trial_np = np.array(trial, dtype=np.int64)
                cnt_t = rebuild_counts(trial)
                
                # Try to add two missing elements
                oa = np.array(missing, dtype=np.int64)
                
                # First, find all elements that can be added individually
                if len(trial) == 0:
                    new_sums_first = 2 * oa[:, None]
                else:
                    cross_first = oa[:, None] + trial_np[None, :]
                    new_sums_first = np.concatenate([cross_first, 2 * oa[:, None]], axis=1)
                
                gathered_first = cnt_t[new_sums_first] + 1
                cand_max_first = gathered_first.max(axis=1)
                feas_first = np.where(cand_max_first <= MAX_REP)[0]
                
                if feas_first.size < 2:
                    continue
                
                # Try pairs of feasible candidates
                best_pair = None
                best_score = float('inf')
                
                for i in range(min(len(feas_first), 10)):
                    for j in range(i + 1, min(len(feas_first), 10)):
                        x1 = int(oa[feas_first[i]])
                        x2 = int(oa[feas_first[j]])
                        
                        # Check if we can add both x1 and x2
                        # First add x1
                        if len(trial) == 0:
                            ns1 = np.array([2 * x1], dtype=np.int64)
                        else:
                            ns1 = np.concatenate([trial_np + x1, np.array([2 * x1], dtype=np.int64)])
                        
                        cnt_temp = cnt_t.copy()
                        cnt_temp[ns1] += 1
                        
                        # Now check if x2 can be added
                        trial_extended = trial + [x1]
                        trial_ext_np = np.array(trial_extended, dtype=np.int64)
                        
                        if len(trial_extended) == 0:
                            ns2 = np.array([2 * x2], dtype=np.int64)
                        else:
                            ns2 = np.concatenate([trial_ext_np + x2, np.array([2 * x2], dtype=np.int64)])
                        
                        if (cnt_temp[ns2] + 1).max() <= MAX_REP:
                            # This pair works! Score by how much room we leave
                            total_sums = np.concatenate([ns1, ns2])
                            final_counts = cnt_temp.copy()
                            final_counts[ns2] += 1
                            score = final_counts[total_sums].max()
                            
                            if score < best_score:
                                best_score = score
                                best_pair = (x1, x2, ns1, ns2)
                
                if best_pair is not None:
                    x1, x2, ns1, ns2 = best_pair
                    
                    # Perform the double add
                    arr = trial
                    arr.append(x1)
                    arr.append(x2)
                    arr_set = set(arr)
                    missing.remove(x1)
                    missing.remove(x2)
                    counts = cnt_t
                    counts[ns1] += 1
                    counts[ns2] += 1
                    n = len(arr)
                    found_double = True
                    improved = True
                    break
            
            if found_double:
                continue
        
        if not improved:
            break

    return arr


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"First 20: {result[:20]}")
