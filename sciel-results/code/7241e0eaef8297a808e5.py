import time
import numpy as np
import random

def get_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted([d for d in divisors if d > 1])

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    divisors = get_divisors(N)
    
    best_uncovered = N
    best_classes = []
    
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Greedy Construction with incremental tracking
        cover_count = np.zeros(N, dtype=np.int16)
        uncovered_count = N
        classes = []
        available_moduli = list(divisors)
        
        # Greedy Step
        while available_moduli:
            if time.time() > deadline - 20:
                break
            
            best_gain = 0
            best_m = None
            best_r = None
            
            for m in available_moduli:
                # Vectorized: for each residue r, count uncovered in arange(r, N, m)
                # We can do this by reshaping
                # Create a 2D array where row i contains indices i, i+m, i+2m, ...
                # But simpler: iterate residues but use fast counting
                
                max_cov = 0
                best_r_for_m = 0
                
                # Optimization: only check residues that might improve
                # Max possible gain is N/m
                if N // m <= best_gain:
                    continue
                
                for r in range(m):
                    idx = np.arange(r, N, m)
                    cov = int(np.sum(cover_count[idx] == 0))
                    if cov > max_cov:
                        max_cov = cov
                        best_r_for_m = r
                
                if max_cov > best_gain:
                    best_gain = max_cov
                    best_m = m
                    best_r = best_r_for_m
            
            if best_gain <= 0:
                break
            
            m = best_m
            r = best_r
            idx = np.arange(r, N, m)
            
            # Update uncovered count incrementally
            newly_covered = int(np.sum(cover_count[idx] == 0))
            cover_count[idx] += 1
            uncovered_count -= newly_covered
            
            classes.append([m, r])
            available_moduli.remove(m)
        
        cur_uncovered = uncovered_count
        
        if cur_uncovered < best_uncovered:
            best_uncovered = cur_uncovered
            best_classes = [c[:] for c in classes]
        
        # Local Search (Simulated Annealing) with incremental tracking
        if time.time() > deadline - 10:
            continue
            
        current_uncovered = cur_uncovered
        current_classes = [c[:] for c in classes]
        current_cover_count = cover_count.copy()
        
        T = 1.0
        cooling = 0.99
        min_T = 0.01
        max_steps = 8000
        
        used_moduli = {c[0] for c in current_classes}
        unused_moduli = [m for m in divisors if m not in used_moduli]
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if T < min_T:
                break
                
            if len(current_classes) == 0:
                if unused_moduli:
                    m = random.choice(unused_moduli)
                    r = random.randint(0, m-1)
                    idx = np.arange(r, N, m)
                    newly_covered = int(np.sum(current_cover_count[idx] == 0))
                    current_cover_count[idx] += 1
                    current_uncovered -= newly_covered
                    current_classes.append([m, r])
                    used_moduli.add(m)
                    unused_moduli.remove(m)
                else:
                    break
                continue
            
            move_type = random.random()
            
            if move_type < 0.6:
                # Change residue
                i = random.randint(0, len(current_classes) - 1)
                m, old_r = current_classes[i]
                new_r = random.randint(0, m - 1)
                if new_r == old_r:
                    continue
                
                old_idx = np.arange(old_r, N, m)
                new_idx = np.arange(new_r, N, m)
                
                # Compute delta incrementally
                # Positions that lose coverage: old_idx where count becomes 0
                # Positions that gain coverage: new_idx where count was 0
                
                # First, check which old positions will become uncovered
                old_will_uncover = int(np.sum((current_cover_count[old_idx] == 1) & (current_cover_count[old_idx] == 1)))
                # More carefully: positions in old_idx with count==1 that are NOT in new_idx
                # This is complex, so let's just apply and count
                
                # Apply changes
                current_cover_count[old_idx] -= 1
                current_cover_count[new_idx] += 1
                
                # Count how many positions changed from 0 to >0 (gained coverage)
                gained = int(np.sum((current_cover_count[new_idx] > 0) & (current_cover_count[old_idx] == 0)))
                # Count how many positions changed from >0 to 0 (lost coverage)
                lost = int(np.sum((current_cover_count[old_idx] == 0) & (current_cover_count[new_idx] > 0)))
                
                # Actually, simpler: just count uncovered before and after for affected positions
                # But that's O(N) again. Let's be more careful.
                
                # The delta in uncovered count:
                # A position becomes uncovered if it was covered (count>0) and now count==0
                # A position becomes covered if it was uncovered (count==0) and now count>0
                
                # For old_idx: positions where count went from 1 to 0 become uncovered
                # For new_idx: positions where count went from 0 to 1 become covered
                # But overlaps need handling.
                
                # Let's just compute the net change by checking affected positions
                affected = np.union1d(old_idx, new_idx)
                # We need to know the state BEFORE the change
                # We already modified current_cover_count, so let's revert temporarily
                
                # Actually, let's compute delta from the modification:
                # delta_uncovered = (positions that became uncovered) - (positions that became covered)
                
                # Positions that became uncovered: were in old_idx, had count==1 before, now count==0
                # But we already decremented. So before: count was 1, after: count is 0
                # These are positions in old_idx where current_cover_count == 0 AND they weren't in new_idx (or if in new_idx, the net effect is still 0)
                
                # This is getting complex. Let's use a simpler approach:
                # Just recompute for the union of affected indices
                # This is O(|old_idx| + |new_idx|) which is much less than O(N)
                
                # But we already modified the array. Let's compute delta from the modification:
                # For each position in the union, we know old and new count
                # But we don't have old counts anymore.
                
                # Alternative: compute delta as:
                # new_uncovered_in_affected - old_uncovered_in_affected
                # We can get new_uncovered_in_affected from current_cover_count
                # For old_uncovered_in_affected, we need to reverse the changes temporarily
                
                # Let's just do: revert, compute, re-apply. But that's slow.
                
                # Simplest correct approach: compute delta from the changes
                # delta = (# positions that went from covered to uncovered) - (# positions that went from uncovered to covered)
                
                # A position goes from covered to uncovered if:
                #   It was in old_idx, had count==1 before the decrement, and is NOT in new_idx (or if in new_idx, the increment doesn't save it)
                #   Actually: it was in old_idx, count_before == 1, and (not in new_idx OR count_before_in_new_idx == 0)
                #   Wait, if it's in both old_idx and new_idx, the net change is 0, so it stays covered if count_before >= 1
                
                # Let's define:
                #   only_old = old_idx \ new_idx
                #   only_new = new_idx \ old_idx
                #   both = old_idx ∩ new_idx
                
                # For only_old: count decreases by 1. Becomes uncovered if count_before == 1
                # For only_new: count increases by 1. Becomes covered if count_before == 0
                # For both: count unchanged. No change in uncovered status.
                
                # So:
                #   lost_coverage = sum over only_old of [count_before == 1]
                #   gained_coverage = sum over only_new of [count_before == 0]
                #   delta_uncovered = lost_coverage - gained_coverage
                
                # But we already modified current_cover_count. So:
                #   For only_old: count_after = count_before - 1, so count_before = count_after + 1
                #     lost_coverage = sum over only_old of [count_after + 1 == 1] = sum over only_old of [count_after == 0]
                #   For only_new: count_after = count_before + 1, so count_before = count_after - 1
                #     gained_coverage = sum over only_new of [count_after - 1 == 0] = sum over only_new of [count_after == 1]
                
                # Compute the sets
                old_set = set(old_idx.tolist())
                new_set = set(new_idx.tolist())
                only_old = np.array(list(old_set - new_set), dtype=np.int64)
                only_new = np.array(list(new_set - old_set), dtype=np.int64)
                
                lost_coverage = int(np.sum(current_cover_count[only_old] == 0)) if len(only_old) > 0 else 0
                gained_coverage = int(np.sum(current_cover_count[only_new] == 1)) if len(only_new) > 0 else 0
                
                delta = lost_coverage - gained_coverage
                new_uncovered = current_uncovered + delta
                
                if delta < 0 or (delta == 0 and random.random() < 0.1) or (random.random() < np.exp(-delta / T)):
                    # Accept
                    current_uncovered = new_uncovered
                    current_classes[i][1] = new_r
                else:
                    # Reject: revert
                    current_cover_count[new_idx] -= 1
                    current_cover_count[old_idx] += 1
            
            elif move_type < 0.85:
                # Swap modulus
                if not unused_moduli:
                    continue
                
                i = random.randint(0, len(current_classes) - 1)
                old_m, old_r = current_classes[i]
                new_m = random.choice(unused_moduli)
                new_r = random.randint(0, new_m - 1)
                
                old_idx = np.arange(old_r, N, old_m)
                new_idx = np.arange(new_r, N, new_m)
                
                current_cover_count[old_idx] -= 1
                current_cover_count[new_idx] += 1
                
                # Compute delta similarly
                old_set = set(old_idx.tolist())
                new_set = set(new_idx.tolist())
                only_old = np.array(list(old_set - new_set), dtype=np.int64)
                only_new = np.array(list(new_set - old_set), dtype=np.int64)
                
                lost_coverage = int(np.sum(current_cover_count[only_old] == 0)) if len(only_old) > 0 else 0
                gained_coverage = int(np.sum(current_cover_count[only_new] == 1)) if len(only_new) > 0 else 0
                
                delta = lost_coverage - gained_coverage
                new_uncovered = current_uncovered + delta
                
                if delta < 0 or (delta == 0 and random.random() < 0.1) or (random.random() < np.exp(-delta / T)):
                    # Accept
                    current_uncovered = new_uncovered
                    current_classes[i][0] = new_m
                    current_classes[i][1] = new_r
                    used_moduli.remove(old_m)
                    used_moduli.add(new_m)
                    unused_moduli.remove(new_m)
                    unused_moduli.append(old_m)
                else:
                    # Reject
                    current_cover_count[new_idx] -= 1
                    current_cover_count[old_idx] += 1
            
            else:
                # Add or remove a class
                if random.random() < 0.5 and unused_moduli:
                    # Add a class
                    m = random.choice(unused_moduli)
                    r = random.randint(0, m-1)
                    idx = np.arange(r, N, m)
                    
                    gained = int(np.sum(current_cover_count[idx] == 0))
                    current_cover_count[idx] += 1
                    
                    delta = -gained
                    new_uncovered = current_uncovered + delta
                    
                    if delta < 0 or (delta == 0 and random.random() < 0.1) or (random.random() < np.exp(-delta / T)):
                        current_uncovered = new_uncovered
                        current_classes.append([m, r])
                        used_moduli.add(m)
                        unused_moduli.remove(m)
                    else:
                        current_cover_count[idx] -= 1
                elif len(current_classes) > 1:
                    # Remove a class
                    i = random.randint(0, len(current_classes) - 1)
                    m, r = current_classes[i]
                    idx = np.arange(r, N, m)
                    
                    current_cover_count[idx] -= 1
                    lost = int(np.sum(current_cover_count[idx] == 0))
                    
                    delta = lost
                    new_uncovered = current_uncovered + delta
                    
                    if delta < 0 or (delta == 0 and random.random() < 0.1) or (random.random() < np.exp(-delta / T)):
                        current_uncovered = new_uncovered
                        current_classes.pop(i)
                        used_moduli.remove(m)
                        unused_moduli.append(m)
                    else:
                        current_cover_count[idx] += 1
            
            T *= cooling
            
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_classes = [c[:] for c in current_classes]
        
        # End SA for this restart
        
    return best_classes
