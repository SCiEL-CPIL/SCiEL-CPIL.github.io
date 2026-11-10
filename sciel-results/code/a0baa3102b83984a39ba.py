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
    
    # Run multiple iterations of Greedy + SA
    num_restarts = 10  # Increase restarts, each will be faster due to optimized delta
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Greedy Construction
        covered = np.zeros(N, dtype=np.uint8)
        cover_count = np.zeros(N, dtype=np.int16)
        classes = []
        available_moduli = set(divisors)
        
        # Greedy Step
        while available_moduli:
            if time.time() > deadline - 20:
                break
            
            best_gain = 0
            best_m = None
            best_r = None
            
            for m in list(available_moduli):
                max_cov = 0
                best_r_for_m = 0
                
                for r in range(m):
                    idx = np.arange(r, N, m)
                    cov = int(np.sum(covered[idx] == 0))
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
            covered[idx] = 1
            cover_count[idx] += 1
            classes.append([m, r])
            available_moduli.remove(m)
            
        cur_uncovered = int(np.sum(covered == 0))
        
        if cur_uncovered < best_uncovered:
            best_uncovered = cur_uncovered
            best_classes = [c[:] for c in classes]
        
        # Local Search (Simulated Annealing)
        if time.time() > deadline - 10:
            continue
            
        # Prepare for SA
        current_uncovered = cur_uncovered
        current_classes = [c[:] for c in classes]
        current_cover_count = cover_count.copy()
        
        # SA Parameters
        T = 5.0  # Higher initial temperature
        cooling = 0.995 # Slower cooling
        min_T = 0.01
        max_steps = 10000 # More steps possible due to speedup
        
        used_moduli = {c[0] for c in current_classes}
        unused_moduli = [m for m in divisors if m not in used_moduli]
        
        def compute_delta(old_idx, new_idx, cover_count, current_uncovered, is_removal=False):
            """
            Compute the change in uncovered count when replacing old_idx with new_idx.
            If is_removal is True, new_idx is empty (removing a class).
            If adding a class, old_idx is empty.
            
            We need to be careful: cover_count is updated in-place by the caller before calling this?
            No, let's have the caller update cover_count, then we calculate delta based on the change in status.
            Actually, it's safer to calculate delta BEFORE updating cover_count globally, or handle it carefully.
            
            Let's do:
            1. Identify affected indices: union of old_idx and new_idx.
            2. For each affected index, determine old status (covered/uncovered) and new status.
            3. Sum the differences.
            
            Old status: cover_count[idx] == 0
            New status: (cover_count[idx] + delta_change) == 0
            delta_change is -1 for old_idx (if removing), +1 for new_idx (if adding).
            
            Wait, the move is: Remove class i (old_m, old_r) and Add class j (new_m, new_r).
            So for indices in old_idx: count decreases by 1.
            For indices in new_idx: count increases by 1.
            For indices in intersection: count stays same.
            
            Let's just update the cover_count array for the affected indices and count how many went from 0 to >0 or >0 to 0.
            """
            if len(old_idx) == 0 and len(new_idx) == 0:
                return 0
            
            # Create a set of affected indices to avoid double counting in intersection?
            # No, we can just iterate over the union.
            # But creating a set of up to N elements is slow.
            
            # Better approach:
            # 1. Save old counts for affected indices.
            # 2. Apply changes to cover_count.
            # 3. Count new zeros.
            # 4. Delta = new_zeros - old_zeros.
            
            # Get unique affected indices
            # Since N is 52k, and m >= 3, max indices is ~17k.
            # Using a set is okay if we do it once per move.
            
            # Optimization: 
            # We can just compute the number of indices in old_idx that were 0 (and become 1? No, they decrease).
            # Indices in old_idx with count 1 -> become 0 (uncovered increases by 1).
            # Indices in new_idx with count 0 -> become 1 (uncovered decreases by 1).
            
            # Let's extract the counts for old_idx and new_idx BEFORE modification.
            old_counts = current_cover_count[old_idx] if len(old_idx) > 0 else np.array([], dtype=np.int16)
            new_counts = current_cover_count[new_idx] if len(new_idx) > 0 else np.array([], dtype=np.int16)
            
            # Number of indices in old_idx that are exactly 1 (will become 0)
            # Note: If an index is in both old and new, it is handled by both?
            # No, if it's in both, count changes by -1 + 1 = 0. So it doesn't change status.
            # So we can just look at old_idx and new_idx separately, BUT we must exclude the intersection from the "status change" calculation?
            # Actually, if idx is in intersection:
            # Old count: c. New count: c. Status unchanged.
            # So we only care about (old_idx - new_idx) and (new_idx - old_idx).
            
            # Let's find intersection
            # Using sets for indices
            # This might be slow if sets are large.
            
            # Alternative:
            # Just update the array and count zeros in the union.
            # Union size is at most N.
            # But we only touch N/m elements.
            
            # Let's try the set approach for indices, it's robust.
            old_set = set(old_idx.tolist())
            new_set = set(new_idx.tolist())
            
            # Indices that only in old: count decreases. If count was 1, becomes 0. Uncovered +1.
            # Indices that only in new: count increases. If count was 0, becomes 1. Uncovered -1.
            # Indices in both: no change.
            
            only_old = old_set - new_set
            only_new = new_set - old_set
            
            delta = 0
            
            if only_old:
                # Get counts for only_old
                # These will be decremented.
                # If count == 1, they become 0.
                # We need to check current cover_count for these indices.
                counts_only_old = current_cover_count[list(only_old)]
                delta += int(np.sum(counts_only_old == 1))
                
            if only_new:
                # These will be incremented.
                # If count == 0, they become 1.
                counts_only_new = current_cover_count[list(only_new)]
                delta -= int(np.sum(counts_only_new == 0))
                
            return delta

        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if T < min_T:
                break
                
            # Pick a random move type
            # 1. Change residue of an existing class (70%)
            # 2. Swap modulus of an existing class with an unused one (20%)
            # 3. Remove a random class (10%)
            
            move_type = random.random()
            
            if move_type < 0.7:
                # Change residue
                if not current_classes:
                    continue
                i = random.randint(0, len(current_classes) - 1)
                m, old_r = current_classes[i]
                new_r = random.randint(0, m - 1)
                if new_r == old_r:
                    continue
                
                old_idx = np.arange(old_r, N, m)
                new_idx = np.arange(new_r, N, m)
                
                # Calculate delta before updating
                delta = compute_delta(old_idx, new_idx, current_cover_count, current_uncovered)
                
                # Acceptance probability
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Accept
                    current_uncovered += delta
                    # Update cover_count
                    current_cover_count[old_idx] -= 1
                    current_cover_count[new_idx] += 1
                    current_classes[i][1] = new_r
                # Else reject, no changes to state
                
            elif move_type < 0.9:
                # Swap modulus
                if not current_classes or not unused_moduli:
                    continue
                
                i = random.randint(0, len(current_classes) - 1)
                old_m, old_r = current_classes[i]
                new_m = random.choice(unused_moduli)
                new_r = random.randint(0, new_m - 1)
                
                old_idx = np.arange(old_r, N, old_m)
                new_idx = np.arange(new_r, N, new_m)
                
                delta = compute_delta(old_idx, new_idx, current_cover_count, current_uncovered)
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Accept
                    current_uncovered += delta
                    current_cover_count[old_idx] -= 1
                    current_cover_count[new_idx] += 1
                    current_classes[i][0] = new_m
                    current_classes[i][1] = new_r
                    # Update used/unused moduli
                    used_moduli.remove(old_m)
                    used_moduli.add(new_m)
                    unused_moduli.remove(new_m)
                    unused_moduli.append(old_m)
                    
            else:
                # Remove a class
                if not current_classes:
                    continue
                
                i = random.randint(0, len(current_classes) - 1)
                old_m, old_r = current_classes[i]
                old_idx = np.arange(old_r, N, old_m)
                
                # new_idx is empty
                # Delta calculation:
                # only_old = old_idx, only_new = empty
                # delta = count of indices in old_idx with count == 1
                # These become 0.
                counts_old = current_cover_count[old_idx]
                delta = int(np.sum(counts_old == 1))
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Accept
                    current_uncovered += delta
                    current_cover_count[old_idx] -= 1
                    # Remove class
                    removed_class = current_classes.pop(i)
                    used_moduli.remove(removed_class[0])
                    unused_moduli.append(removed_class[0])
            
            T *= cooling
            
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_classes = [c[:] for c in current_classes]
                
    return best_classes
