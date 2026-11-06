import numpy as np
import time
import random

def construct():
    base = 51975
    start_time = time.time()
    
    # Get all divisors > 1 of base
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    divisors.sort()
    
    # We will maintain:
    # covered: np.array of bools, size base
    # classes: list of [m, r]
    # used_moduli: set of m in classes
    
    def get_best_residue(m, uncovered_mask):
        """Find residue r in [0, m) that covers most currently uncovered residues."""
        # uncovered_mask is a bool array of length base
        # We want to maximize sum(uncovered_mask[r::m])
        # For large m, the slice is short. For small m, it's long.
        # Efficient way: 
        # If m is small, we can loop r.
        # If m is large, m is close to base, so slices are tiny.
        
        best_r = 0
        best_count = -1
        
        # Optimization: If m > base // 2, each residue class has at most 2 elements.
        # General case:
        # We can use a trick: 
        # counts = np.zeros(m)
        # for r in range(m):
        #     counts[r] = uncovered_mask[r::m].sum()
        # This is O(base) total operations if implemented with numpy sums on slices?
        # No, slicing creates views, .sum() is O(len(slice)).
        # Total sum of lengths is base. So it is O(base) per modulus.
        
        # Let's just do it.
        # To speed up, we can skip if current best is already very high?
        # No, just compute.
        
        # Vectorized approach for finding best r:
        # This is tricky with standard numpy for arbitrary strides without loops.
        # However, base is only 51975.
        # A pure python loop for r in range(m) calling .sum() on a slice might be slow if m is large (50k).
        # But if m is 50k, slice length is 1 or 2.
        # If m is 3, slice length is 17k.
        
        # Let's use a hybrid:
        if m < 1000:
            # m is small, slices are long. Number of slices is small.
            best_r = 0
            best_count = -1
            for r in range(m):
                cnt = uncovered_mask[r::m].sum()
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
        else:
            # m is large, slices are short. Number of slices is large.
            # We can iterate over the array and accumulate?
            # Or just loop r. 50k iterations of .sum() on length 1-2 array.
            # This might be slow in Python.
            # Alternative: 
            # The value is sum of uncovered_mask at indices r, r+m, ...
            # Since m is large, there are at most 2 indices per r.
            # We can compute this faster.
            
            # Let's try a different approach for large m:
            # Iterate through all indices i where uncovered_mask[i] is True.
            # For each i, it contributes to residue r = i % m.
            # So we can build a histogram of residues for uncovered elements.
            
            # Get indices of uncovered elements
            # This is O(base) to get indices, O(k) to histogram.
            # If k (uncovered count) is small, this is fast.
            # If k is large, m is likely small (handled above).
            
            # Actually, if m is large, k could be anything.
            # But base is 51975.
            # Let's just use the histogram method for m > 1000.
            # It requires np.nonzero which is O(base).
            
            indices = np.nonzero(uncovered_mask)[0]
            if len(indices) == 0:
                return 0, 0
            
            residues = indices % m
            # Count occurrences
            # np.bincount is fast
            counts = np.bincount(residues, minlength=m)
            best_r = int(np.argmax(counts))
            best_count = int(counts[best_r])
            
        return best_r, best_count

    def update_coverage(m, r, covered, add=True):
        """Update covered array by adding or removing class (m, r)."""
        if add:
            covered[r::m] = True
        else:
            # We need to know if other classes cover these spots.
            # So we can't just set to False.
            # This function should be used carefully or we recalculate.
            # For local search, it's better to recalculate the specific slice 
            # or maintain a count array.
            pass

    # Strategy: 
    # 1. Greedy construction
    # 2. Local search with "re-optimize one modulus"
    
    covered = np.zeros(base, dtype=bool)
    classes = []
    used_moduli = set()
    
    # Greedy Phase
    # Time limit for greedy: 10s
    while time.time() - start_time < 10:
        best_m = None
        best_r = None
        best_gain = 0
        
        # Try adding each unused divisor
        for m in divisors:
            if m in used_moduli:
                continue
            
            r, gain = get_best_residue(m, ~covered)
            
            if gain > best_gain:
                best_gain = gain
                best_m = m
                best_r = r
        
        if best_m is None or best_gain == 0:
            break
        
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        covered[best_r::best_m] = True
        
        # Early stop if fully covered
        if not (~covered).any():
            break

    # Local Search Phase
    # Time limit: 110s
    # Moves:
    # 1. Re-optimize residue of an existing modulus
    # 2. Swap an existing modulus with an unused one
    
    random.seed(42)
    
    while time.time() - start_time < 110:
        if not classes:
            break
            
        # Randomly pick a type of move
        if random.random() < 0.7:
            # Move 1: Re-optimize residue
            idx = random.randint(0, len(classes) - 1)
            m, r = classes[idx]
            
            # Remove this class from covered to see what's uncovered by others
            # We need to temporarily "uncover" the residues covered by (m, r) 
            # BUT only if they aren't covered by others.
            # This is hard to do incrementally without a count array.
            # Let's use a count array approach for local search?
            # Or just recalculate the gain for the new residue vs old?
            
            # Gain of current r:
            # It's hard to know exactly how many *new* covers it provides without 
            # knowing the background.
            
            # Alternative: 
            # Calculate the "uncovered mask" if we removed (m, r).
            # This requires knowing which residues are covered ONLY by (m, r).
            
            # Let's switch to a Count Array for the local search phase.
            # count[i] = number of classes covering residue i.
            # covered[i] = count[i] > 0.
            
            pass # Placeholder, we will implement count array below

    # Re-implementation with Count Array for Local Search
    # This allows safe removal/addition.
    
    # Re-initialize for the local search phase using the greedy result
    count = np.zeros(base, dtype=np.int32)
    for m, r in classes:
        count[r::m] += 1
        
    def get_uncovered_mask(count_arr):
        return count_arr == 0

    # We will modify the classes and count array.
    
    # Helper to apply a class
    def apply_class(m, r, count_arr, add=True):
        if add:
            count_arr[r::m] += 1
        else:
            count_arr[r::m] -= 1
            
    # Helper to get best residue for m given current count_arr (which includes m? or not?)
    # If we are re-optimizing m, we should exclude m from the "background".
    # So we pass the count array WITHOUT m.
    
    def get_best_residue_excluding(m, count_arr_without_m):
        # Uncovered mask based on count_arr_without_m
        uncovered = (count_arr_without_m == 0)
        # Find r maximizing sum(uncovered[r::m])
        # Same logic as before
        if m < 1000:
            best_r = 0
            best_count = -1
            for r in range(m):
                cnt = uncovered[r::m].sum()
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
        else:
            indices = np.nonzero(uncovered)[0]
            if len(indices) == 0:
                return 0, 0
            residues = indices % m
            counts = np.bincount(residues, minlength=m)
            best_r = int(np.argmax(counts))
            best_count = int(counts[best_r])
        return best_r, best_count

    # Local Search Loop
    current_uncovered = (~ (count > 0)).sum()
    
    while time.time() - start_time < 110:
        if not classes:
            break
            
        move_type = random.random()
        
        if move_type < 0.7:
            # Re-optimize residue of a random class
            idx = random.randint(0, len(classes) - 1)
            m, r_old = classes[idx]
            
            # Temporarily remove m
            apply_class(m, r_old, count, add=False)
            
            # Find best r_new
            r_new, gain_new = get_best_residue_excluding(m, count)
            
            # Compare gain_new with gain_old?
            # gain_new is the number of *currently uncovered* (by others) residues covered by (m, r_new).
            # The total uncovered after adding (m, r_new) will be:
            # (current_uncovered_excluding_m) - gain_new
            # But we need to compare with the state BEFORE the move.
            
            # Let's calculate the actual change in uncovered count.
            # State before: count includes (m, r_old).
            # State after: count includes (m, r_new).
            
            # Uncovered_before = sum(count == 0)
            # Uncovered_after = sum((count - old_contrib + new_contrib) == 0)
            
            # This is complex to calculate directly.
            # Easier: 
            # 1. Remove (m, r_old).
            # 2. Calculate uncovered_mask_mid = (count == 0).
            # 3. Add (m, r_new).
            # 4. Calculate uncovered_mask_end = (count == 0).
            # 5. If sum(uncovered_mask_end) < sum(uncovered_mask_before), accept.
            
            # We have count (without m).
            uncovered_mid = (count == 0)
            
            # Apply new
            apply_class(m, r_new, count, add=True)
            uncovered_end = (count == 0)
            
            # We need uncovered_before. 
            # We can compute it from the start of the iteration.
            # Let's store it.
            
            # Actually, we can just check if the new configuration is better than the old.
            # We know the old configuration was `classes` with `r_old`.
            # We can compute `uncovered_before` at the start of the loop.
            
            # Let's restructure:
            # 1. Compute uncovered_before
            # 2. Remove old
            # 3. Find best new
            # 4. Add new
            # 5. Compute uncovered_after
            # 6. If after < before, keep. Else, revert.
            
            # We didn't compute uncovered_before in this branch yet.
            # Let's do it.
            
            # Note: `count` currently has (m, r_old) REMOVED.
            # So `uncovered_mid` is the uncovered set if m was absent.
            # `uncovered_end` is the uncovered set if (m, r_new) is present.
            
            # We need `uncovered_before` (with m, r_old).
            # We can reconstruct it or compute it before removing.
            
            # Let's redo this block properly.
            
            # Revert the changes made above for this iteration if we mess up.
            # For now, let's assume we compute `uncovered_before` at the top of the loop.
            
            pass # This logic is getting messy in the draft. I will write the final clean code.

    # Final Clean Implementation Plan:
    # 1. Greedy to get initial classes and count array.
    # 2. Local Search:
    #    - Compute current uncovered count.
    #    - Pick move (re-optimize or swap).
    #    - For re-optimize:
    #       - Save old r.
    #       - Remove (m, r_old) from count.
    #       - Find best r_new for m given current count.
    #       - Add (m, r_new) to count.
    #       - Compute new uncovered count.
    #       - If new < old, accept. Else, revert (remove new, add old).
    #    - For swap:
    #       - Pick random used m_old and random unused m_new.
    #       - Remove (m_old, r_old) from count.
    #       - Find best r_new for m_new given current count.
    #       - Add (m_new, r_new) to count.
    #       - Compute new uncovered count.
    #       - If new < old, accept (update classes list). Else, revert.

    # Let's implement this cleanly.

    # Reset state to Greedy result
    count = np.zeros(base, dtype=np.int32)
    for m, r in classes:
        count[r::m] += 1
        
    current_uncovered = (~ (count > 0)).sum()
    
    while time.time() - start_time < 110:
        if not classes:
            break
            
        # Compute current uncovered count (expensive, but base is small)
        # We can maintain it incrementally, but let's just compute for safety in logic.
        # O(base) is ~50k ops. 1000 iterations/s is fine.
        
        move_type = random.random()
        
        if move_type < 0.7:
            # Re-optimize
            idx = random.randint(0, len(classes) - 1)
            m, r_old = classes[idx]
            
            # Remove old
            count[r_old::m] -= 1
            
            # Find best new
            # Uncovered mask is (count == 0)
            # We want to maximize coverage of (count == 0)
            
            # Use get_best_residue_excluding logic
            if m < 1000:
                best_r = 0
                best_cnt = -1
                # We can optimize: only check r if it might improve?
                # No, just check all.
                # Note: count is the background.
                for r in range(m):
                    # Count how many of count[r::m] are 0
                    # This is sum(count[r::m] == 0)
                    cnt = (count[r::m] == 0).sum()
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_r = r
            else:
                # Histogram approach
                # Indices where count == 0
                zero_indices = np.nonzero(count == 0)[0]
                if len(zero_indices) == 0:
                    best_r = 0
                    best_cnt = 0
                else:
                    residues = zero_indices % m
                    counts = np.bincount(residues, minlength=m)
                    best_r = int(np.argmax(counts))
                    best_cnt = int(counts[best_r])
            
            # Apply new
            count[best_r::m] += 1
            
            # Evaluate
            new_uncovered = (~ (count > 0)).sum()
            
            if new_uncovered < current_uncovered:
                # Accept
                classes[idx] = [m, best_r]
                current_uncovered = new_uncovered
            else:
                # Revert
                count[best_r::m] -= 1
                count[r_old::m] += 1
        else:
            # Swap
            # Pick random used index
            idx = random.randint(0, len(classes) - 1)
            m_old, r_old = classes[idx]
            
            # Pick random unused modulus
            unused = [m for m in divisors if m not in used_moduli]
            if not unused:
                continue # No unused moduli
            
            m_new = random.choice(unused)
            
            # Remove old
            count[r_old::m_old] -= 1
            used_moduli.remove(m_old)
            
            # Find best r for m_new
            if m_new < 1000:
                best_r = 0
                best_cnt = -1
                for r in range(m_new):
                    cnt = (count[r::m_new] == 0).sum()
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_r = r
            else:
                zero_indices = np.nonzero(count == 0)[0]
                if len(zero_indices) == 0:
                    best_r = 0
                    best_cnt = 0
                else:
                    residues = zero_indices % m_new
                    counts = np.bincount(residues, minlength=m_new)
                    best_r = int(np.argmax(counts))
                    best_cnt = int(counts[best_r])
            
            # Apply new
            count[best_r::m_new] += 1
            used_moduli.add(m_new)
            
            # Evaluate
            new_uncovered = (~ (count > 0)).sum()
            
            if new_uncovered < current_uncovered:
                # Accept
                classes[idx] = [m_new, best_r]
                current_uncovered = new_uncovered
            else:
                # Revert
                count[best_r::m_new] -= 1
                used_moduli.remove(m_new)
                count[r_old::m_old] += 1
                used_moduli.add(m_old)
        
        if current_uncovered == 0:
            break

    # Return the best classes found
    return classes
