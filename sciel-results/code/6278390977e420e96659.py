import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin for 150s kill
    
    base = 51975
    
    # Get all odd divisors > 1 of 51975
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # Precompute residue class indices for each divisor as numpy arrays
    class_indices = {}
    for m in divisors:
        class_indices[m] = [np.arange(r, base, m) for r in range(m)]
    
    def count_uncovered(covered):
        return int(np.count_nonzero(covered == 0))
    
    def greedy_initial(order):
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        for m in order:
            if time.time() >= deadline:
                break
            best_r = 0
            best_count = -1
            for r in range(m):
                idxs = class_indices[m][r]
                cnt = int(np.count_nonzero(covered[idxs] == 0))
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            classes.append([m, best_r])
            covered[class_indices[m][best_r]] = 1
        return classes, covered, count_uncovered(covered)
    
    # Try multiple greedy orderings
    best_classes = None
    best_uncovered = base
    best_covered = None
    
    # Standard increasing order
    classes, covered, unc = greedy_initial(divisors)
    if unc < best_uncovered:
        best_uncovered = unc
        best_classes = [c[:] for c in classes]
        best_covered = covered.copy()
    
    # Try decreasing order
    if time.time() < deadline:
        classes, covered, unc = greedy_initial(sorted(divisors, reverse=True))
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = [c[:] for c in classes]
            best_covered = covered.copy()
    
    # Try random orderings
    random.seed(42)
    for attempt in range(30):
        if time.time() >= deadline - 40:
            break
        random_order = divisors.copy()
        random.shuffle(random_order)
        classes, covered, unc = greedy_initial(random_order)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = [c[:] for c in classes]
            best_covered = covered.copy()
    
    if best_classes is None:
        best_classes, best_covered, best_uncovered = greedy_initial(divisors)
    
    # Local search with vectorized move evaluation
    classes = [c[:] for c in best_classes]
    
    def compute_cover_count(cls):
        cover_count = np.zeros(base, dtype=np.int32)
        for m, r in cls:
            cover_count[class_indices[m][r]] += 1
        return cover_count
    
    cover_count = compute_cover_count(classes)
    uncovered = int(np.count_nonzero(cover_count == 0))
    
    used_moduli = set(m for m, r in classes)
    available_moduli = [m for m in divisors if m not in used_moduli]
    
    improved = True
    max_rounds = 30
    round_num = 0
    
    while improved and time.time() < deadline - 5 and round_num < max_rounds:
        improved = False
        round_num += 1
        
        # Try changing residue for each existing class - vectorized
        for idx in range(len(classes)):
            if time.time() >= deadline - 5:
                break
            m, r = classes[idx]
            old_idxs = class_indices[m][r]
            
            # Vectorized: for each candidate new_r, compute delta
            # delta = (uncovered in new) - (uniquely covered in old that aren't in new)
            # More precisely:
            #   loss = count of i in old_idxs where cover_count[i]==1 and i not in new_idxs
            #   gain = count of i in new_idxs where cover_count[i]==0 and i not in old_idxs
            #   delta = loss - gain
            # Since old_idxs and new_idxs have same size (base//m), and their intersection
            # is complex, let's use a different approach:
            # For each new_r, compute:
            #   new_uncovered_gain = count of i in new_idxs where cover_count[i] == 0
            #   old_unique_loss = count of i in old_idxs where cover_count[i] == 1 and i not in new_idxs
            # But "i not in new_idxs" is the tricky part.
            # Alternative: delta = (count of i where cover_count[i]==0 in new_idxs) 
            #               - (count of i where cover_count[i]==1 in old_idxs but not in new_idxs)
            # We can compute: total_in_old_unique = count(cover_count[old_idxs]==1)
            #                 overlap_unique = count of i in both old and new where cover_count==1
            #                 old_unique_loss = total_in_old_unique - overlap_unique
            # Similarly for gain:
            #                 total_new_uncovered = count(cover_count[new_idxs]==0)
            #                 overlap_uncovered = count of i in both old and new where cover_count==0
            #                 new_uncovered_gain = total_new_uncovered - overlap_uncovered
            # This is still complex. Let's just do it per candidate but vectorized over residues.
            
            best_delta = 0
            best_new_r = None
            
            # Precompute which residues are covered by exactly 1 class
            exactly_one = (cover_count == 1)
            uncovered_mask = (cover_count == 0)
            
            for new_r in range(m):
                if new_r == r:
                    continue
                new_idxs = class_indices[m][new_r]
                
                # Vectorized computation
                # Gain: residues in new_idxs that are currently uncovered
                gain = int(np.count_nonzero(uncovered_mask[new_idxs]))
                # Loss: residues in old_idxs that are uniquely covered AND not in new_idxs
                # A residue i is in old_idxs and not in new_idxs iff (i - r) % m != 0 and (i - new_r) % m == 0
                # Actually simpler: old_idxs = {r, r+m, r+2m, ...}, new_idxs = {new_r, new_r+m, ...}
                # They intersect if r ≡ new_r (mod gcd(m, ...)) - actually they're disjoint if r != new_r mod m
                # Wait, old_idxs and new_idxs are residue classes mod m. If r != new_r, they're disjoint!
                # So if r != new_r: no overlap, gain = count(uncovered in new_idxs), loss = count(exactly_one in old_idxs)
                # If r == new_r: skip (already handled)
                
                loss = int(np.count_nonzero(exactly_one[old_idxs]))
                delta = loss - gain
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_r = new_r
            
            if best_new_r is not None:
                # Apply the move
                m, old_r = classes[idx]
                old_idxs = class_indices[m][old_r]
                new_idxs = class_indices[m][best_new_r]
                
                # Update cover_count
                cover_count[old_idxs] -= 1
                cover_count[new_idxs] += 1
                
                classes[idx][1] = best_new_r
                uncovered += best_delta
                improved = True
        
        # Try adding a new modulus - vectorized
        if time.time() < deadline - 5 and available_moduli:
            best_add_gain = 0
            best_add = None
            
            uncovered_mask = (cover_count == 0)
            
            for m in available_moduli:
                if time.time() >= deadline - 5:
                    break
                best_r_for_m = None
                best_gain_for_m = 0
                
                for r in range(m):
                    idxs = class_indices[m][r]
                    gain = int(np.count_nonzero(uncovered_mask[idxs]))
                    if gain > best_gain_for_m:
                        best_gain_for_m = gain
                        best_r_for_m = r
                
                if best_gain_for_m > best_add_gain:
                    best_add_gain = best_gain_for_m
                    best_add = (m, best_r_for_m)
            
            if best_add is not None:
                m, r = best_add
                classes.append([m, r])
                idxs = class_indices[m][r]
                cover_count[idxs] += 1
                uncovered -= best_add_gain
                used_moduli.add(m)
                available_moduli.remove(m)
                improved = True
        
        # Try removing a modulus that doesn't hurt
        if time.time() < deadline - 5:
            for idx in range(len(classes)):
                if time.time() >= deadline - 5:
                    break
                m, r = classes[idx]
                idxs = class_indices[m][r]
                loss = int(np.count_nonzero(cover_count[idxs] == 1))
                if loss == 0:
                    # Safe to remove
                    cover_count[idxs] -= 1
                    classes.pop(idx)
                    used_moduli.discard(m)
                    if m not in available_moduli:
                        available_moduli.append(m)
                    improved = True
                    break
    
    # Final coverage recomputation for verification
    final_covered = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        final_covered[class_indices[m][r]] = 1
    
    return classes
