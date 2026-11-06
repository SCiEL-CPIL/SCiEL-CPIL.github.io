import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    base = 51975
    
    # Get all odd divisors > 1 of 51975
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # Precompute residue class indices for each divisor
    class_indices = {}
    for m in divisors:
        class_indices[m] = [list(range(r, base, m)) for r in range(m)]
    
    def count_uncovered(covered):
        return np.count_nonzero(covered == 0)
    
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
                cnt = np.count_nonzero(covered[idxs] == 0)
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
    for attempt in range(20):
        if time.time() >= deadline - 60:
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
    
    # Local search with efficient move evaluation
    classes = [c[:] for c in best_classes]
    covered = best_covered.copy()
    uncovered = best_uncovered
    
    used_moduli = set(m for m, r in classes)
    available_moduli = [m for m in divisors if m not in used_moduli]
    
    def compute_cover_count(covered, classes):
        """Compute how many classes cover each residue"""
        cover_count = np.zeros(base, dtype=np.int32)
        for m, r in classes:
            cover_count[class_indices[m][r]] += 1
        return cover_count
    
    cover_count = compute_cover_count(covered, classes)
    
    def try_move_change_residue(idx, new_r):
        """Evaluate changing residue of class idx to new_r"""
        m, old_r = classes[idx]
        old_idxs = class_indices[m][old_r]
        new_idxs = class_indices[m][new_r]
        
        # Residues that will lose coverage: in old but not in new, and cover_count == 1
        # Residues that will gain coverage: in new but not in old, and cover_count == 0
        
        # Efficient: compute delta
        delta = 0
        # Old residues that become uncovered (cover_count drops to 0)
        for i in old_idxs:
            if i not in new_idxs and cover_count[i] == 1:
                delta += 1  # becomes uncovered
        
        # New residues that were uncovered (cover_count was 0)
        for i in new_idxs:
            if i not in old_idxs and cover_count[i] == 0:
                delta -= 1  # becomes covered
        
        return delta
    
    # Better: use set operations for efficiency
    def try_move_change_residue_v2(idx, new_r):
        m, old_r = classes[idx]
        old_set = set(class_indices[m][old_r])
        new_set = set(class_indices[m][new_r])
        
        # Only residues in symmetric difference matter
        diff = old_set.symmetric_difference(new_set)
        
        delta = 0
        for i in diff:
            if i in old_set and not i in new_set:
                # Losing coverage
                if cover_count[i] == 1:
                    delta += 1
            else:
                # Gaining coverage
                if cover_count[i] == 0:
                    delta -= 1
        
        return delta
    
    improved = True
    max_rounds = 50
    round_num = 0
    
    while improved and time.time() < deadline - 10 and round_num < max_rounds:
        improved = False
        round_num += 1
        
        # Try changing residue for each existing class
        for idx in range(len(classes)):
            if time.time() >= deadline - 10:
                break
            m, r = classes[idx]
            best_delta = 0
            best_new_r = None
            
            for new_r in range(m):
                if new_r == r:
                    continue
                delta = try_move_change_residue_v2(idx, new_r)
                if delta < best_delta:
                    best_delta = delta
                    best_new_r = new_r
            
            if best_new_r is not None:
                # Apply the move
                m, old_r = classes[idx]
                old_idxs = class_indices[m][old_r]
                new_idxs = class_indices[m][best_new_r]
                
                # Update cover_count
                for i in old_idxs:
                    cover_count[i] -= 1
                for i in new_idxs:
                    cover_count[i] += 1
                
                classes[idx][1] = best_new_r
                uncovered += best_delta
                improved = True
        
        # Try adding a new modulus
        if time.time() < deadline - 10:
            best_add_delta = 0
            best_add = None
            
            for m in available_moduli:
                if time.time() >= deadline - 10:
                    break
                best_r_for_m = None
                best_delta_for_m = 0
                
                for r in range(m):
                    idxs = class_indices[m][r]
                    # Count how many uncovered residues this would cover
                    gain = np.count_nonzero(cover_count[idxs] == 0)
                    if gain > best_delta_for_m:
                        best_delta_for_m = gain
                        best_r_for_m = r
                
                if best_delta_for_m > best_add_delta:
                    best_add_delta = best_delta_for_m
                    best_add = (m, best_r_for_m)
            
            if best_add is not None:
                m, r = best_add
                classes.append([m, r])
                idxs = class_indices[m][r]
                cover_count[idxs] += 1
                uncovered -= best_add_delta
                used_moduli.add(m)
                available_moduli.remove(m)
                improved = True
        
        # Try removing a modulus
        if time.time() < deadline - 10:
            best_remove_delta = 0
            best_remove_idx = None
            
            for idx in range(len(classes)):
                if time.time() >= deadline - 10:
                    break
                m, r = classes[idx]
                idxs = class_indices[m][r]
                # Count how many residues would become uncovered
                loss = np.count_nonzero(cover_count[idxs] == 1)
                if loss < best_remove_delta:
                    best_remove_delta = loss
                    best_remove_idx = idx
            
            if best_remove_idx is not None and best_remove_delta < 0:
                # Removing improves (negative loss means we gain)
                # Actually, loss should be >= 0. We remove if loss == 0
                pass
        
        # Try removing if it doesn't increase uncovered
        if time.time() < deadline - 10:
            for idx in range(len(classes)):
                if time.time() >= deadline - 10:
                    break
                m, r = classes[idx]
                idxs = class_indices[m][r]
                loss = np.count_nonzero(cover_count[idxs] == 1)
                if loss == 0:
                    # Safe to remove
                    cover_count[idxs] -= 1
                    classes.pop(idx)
                    used_moduli.discard(m)
                    if m not in available_moduli:
                        available_moduli.append(m)
                    improved = True
                    break
    
    # Final coverage recomputation
    final_covered = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        final_covered[class_indices[m][r]] = 1
    final_uncovered = count_uncovered(final_covered)
    
    return classes
