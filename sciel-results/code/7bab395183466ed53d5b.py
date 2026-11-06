import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    base = 51975
    
    # Find all divisors of base
    divisors = []
    for i in range(1, int(base**0.5) + 1):
        if base % i == 0:
            divisors.append(i)
            if i != base // i:
                divisors.append(base // i)
    
    # Filter: m > 1, odd
    divisors = [m for m in divisors if m > 1 and m % 2 == 1]
    divisors.sort()
    
    # Precompute indices
    indices_by_m = {}
    for m in divisors:
        res_indices = []
        for r in range(m):
            idx = np.arange(r, base, m)
            res_indices.append(idx)
        indices_by_m[m] = res_indices

    best_uncovered = base
    best_classes = []
    
    def local_search(classes, covered, max_iter=300):
        """Try swap operations to improve coverage."""
        if time.time() > deadline - 5:
            return classes, covered
        
        classes = [c[:] for c in classes]
        covered = covered.copy()
        current_uncovered = np.sum(covered == 0)
        improved = True
        
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            
            if time.time() > deadline - 5:
                break
            
            used_moduli = set(m for m, r in classes)
            
            # Try residue swaps for existing classes
            for i, (m_i, r_i) in enumerate(classes):
                if time.time() > deadline - 5:
                    break
                
                old_idx = indices_by_m[m_i][r_i]
                
                # Try all residues for this modulus
                best_r = r_i
                best_uncovered = current_uncovered
                temp_covered = covered.copy()
                temp_covered[old_idx] = 0  # Remove old class
                
                for r_new in range(m_i):
                    if r_new == r_i:
                        continue
                    new_idx = indices_by_m[m_i][r_new]
                    temp_covered[new_idx] = 1
                    unc = np.sum(temp_covered == 0)
                    if unc < best_uncovered:
                        best_uncovered = unc
                        best_r = r_new
                    temp_covered[new_idx] = 0  # Reset for next iteration
                
                if best_r != r_i:
                    classes[i] = [m_i, best_r]
                    covered[old_idx] = 0
                    covered[indices_by_m[m_i][best_r]] = 1
                    current_uncovered = best_uncovered
                    improved = True
                    break
            
            # Try adding a new class
            if not improved:
                for m in divisors:
                    if m in used_moduli:
                        continue
                    
                    best_r = None
                    best_gain = -1
                    for r in range(m):
                        idx = indices_by_m[m][r]
                        gain = np.sum(covered[idx] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_r = r
                    
                    if best_gain > 0:
                        classes.append([m, best_r])
                        covered[indices_by_m[m][best_r]] = 1
                        current_uncovered = np.sum(covered == 0)
                        improved = True
                        break
            
            # Try removing a class that doesn't help
            if not improved:
                for i in range(len(classes)):
                    m_i, r_i = classes[i]
                    idx = indices_by_m[m_i][r_i]
                    
                    temp_covered = covered.copy()
                    temp_covered[idx] = 0
                    uncovered_after_remove = np.sum(temp_covered == 0)
                    
                    if uncovered_after_remove == current_uncovered:
                        del classes[i]
                        covered = temp_covered
                        improved = True
                        break
        
        return classes, covered
    
    def bipartite_assignment(covered, available_moduli):
        """Use Hungarian algorithm to find good residue assignments for available moduli."""
        if not available_moduli:
            return [], covered
        
        # For each available modulus, compute the best residue based on uncovered coverage
        # This is a simplified version - we'll use it as a heuristic initialization
        assignments = []
        temp_covered = covered.copy()
        
        # Sort by modulus size (larger first for more impact)
        sorted_moduli = sorted(available_moduli, reverse=True)
        
        for m in sorted_moduli:
            if time.time() > deadline - 8:
                break
            
            best_r = 0
            best_gain = -1
            for r in range(m):
                idx = indices_by_m[m][r]
                gain = np.sum(temp_covered[idx] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_gain > 0:
                assignments.append([m, best_r])
                temp_covered[indices_by_m[m][best_r]] = 1
        
        return assignments, temp_covered

    # Run multiple greedy strategies
    num_runs = 8
    
    for run in range(num_runs):
        if time.time() > deadline - 10:
            break
            
        covered = np.zeros(base, dtype=np.uint8)
        available_moduli = set(divisors)
        classes = []
        
        # Choose ordering: descending, ascending, or random
        if run == 0:
            ordered_moduli = sorted(divisors, reverse=True)
        elif run == 1:
            ordered_moduli = sorted(divisors)
        else:
            ordered_moduli = list(divisors)
            random.shuffle(ordered_moduli)
        
        for m in ordered_moduli:
            if time.time() > deadline - 8:
                break
            if m not in available_moduli:
                continue
                
            best_r = 0
            best_gain = -1
            for r in range(m):
                idx = indices_by_m[m][r]
                gain = np.sum(covered[idx] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_gain <= 0:
                continue
                
            classes.append([m, best_r])
            covered[indices_by_m[m][best_r]] = 1
            available_moduli.remove(m)
            
            current_uncovered = np.sum(covered == 0)
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_classes = [c[:] for c in classes]
                
            if best_uncovered == 0:
                return best_classes
        
        # Local search refinement
        if classes and time.time() < deadline - 5:
            refined_classes, refined_covered = local_search(classes, covered)
            refined_uncovered = np.sum(refined_covered == 0)
            if refined_uncovered < best_uncovered:
                best_uncovered = refined_uncovered
                best_classes = [c[:] for c in refined_classes]
                if best_uncovered == 0:
                    return best_classes

    return best_classes
