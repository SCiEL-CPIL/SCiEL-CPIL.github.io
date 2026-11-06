from fractions import Fraction
import time

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    recips = [Fraction(1, d) for d in denominators]
    
    start_time = time.time()
    max_time = 55.0
    
    def greedy_smallest_deficit():
        """Add elements in order of how close they bring us to 1 (smallest deficit after adding)"""
        current_set = set()
        current_total = Fraction(0, 1)
        remaining = list(range(n))
        
        while remaining:
            best_idx = -1
            best_def = None
            for i in remaining:
                new_total = current_total + recips[i]
                if new_total < 1:
                    new_def = 1 - new_total
                    if best_def is None or new_def < best_def:
                        best_def = new_def
                        best_idx = i
            if best_idx < 0:
                break
            current_set.add(best_idx)
            current_total += recips[best_idx]
            remaining.remove(best_idx)
        
        return current_set, current_total
    
    def greedy_largest_first():
        """Add elements in descending order of reciprocal value"""
        current_set = set()
        current_total = Fraction(0, 1)
        for i in range(n):
            if current_total + recips[i] < 1:
                current_set.add(i)
                current_total += recips[i]
        return current_set, current_total
    
    def greedy_smallest_first():
        """Add elements in ascending order of reciprocal value (smallest denominators last)"""
        current_set = set()
        current_total = Fraction(0, 1)
        for i in range(n-1, -1, -1):
            if current_total + recips[i] < 1:
                current_set.add(i)
                current_total += recips[i]
        return current_set, current_total
    
    def greedy_alternating():
        """Alternate between adding the best possible element and skipping the next best"""
        current_set = set()
        current_total = Fraction(0, 1)
        remaining = list(range(n))
        flip = False
        while remaining:
            best_idx = -1
            best_def = None
            for i in remaining:
                new_total = current_total + recips[i]
                if new_total < 1:
                    new_def = 1 - new_total
                    if best_def is None or new_def < best_def:
                        best_def = new_def
                        best_idx = i
            if best_idx < 0:
                break
            current_set.add(best_idx)
            current_total += recips[best_idx]
            remaining.remove(best_idx)
            if len(remaining) > 1 and not flip:
                best_idx2 = -1
                best_def2 = None
                for i in remaining:
                    new_total = current_total + recips[i]
                    if new_total < 1:
                        new_def = 1 - new_total
                        if best_def2 is None or new_def < best_def2:
                            best_def2 = new_def
                            best_idx2 = i
                if best_idx2 >= 0:
                    remaining.remove(best_idx2)
            flip = not flip
        return current_set, current_total
    
    def local_search_pairwise(start_set, max_iters=10000):
        """Iteratively improve by finding the best pairwise swap or single add/remove"""
        current_set = set(start_set)
        current_total = Fraction(0, 1)
        for i in current_set:
            current_total += recips[i]
        if current_total >= 1:
            return None, None
        current_deficit = 1 - current_total
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() - start_time < max_time:
            improved = False
            iters += 1
            
            # Try all single additions
            best_add = -1
            best_add_def = current_deficit
            for i in range(n):
                if i not in current_set:
                    new_total = current_total + recips[i]
                    if new_total < 1:
                        new_def = 1 - new_total
                        if new_def < best_add_def:
                            best_add_def = new_def
                            best_add = i
            if best_add >= 0:
                current_set.add(best_add)
                current_total += recips[best_add]
                current_deficit = best_add_def
                improved = True
                continue
            
            # Try all pairwise swaps
            best_swap_i = -1
            best_swap_j = -1
            best_swap_def = current_deficit
            in_list = list(current_set)
            for i in in_list:
                for j in range(n):
                    if j not in current_set:
                        new_total = current_total - recips[i] + recips[j]
                        if new_total < 1:
                            new_def = 1 - new_total
                            if new_def < best_swap_def:
                                best_swap_def = new_def
                                best_swap_i = i
                                best_swap_j = j
            if best_swap_i >= 0:
                current_set.remove(best_swap_i)
                current_set.add(best_swap_j)
                current_total = current_total - recips[best_swap_i] + recips[best_swap_j]
                current_deficit = best_swap_def
                improved = True
                continue
            
            # Try all single removals
            best_rem = -1
            best_rem_def = current_deficit
            for i in current_set:
                new_total = current_total - recips[i]
                if new_total < 1:
                    new_def = 1 - new_total
                    if new_def < best_rem_def:
                        best_rem_def = new_def
                        best_rem = i
            if best_rem >= 0:
                current_set.remove(best_rem)
                current_total -= recips[best_rem]
                current_deficit = best_rem_def
                improved = True
        
        return set(current_set), current_deficit
    
    # Generate multiple diverse starting solutions
    candidates = []
    
    s1, t1 = greedy_smallest_deficit()
    candidates.append(s1)
    
    s2, t2 = greedy_largest_first()
    candidates.append(s2)
    
    s3, t3 = greedy_smallest_first()
    candidates.append(s3)
    
    s4, t4 = greedy_alternating()
    candidates.append(s4)
    
    # Run local search on each candidate
    best_set = None
    best_deficit = Fraction(1, 1)
    
    for cand in candidates:
        if time.time() - start_time >= max_time:
            break
        refined_set, refined_def = local_search_pairwise(cand, max_iters=5000)
        if refined_set is not None and refined_def < best_deficit:
            best_deficit = refined_def
            best_set = refined_set
    
    # Final polish on the best solution with more iterations
    if time.time() - start_time < max_time and best_set is not None:
        final_set, final_def = local_search_pairwise(best_set, max_iters=20000)
        if final_set is not None and final_def < best_deficit:
            best_deficit = final_def
            best_set = final_set
    
    if best_set is None:
        best_set = set()
        total = Fraction(0, 1)
        for i in range(n):
            if total + recips[i] < 1:
                best_set.add(i)
                total += recips[i]
    
    return sorted(best_set)
