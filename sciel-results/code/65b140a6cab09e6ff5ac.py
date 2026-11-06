import time
import random

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin
    
    # Generate divisors > 1
    divisors = [d for d in range(3, N + 1, 2) if N % d == 0]
    
    # Precompute masks for each (m, r)
    # masks[m][r] is an integer bitset where bit i is 1 if i % m == r
    masks = {}
    for m in divisors:
        masks[m] = [0] * m
        for r in range(m):
            mask = 0
            for x in range(r, N, m):
                mask |= (1 << x)
            masks[m][r] = mask
            
    full_mask = (1 << N) - 1
    
    def get_uncovered_count(covered_mask):
        return N - covered_mask.bit_count()
        
    def find_best_residue(m, current_covered):
        """Find r that maximizes gain for modulus m"""
        best_r = 0
        best_gain = -1
        inv_covered = ~current_covered & full_mask
        for r in range(m):
            gain = (masks[m][r] & inv_covered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_r = r
        return best_r, best_gain

    def greedy_init():
        """Greedy initialization"""
        selected = {}
        available = set(divisors)
        covered_mask = 0
        
        while available:
            if time.time() > deadline:
                break
                
            best_m = None
            best_r = None
            best_gain = 0
            
            for m in available:
                r, gain = find_best_residue(m, covered_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
                    
            if best_m is None or best_gain == 0:
                break
                
            selected[best_m] = best_r
            covered_mask |= masks[best_m][best_r]
            available.remove(best_m)
            
            if get_uncovered_count(covered_mask) == 0:
                break
                
        return selected, covered_mask, available

    def local_search(selected, covered_mask, available):
        """Local search to improve the solution"""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_uncovered = get_uncovered_count(covered_mask)
            
            if current_uncovered == 0:
                return selected, covered_mask, available, True
                
            # a) Try changing residue for each selected modulus
            for m in list(selected.keys()):
                if time.time() > deadline:
                    break
                old_r = selected[m]
                old_mask = masks[m][old_r]
                
                # Compute covered_by_others once
                covered_by_others = 0
                for mm, rr in selected.items():
                    if mm != m:
                        covered_by_others |= masks[mm][rr]
                
                best_new_r = old_r
                best_new_uncovered = current_uncovered
                
                for new_r in range(m):
                    if new_r == old_r:
                        continue
                    new_mask = masks[m][new_r]
                    new_covered = covered_by_others | new_mask
                    new_uncovered = N - new_covered.bit_count()
                    
                    if new_uncovered < best_new_uncovered:
                        best_new_uncovered = new_uncovered
                        best_new_r = new_r
                
                if best_new_r != old_r:
                    selected[m] = best_new_r
                    # Recompute covered_mask
                    covered_mask = 0
                    for mm, rr in selected.items():
                        covered_mask |= masks[mm][rr]
                    improved = True
                    current_uncovered = best_new_uncovered
                    if current_uncovered == 0:
                        return selected, covered_mask, available, True

            # b) Try swapping selected m_out with unselected m_in
            selected_list = list(selected.keys())
            for m_out in selected_list:
                if time.time() > deadline:
                    break
                
                old_r_out = selected[m_out]
                
                # Compute covered mask without m_out
                covered_without_out = 0
                for mm, rr in selected.items():
                    if mm != m_out:
                        covered_without_out |= masks[mm][rr]
                
                best_m_in = None
                best_r_in = None
                best_new_uncovered = current_uncovered
                
                for m_in in available:
                    if time.time() > deadline:
                        break
                    
                    r_in, gain_in = find_best_residue(m_in, covered_without_out)
                    
                    if gain_in > 0:
                        new_covered = covered_without_out | masks[m_in][r_in]
                        new_uncovered = N - new_covered.bit_count()
                        
                        if new_uncovered < best_new_uncovered:
                            best_new_uncovered = new_uncovered
                            best_m_in = m_in
                            best_r_in = r_in
                
                if best_m_in is not None:
                    # Perform swap
                    del selected[m_out]
                    available.remove(best_m_in)
                    available.add(m_out)
                    selected[best_m_in] = best_r_in
                    
                    # Recompute covered_mask
                    covered_mask = 0
                    for mm, rr in selected.items():
                        covered_mask |= masks[mm][rr]
                    
                    improved = True
                    current_uncovered = best_new_uncovered
                    if current_uncovered == 0:
                        return selected, covered_mask, available, True
                    break  # Restart swap loop
            
            # c) Try adding a new modulus (if we haven't used all)
            if len(selected) < len(divisors):
                best_m_add = None
                best_r_add = None
                best_gain_add = 0
                
                for m_add in available:
                    if time.time() > deadline:
                        break
                    r_add, gain_add = find_best_residue(m_add, covered_mask)
                    if gain_add > best_gain_add:
                        best_gain_add = gain_add
                        best_m_add = m_add
                        best_r_add = r_add
                
                if best_m_add is not None and best_gain_add > 0:
                    selected[best_m_add] = best_r_add
                    covered_mask |= masks[best_m_add][best_r_add]
                    available.remove(best_m_add)
                    improved = True
                    if get_uncovered_count(covered_mask) == 0:
                        return selected, covered_mask, available, True

        return selected, covered_mask, available, False

    # Try multiple restarts with different random seeds/orderings
    best_selected = None
    best_uncovered = N
    best_covered = 0
    best_available = None
    found_zero = False
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 30:  # Leave time for final polish
            break
            
        if restart == 0:
            # First restart: deterministic greedy
            selected, covered_mask, available = greedy_init()
        else:
            # Subsequent restarts: randomize the order of moduli in greedy
            random.seed(restart * 12345 + 67890)
            selected = {}
            available = set(divisors)
            covered_mask = 0
            
            # Randomize by shuffling divisors and trying in random order
            shuffled_divs = list(divisors)
            random.shuffle(shuffled_divs)
            
            for m in shuffled_divs:
                if time.time() > deadline - 30:
                    break
                if m not in available:
                    continue
                r, gain = find_best_residue(m, covered_mask)
                if gain > 0:
                    selected[m] = r
                    covered_mask |= masks[m][r]
                    available.remove(m)
                if get_uncovered_count(covered_mask) == 0:
                    break
        
        # Local search
        selected, covered_mask, available, found_zero = local_search(selected, covered_mask, available)
        
        current_uncovered = get_uncovered_count(covered_mask)
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_selected = dict(selected)
            best_covered = covered_mask
            best_available = set(available)
            found_zero = found_zero
            
        if found_zero or best_uncovered == 0:
            break
    
    return [[m, r] for m, r in best_selected.items()]
