import time
import random
import numpy as np

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Generate all divisors > 1 of N
    N_val = 51975
    divisors = []
    for d in range(3, N_val + 1, 2):
        if N_val % d == 0:
            divisors.append(d)
    
    # Sort divisors - smaller moduli first for better greedy performance
    divisors.sort()
    
    # Precompute coverage vectors using numpy
    # For each modulus m, precompute which residues are covered by each residue class r
    # cov[m][r] is a boolean array of length N
    cov = {}
    for m in divisors:
        cov[m] = np.zeros((m, N), dtype=np.int8)
        for r in range(m):
            # Residues x where x % m == r
            cov[m][r] = (np.arange(N) % m == r).astype(np.int8)
    
    # Precompute column sums for each modulus (useful for gain computation)
    # Actually, let's use a different approach: maintain a coverage count array
    
    def compute_covered(selected):
        """Compute the covered boolean array from selected moduli"""
        covered = np.zeros(N, dtype=np.int8)
        for m, r in selected.items():
            covered |= cov[m][r]
        return covered
    
    def uncovered_count(covered):
        return N - covered.sum()
    
    def greedy_init(div_order):
        """Greedy initialization with given divisor order"""
        selected = {}
        covered = np.zeros(N, dtype=np.int8)
        available = set(divisors)
        
        for m in div_order:
            if time.time() > deadline:
                break
            if m not in available:
                continue
            
            # Find best residue for m
            best_r = 0
            best_gain = 0
            uncovered = 1 - covered
            
            for r in range(m):
                gain = int((cov[m][r] * uncovered).sum())
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_gain > 0:
                selected[m] = best_r
                covered |= cov[m][best_r]
                available.remove(m)
            
            if covered.sum() == N:
                break
        
        return selected, covered, available
    
    def local_search(selected, covered, available):
        """Local search to improve the solution"""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_uncovered = uncovered_count(covered)
            
            if current_uncovered == 0:
                return selected, covered, available, True
            
            # a) Try changing residue for each selected modulus
            for m in list(selected.keys()):
                if time.time() > deadline:
                    break
                old_r = selected[m]
                
                # Compute covered by others
                covered_by_others = np.zeros(N, dtype=np.int8)
                for mm, rr in selected.items():
                    if mm != m:
                        covered_by_others |= cov[mm][rr]
                
                best_new_r = old_r
                best_new_uncovered = current_uncovered
                
                for new_r in range(m):
                    if new_r == old_r:
                        continue
                    new_covered = covered_by_others | cov[m][new_r]
                    new_uncovered = N - int(new_covered.sum())
                    
                    if new_uncovered < best_new_uncovered:
                        best_new_uncovered = new_uncovered
                        best_new_r = new_r
                
                if best_new_r != old_r:
                    selected[m] = best_new_r
                    covered = np.zeros(N, dtype=np.int8)
                    for mm, rr in selected.items():
                        covered |= cov[mm][rr]
                    improved = True
                    current_uncovered = best_new_uncovered
                    if current_uncovered == 0:
                        return selected, covered, available, True

            # b) Try swapping selected m_out with unselected m_in
            selected_list = list(selected.keys())
            for m_out in selected_list:
                if time.time() > deadline:
                    break
                
                # Compute covered mask without m_out
                covered_without_out = np.zeros(N, dtype=np.int8)
                for mm, rr in selected.items():
                    if mm != m_out:
                        covered_without_out |= cov[mm][rr]
                
                best_m_in = None
                best_r_in = None
                best_new_uncovered = current_uncovered
                
                for m_in in available:
                    if time.time() > deadline:
                        break
                    
                    # Find best residue for m_in given covered_without_out
                    uncovered = 1 - covered_without_out
                    best_r_in_candidate = 0
                    best_gain_in = 0
                    for r_in in range(m_in):
                        gain = int((cov[m_in][r_in] * uncovered).sum())
                        if gain > best_gain_in:
                            best_gain_in = gain
                            best_r_in_candidate = r_in
                    
                    if best_gain_in > 0:
                        new_covered = covered_without_out | cov[m_in][best_r_in_candidate]
                        new_uncovered = N - int(new_covered.sum())
                        
                        if new_uncovered < best_new_uncovered:
                            best_new_uncovered = new_uncovered
                            best_m_in = m_in
                            best_r_in = best_r_in_candidate
                
                if best_m_in is not None:
                    # Perform swap
                    del selected[m_out]
                    available.remove(best_m_in)
                    available.add(m_out)
                    selected[best_m_in] = best_r_in
                    
                    covered = np.zeros(N, dtype=np.int8)
                    for mm, rr in selected.items():
                        covered |= cov[mm][rr]
                    
                    improved = True
                    current_uncovered = best_new_uncovered
                    if current_uncovered == 0:
                        return selected, covered, available, True
                    break  # Restart swap loop
            
            # c) Try adding a new modulus (if we haven't used all)
            if len(selected) < len(divisors):
                best_m_add = None
                best_r_add = None
                best_gain_add = 0
                
                uncovered = 1 - covered
                for m_add in available:
                    if time.time() > deadline:
                        break
                    for r_add in range(m_add):
                        gain = int((cov[m_add][r_add] * uncovered).sum())
                        if gain > best_gain_add:
                            best_gain_add = gain
                            best_m_add = m_add
                            best_r_add = r_add
                
                if best_m_add is not None and best_gain_add > 0:
                    selected[best_m_add] = best_r_add
                    covered |= cov[best_m_add][best_r_add]
                    available.remove(best_m_add)
                    improved = True
                    if covered.sum() == N:
                        return selected, covered, available, True

        return selected, covered, available, False

    # Try multiple restarts with different random seeds/orderings
    best_selected = None
    best_uncovered = N
    best_covered = None
    best_available = None
    found_zero = False
    
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 20:  # Leave time for final polish
            break
            
        if restart == 0:
            # First restart: deterministic greedy with sorted divisors
            selected, covered, available = greedy_init(divisors)
        else:
            # Subsequent restarts: randomize the order of moduli in greedy
            random.seed(restart * 12345 + 67890)
            shuffled_divs = list(divisors)
            random.shuffle(shuffled_divs)
            selected, covered, available = greedy_init(shuffled_divs)
        
        # Local search
        selected, covered, available, found_zero = local_search(selected, covered, available)
        
        current_uncovered = uncovered_count(covered)
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_selected = dict(selected)
            best_covered = covered.copy()
            best_available = set(available)
            found_zero = found_zero
            
        if found_zero or best_uncovered == 0:
            break
    
    # Final polish: try to reduce uncovered further
    if best_selected is not None and best_uncovered > 0 and time.time() < deadline - 5:
        selected = dict(best_selected)
        available = set(best_available)
        covered = best_covered.copy()
        
        # Try all possible single-residue changes
        for m in list(selected.keys()):
            if time.time() > deadline - 5:
                break
            old_r = selected[m]
            covered_by_others = np.zeros(N, dtype=np.int8)
            for mm, rr in selected.items():
                if mm != m:
                    covered_by_others |= cov[mm][rr]
            
            for new_r in range(m):
                if new_r == old_r:
                    continue
                new_covered = covered_by_others | cov[m][new_r]
                new_uncovered = N - int(new_covered.sum())
                if new_uncovered < best_uncovered:
                    best_uncovered = new_uncovered
                    best_selected = dict(selected)
                    best_selected[m] = new_r
                    best_covered = new_covered.copy()
    
    return [[m, r] for m, r in best_selected.items()]
