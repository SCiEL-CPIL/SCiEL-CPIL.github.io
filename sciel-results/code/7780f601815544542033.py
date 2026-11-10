import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    moduli_range = list(range(2, 129))
    n_moduli = len(moduli_range)
    
    # Precompute gcd matrix
    gcd_table = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(n_moduli):
            gcd_table[i][j] = gcd(moduli_range[i], moduli_range[j])
    
    def is_compatible(residues, moduli_idx_list, new_r, new_m_idx):
        """Check if new residue r with modulus at new_m_idx is compatible with all existing."""
        for i in moduli_idx_list:
            g = gcd_table[new_m_idx][i]
            if new_r % g == residues[i] % g:
                return False
        return True
    
    def greedy_search(order, max_time=None):
        """Greedy: for each modulus in order, pick first valid residue."""
        if max_time is None:
            max_time = deadline
        residues = [0] * n_moduli
        selected = []
        selected_idx = []
        
        for m_idx in order:
            m = moduli_range[m_idx]
            found = False
            for r in range(m):
                if is_compatible(residues, selected_idx, r, m_idx):
                    residues[m_idx] = r
                    selected.append([r, m])
                    selected_idx.append(m_idx)
                    found = True
                    break
            if time.time() > max_time:
                break
        return selected
    
    def backtracking_search(order, current_set, current_idx, residues, selected_idx, best_ref):
        """Limited backtracking: try to extend current_set by picking residues for order[current_idx:]."""
        if time.time() > deadline:
            return
        if len(current_set) > len(best_ref[0]):
            best_ref[0] = list(current_set)
        
        if current_idx >= len(order):
            return
        
        m_idx = order[current_idx]
        m = moduli_range[m_idx]
        
        # Try each residue
        for r in range(m):
            if is_compatible(residues, selected_idx, r, m_idx):
                residues[m_idx] = r
                current_set.append([r, m])
                selected_idx.append(m_idx)
                backtracking_search(order, current_set, current_idx + 1, residues, selected_idx, best_ref)
                selected_idx.pop()
                current_set.pop()
        
        # Also try skipping this modulus
        backtracking_search(order, current_set, current_idx + 1, residues, selected_idx, best_ref)
    
    # Compute conflict count for ordering
    def conflict_count(m_idx):
        m = moduli_range[m_idx]
        count = 0
        for j in range(n_moduli):
            if j != m_idx and gcd_table[m_idx][j] > 1:
                count += 1
        return count
    
    conflict_counts = [conflict_count(i) for i in range(n_moduli)]
    
    # Strategy 1: Greedy with different orderings
    best = []
    
    # Ordering by conflict count descending (hardest first)
    order1 = sorted(range(n_moduli), key=lambda i: conflict_counts[i], reverse=True)
    r1 = greedy_search(order1)
    if len(r1) > len(best):
        best = r1
    
    # Ordering by conflict count ascending
    order2 = sorted(range(n_moduli), key=lambda i: conflict_counts[i])
    r2 = greedy_search(order2)
    if len(r2) > len(best):
        best = r2
    
    # Natural order
    r3 = greedy_search(list(range(n_moduli)))
    if len(r3) > len(best):
        best = r3
    
    # Reverse natural order
    r4 = greedy_search(list(reversed(range(n_moduli))))
    if len(r4) > len(best):
        best = r4
    
    # Strategy 2: Randomized greedy with multiple restarts
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        # Random permutation with bias
        order = list(range(n_moduli))
        random.shuffle(order)
        result = greedy_search(order)
        if len(result) > len(best):
            best = result
    
    # Strategy 3: Local search - try to add missing moduli
    def local_improve(result):
        used_moduli = set(m for _, m in result)
        residues_arr = [0] * n_moduli
        selected_idx = []
        for [r, m] in result:
            m_idx = moduli_range.index(m)
            residues_arr[m_idx] = r
            selected_idx.append(m_idx)
        
        improved = list(result)
        
        # Try adding each unused modulus
        for m_idx in range(n_moduli):
            if moduli_range[m_idx] in used_moduli:
                continue
            if time.time() > deadline - 5:
                break
            m = moduli_range[m_idx]
            for r in range(m):
                if is_compatible(residues_arr, selected_idx, r, m_idx):
                    residues_arr[m_idx] = r
                    improved.append([r, m])
                    selected_idx.append(m_idx)
                    used_moduli.add(m)
                    break
        
        return improved
    
    # Multiple passes of local improvement
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        new_best = local_improve(best)
        if len(new_best) > len(best):
            best = new_best
        else:
            break
    
    # Strategy 4: Limited backtracking on a subset
    # Use the greedy result as starting point, try to add more with backtracking
    if time.time() < deadline - 15:
        used_moduli = set(m for _, m in best)
        remaining = [i for i in range(n_moduli) if moduli_range[i] not in used_moduli]
        
        if remaining:
            # Sort remaining by conflict count
            remaining_sorted = sorted(remaining, key=lambda i: conflict_counts[i], reverse=True)
            
            residues_arr = [0] * n_moduli
            selected_idx = []
            for [r, m] in best:
                m_idx = moduli_range.index(m)
                residues_arr[m_idx] = r
                selected_idx.append(m_idx)
            
            best_ref = [list(best)]
            current_set = list(best)
            backtracking_search(remaining_sorted, current_set, 0, residues_arr, selected_idx, best_ref)
            if len(best_ref[0]) > len(best):
                best = best_ref[0]
    
    if not best:
        best = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    return best
