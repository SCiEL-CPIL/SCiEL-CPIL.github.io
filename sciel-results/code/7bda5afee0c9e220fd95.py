import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def check_conflict(res, m, selected):
        """Check if residue res with modulus m conflicts with any selected class."""
        for m2, r2 in selected:
            g = gcd(m, m2)
            if res % g == r2 % g:
                return True
        return False

    def find_valid_residue(m, selected):
        """Find any valid residue for m given selected set. Returns None if none."""
        for r in range(m):
            if not check_conflict(r, m, selected):
                return r
        return None

    def calculate_score(selected):
        total = Fraction(0, 1)
        for m, r in selected:
            total += Fraction(1, m)
        return total

    # 1. Initial Greedy Construction
    selected = []
    selected_moduli = set()
    
    # Greedy pass: Try to add smallest moduli first
    for m in range(20, 201):
        if time.time() > deadline:
            break
        r = find_valid_residue(m, selected)
        if r is not None:
            selected.append([m, r])
            selected_moduli.add(m)
            
    current_score = calculate_score(selected)
    all_moduli = set(range(20, 201))
    missing_moduli = all_moduli - selected_moduli
    
    # 2. Local Search
    no_improvement_count = 0
    max_no_improvement = 10000
    
    while time.time() < deadline:
        if no_improvement_count > max_no_improvement:
            break
            
        improvement_found = False
        
        # Strategy A: Try to add any missing modulus
        # Prioritize smaller moduli (larger reciprocal)
        sorted_missing = sorted(missing_moduli)
        for m in sorted_missing:
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append([m, r])
                selected_moduli.add(m)
                missing_moduli.remove(m)
                current_score += Fraction(1, m)
                improvement_found = True
                no_improvement_count = 0
                break # Restart loop to try more adds
        
        if improvement_found:
            continue
            
        # Strategy B: Try to swap
        # Replace a larger modulus with a smaller one to increase score.
        sorted_selected_desc = sorted(selected, key=lambda x: x[0], reverse=True)
        sorted_missing_asc = sorted(missing_moduli)
        
        swap_done = False
        for m_old, r_old in sorted_selected_desc:
            # Only consider swaps where m_new < m_old
            for m_new in sorted_missing_asc:
                if m_new >= m_old:
                    continue # 1/m_new <= 1/m_old, no score gain
                
                # Temporarily remove m_old
                idx_old = -1
                for i, (m, r) in enumerate(selected):
                    if m == m_old:
                        idx_old = i
                        break
                if idx_old == -1: continue
                
                selected.pop(idx_old)
                selected_moduli.remove(m_old)
                
                r_new = find_valid_residue(m_new, selected)
                if r_new is not None:
                    # Accept swap
                    selected.append([m_new, r_new])
                    selected_moduli.add(m_new)
                    missing_moduli.remove(m_new)
                    missing_moduli.add(m_old)
                    current_score += Fraction(1, m_new) - Fraction(1, m_old)
                    improvement_found = True
                    swap_done = True
                    no_improvement_count = 0
                    break
                else:
                    # Revert
                    selected.append([m_old, r_old])
                    selected_moduli.add(m_old)
                    
            if swap_done:
                break # Restart loop
        
        if swap_done:
            continue

        # Strategy C: Residue Perturbation to enable Add
        if selected:
            idx = random.randint(0, len(selected) - 1)
            m_cur, r_cur = selected[idx]
            
            for _ in range(10):
                new_r = random.randint(0, m_cur - 1)
                if new_r == r_cur:
                    continue
                
                if check_conflict(new_r, m_cur, selected):
                    continue
                
                old_r = selected[idx][1]
                selected[idx][1] = new_r
                
                add_success = False
                sorted_missing = sorted(missing_moduli)
                for m_add in sorted_missing:
                    r_add = find_valid_residue(m_add, selected)
                    if r_add is not None:
                        selected.append([m_add, r_add])
                        selected_moduli.add(m_add)
                        missing_moduli.remove(m_add)
                        current_score += Fraction(1, m_add)
                        improvement_found = True
                        no_improvement_count = 0
                        add_success = True
                        break
                    
                if not add_success:
                    selected[idx][1] = old_r
                    
                if add_success:
                    break
        
        if not improvement_found:
            no_improvement_count += 1
        else:
            no_improvement_count = 0

    return selected
