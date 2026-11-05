import math
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    def is_compatible(m_new, r_new, selected):
        """Check if (m_new, r_new) is compatible with all selected pairs."""
        for m_sel, r_sel in selected:
            g = math.gcd(m_new, m_sel)
            if r_new % g == r_sel % g:
                return False
        return True
    
    def find_valid_residue(m, selected):
        """Find a valid residue for modulus m given selected pairs."""
        for r in range(m):
            if is_compatible(m, r, selected):
                return r
        return None
    
    def score(selected):
        return sum(1.0 / m for m, r in selected)
    
    # Greedy: process moduli in increasing order
    selected = []
    for m in range(20, 201):
        if time_left() < 5:
            break
        r = find_valid_residue(m, selected)
        if r is not None:
            selected.append((m, r))
    
    best_score = score(selected)
    best_selected = list(selected)
    
    # Try local search: attempt to add missed moduli or swap
    # First, try to add any moduli we skipped
    if time_left() > 10:
        improved = True
        while improved and time_left() > 10:
            improved = False
            for m in range(20, 201):
                if (m, None) not in [(sm, sr) for sm, sr in best_selected] and not any(sm == m for sm, sr in best_selected):
                    r = find_valid_residue(m, best_selected)
                    if r is not None:
                        new_selected = best_selected + [(m, r)]
                        new_score = score(new_selected)
                        if new_score > best_score:
                            best_selected = new_selected
                            best_score = new_score
                            improved = True
    
    # Try removing one and adding two (swap improvement)
    if time_left() > 15:
        for i in range(len(best_selected)):
            if time_left() < 10:
                break
            # Try removing best_selected[i] and see if we can add more
            remaining = best_selected[:i] + best_selected[i+1:]
            # Try to add back some moduli
            for m in range(20, 201):
                if any(sm == m for sm, sr in remaining):
                    continue
                r = find_valid_residue(m, remaining)
                if r is not None:
                    candidate = remaining + [(m, r)]
                    cand_score = score(candidate)
                    if cand_score > best_score:
                        best_selected = candidate
                        best_score = cand_score
    
    # Convert to required format [[m, r], ...]
    result = [[m, r] for m, r in best_selected]
    
    # Verify validity
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            m1, r1 = result[i]
            m2, r2 = result[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                # Invalid, fall back to greedy
                result = [[m, r] for m, r in selected]
                break
    
    return result
