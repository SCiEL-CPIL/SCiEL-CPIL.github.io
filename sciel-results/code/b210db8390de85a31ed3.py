import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    def is_compatible(m_new, r_new, selected):
        """Check if (m_new, r_new) is disjoint from all in selected."""
        for m_i, r_i in selected:
            g = math.gcd(m_new, m_i)
            if r_new % g == r_i % g:
                return False
        return True
    
    def compute_score(selected):
        """Compute exact rational score as float."""
        total = Fraction(0, 1)
        for m, r in selected:
            total += Fraction(1, m)
        return float(total)
    
    def count_conflicts(m_new, r_new, selected):
        """Count how many residues of m_new are blocked by selected classes."""
        blocked = 0
        for r in range(m_new):
            for m_i, r_i in selected:
                g = math.gcd(m_new, m_i)
                if r % g == r_i % g:
                    blocked += 1
                    break
        return blocked
    
    def greedy_construct(order):
        """Greedy construction following given modulus order."""
        selected = []
        for m in order:
            best_r = None
            min_conflicts = m + 1  # worst case: all blocked
            for r in range(m):
                if not is_compatible(m, r, selected):
                    continue
                # Count future conflicts: how many residues of this m are blocked
                # Actually, we want to pick r that leaves most options open for others
                # Simpler heuristic: just pick first valid, or pick one that minimizes
                # the number of "tight" constraints
                if best_r is None:
                    best_r = r
                # Try to find a residue that's "more spread out" - pick r that
                # has the largest minimum distance to any existing r_i mod gcd
                # For simplicity, just take first valid
                break
            
            if best_r is not None:
                selected.append([m, best_r])
        
        return selected
    
    # Phase 1: Greedy with natural order (small to large m)
    natural_order = list(range(20, 201))
    selected = greedy_construct(natural_order)
    best_score = compute_score(selected)
    best_selected = [x[:] for x in selected]
    
    if time.time() - start_time > DEADLINE:
        return best_selected
    
    # Phase 2: Try different orderings
    # Sort by some heuristic: prefer moduli with more "freedom"
    # For each m, count how many residues are available given no other constraints
    # Actually, try: sort by number of divisors (fewer divisors = fewer conflicts)
    
    def divisor_count(m):
        count = 0
        for d in range(1, int(math.isqrt(m)) + 1):
            if m % d == 0:
                count += 1
                if d != m // d:
                    count += 1
        return count
    
    # Try ordering by descending divisor count (more composite = more conflicts, do later)
    # Or ascending: do highly composite ones first when they have most freedom
    order_by_divs = sorted(range(20, 201), key=lambda m: divisor_count(m))
    selected2 = greedy_construct(order_by_divs)
    score2 = compute_score(selected2)
    if score2 > best_score:
        best_score = score2
        best_selected = [x[:] for x in selected2]
    
    if time.time() - start_time > DEADLINE:
        return best_selected
    
    # Phase 3: Local search - try to improve by swapping residues or replacing moduli
    # For each selected [m, r], try changing r to a different valid residue
    # that allows adding a new small modulus
    
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        best_local = [x[:] for x in best_selected]
        best_local_score = best_score
        
        for i, (m_i, r_i) in enumerate(best_local):
            # Try removing this and adding a different modulus
            remaining = [x[:] for x in best_local if x[0] != m_i]
            
            # Try to add a modulus that wasn't in the set
            current_mods = set(x[0] for x in remaining)
            for m_try in range(20, 201):
                if m_try in current_mods:
                    continue
                # Check if we can add m_try
                for r_try in range(m_try):
                    if is_compatible(m_try, r_try, remaining):
                        new_set = remaining + [[m_try, r_try]]
                        new_score = compute_score(new_set)
                        if new_score > best_local_score:
                            best_local_score = new_score
                            best_local = [x[:] for x in new_set]
                            improved = True
                        break  # Only try first valid r for this m_try
                # Also try just keeping m_i but with different r
                # (this is less likely to help since m_i is already in)
            
            # Try changing residue of m_i
            for r_try in range(m_i):
                if r_try == r_i:
                    continue
                other = [x[:] for x in best_local if x[0] != m_i]
                if is_compatible(m_i, r_try, other):
                    # Score doesn't change with residue, so only useful if it enables additions
                    # Skip for now, focus on modulus swaps
                    pass
        
        if improved:
            best_selected = [x[:] for x in best_local]
            best_score = best_local_score
    
    if time.time() - start_time > DEADLINE:
        return best_selected
    
    # Phase 4: Try to add more moduli to the current set
    current_mods = set(x[0] for x in best_selected)
    for m_try in range(20, 201):
        if m_try in current_mods:
            continue
        for r_try in range(m_try):
            if is_compatible(m_try, r_try, best_selected):
                best_selected.append([m_try, r_try])
                best_score = compute_score(best_selected)
                current_mods.add(m_try)
                break
    
    return best_selected
