import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Candidate moduli: All even numbers in [20, 200]
    # We can also include other numbers, but evens are a dense, compatible set.
    # Let's start with evens.
    moduli = list(range(20, 201, 2))
    
    # We want to maximize sum(1/m), so we process smaller moduli first.
    # They have higher weight and are "harder" to fit? 
    # Actually, smaller moduli have smaller residue domains, so they are more constrained.
    # Processing them first is a good heuristic.
    
    selected = []
    
    # Greedy assignment
    for m in moduli:
        if time.time() > deadline:
            break
            
        # Try to find a valid residue r for m
        found_r = None
        # We can try residues in a specific order.
        # 0 is often a good start.
        for r in range(m):
            valid = True
            for m_sel, r_sel in selected:
                g = gcd(m, m_sel)
                if r % g == r_sel % g:
                    valid = False
                    break
            if valid:
                found_r = r
                break
        
        if found_r is not None:
            selected.append((m, found_r))
            
    # If we have time, we can try to improve by adding odd numbers
    # or by doing a local search.
    # But the greedy on evens should be very strong.
    
    # Convert to required format
    result = [[m, r] for m, r in selected]
    
    # Fallback if empty (should not happen)
    if not result:
        result = [[20, 0]]
        
    return result
