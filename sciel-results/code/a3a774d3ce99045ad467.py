import numpy as np
import time

def construct():
    base = 51975
    start_time = time.time()
    time_limit = 110  # Safety margin for 120s limit
    
    # Get all divisors of base > 1
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    used_moduli = set()
    
    # Minimum Set Cover greedy: at each step, pick the class (m, r) with unused modulus
    # that maximizes newly covered residues / cost. Cost = 1/m (smaller modulus = higher cost per coverage)
    # This prioritizes high-impact moduli early.
    
    while len(used_moduli) < len(divisors):
        if time.time() - start_time > time_limit:
            break
        
        best_m = None
        best_r = None
        best_gain = -1
        best_ratio = -1
        
        for m in divisors:
            if m in used_moduli:
                continue
            
            # For this modulus, find the residue that maximizes new coverage
            # New coverage = total positions - already covered positions
            # We want to maximize: (newly_covered) / (1/m) = newly_covered * m
            # But actually, for set cover, we want max newly_covered / cost
            # If cost is fixed per modulus, just maximize newly_covered
            
            best_r_for_m = 0
            max_newly = -1
            
            for r in range(m):
                # Count newly covered: positions in r::m that are not yet covered
                newly = base - covered[r::m].sum()
                if newly > max_newly:
                    max_newly = newly
                    best_r_for_m = r
            
            # Use ratio: newly_covered / (1/m) = newly_covered * m
            # This favors small moduli that cover many new residues
            ratio = max_newly * m
            
            if ratio > best_ratio or (ratio == best_ratio and max_newly > best_gain):
                best_ratio = ratio
                best_gain = max_newly
                best_m = m
                best_r = best_r_for_m
        
        if best_m is None:
            break
        
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        covered[best_r::best_m] = 1
    
    return classes
