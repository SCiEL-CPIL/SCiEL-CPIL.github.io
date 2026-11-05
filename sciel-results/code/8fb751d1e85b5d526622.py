import math
import time
import random

def construct():
    deadline = time.time() + 115  # Safety margin
    best_score = 0.0
    best_solution = []
    
    # Moduli range
    moduli = list(range(20, 201))
    
    # We will run multiple random restarts
    # Estimate time per run:
    # For each m (180 times):
    #   Create forbidden array of size m (avg 110)
    #   For each selected (avg 50-100):
    #     Mark forbidden residues (m/g steps)
    #   Pick random valid residue
    # Total ops ~ 180 * 100 * (110/avg_g)
    # avg_g is small, so 110/1 = 110.
    # 180 * 100 * 110 = 2,000,000 ops.
    # Python: ~0.1s - 0.2s per run.
    # So ~500-1000 runs possible.
    
    for _ in range(2000):
        if time.time() > deadline:
            break
        
        selected = [] # list of (m, r)
        # Sort moduli? We must process in increasing order to ensure disjointness check is against previous ones.
        # The problem doesn't require sorted output, but the construction logic relies on checking against previously added ones.
        # Since disjointness is symmetric, we can add in any order as long as we check against all previously added.
        # Adding smaller m first is better for score (higher weight).
        
        for m in moduli:
            # Determine forbidden residues for m
            forbidden = [False] * m
            for m2, r2 in selected:
                g = math.gcd(m, m2)
                # r is forbidden if r % g == r2 % g
                start = r2 % g
                for r in range(start, m, g):
                    forbidden[r] = True
            
            # Find valid residues
            valid_residues = [r for r in range(m) if not forbidden[r]]
            
            if valid_residues:
                # Heuristic: Pick one. Random is good for diversity.
                # Could also pick the one that minimizes future conflicts, but random is simpler.
                r = random.choice(valid_residues)
                selected.append((m, r))
        
        # Calculate score
        score = sum(1.0 / m for m, r in selected)
        
        if score > best_score:
            best_score = score
            best_solution = [[m, r] for m, r in selected]
            
    return best_solution
