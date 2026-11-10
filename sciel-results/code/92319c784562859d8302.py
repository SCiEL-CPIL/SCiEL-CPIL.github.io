import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    candidates = list(range(20, 201))
    n = len(candidates)

    # Precompute gcd matrix
    gcd_matrix = [[gcd(candidates[i], candidates[j]) for j in range(n)] for i in range(n)]

    # We want to maximize sum of 1/m over selected classes.
    # Greedy: try to include all moduli, but we need to check if a valid assignment exists.
    # This is a constraint satisfaction problem.
    # Since n=181, we can't do brute force.
    # However, the constraints are local (pairwise).
    # Let's use a smarter greedy with backtracking or local search.
    
    # Strategy:
    # 1. Start with an empty set.
    # 2. Add moduli in order of decreasing 1/m (i.e., increasing m).
    #    Wait, maximizing sum of 1/m means we prefer small m.
    #    So we should try to include small moduli first.
    # 3. For each new modulus m, try to find a residue r that is compatible with all existing.
    # 4. If no such r exists, we might need to change a previous residue or skip m.
    #    Changing previous residues is expensive.
    #    Skipping m loses 1/m.
    #    Maybe a local search on the set of moduli is better?
    #    Or, we can use a CSP solver approach.
    
    # Let's try a different angle:
    # The condition is r1 % gcd(m1, m2) != r2 % gcd(m1, m2).
    # This is equivalent to: the system of congruences x ≡ r1 (mod m1), x ≡ r2 (mod m2) has no solution.
    # By CRT, a solution exists iff r1 ≡ r2 (mod gcd(m1, m2)).
    # So we need to choose r_i such that for all i != j, r_i NOT≡ r_j (mod gcd(m_i, m_j)).
    
    # This is a graph coloring problem on a graph where edges are weighted by gcd.
    # Actually, it's a constraint satisfaction problem.
    
    # Let's try a greedy algorithm that adds moduli in increasing order of m (small m first).
    # For each m, try all residues r in [0, m) and pick one that satisfies constraints with all previous.
    # If none works, we might need to backtrack.
    # Given the density, backtracking might be deep.
    
    # Alternative: Use random restarts with greedy.
    # Or, use a local search: start with a random valid set, then try to add more moduli or swap residues.
    
    # Let's stick to the greedy with lookahead but improve the cost function.
    # The previous cost was "number of residues blocked in future moduli".
    # A better cost might be "sum of 1/m_future for blocked residues" or similar.
    
    # Actually, the previous program scored 2.99. The inspiration scored 3.10.
    # Let's look at the inspiration: "uses math, random, time".
    # This suggests randomization might help.
    
    # Let's try a randomized greedy:
    # 1. Shuffle the order of moduli? No, small m are more valuable.
    # 2. When choosing r for m, try multiple random valid r's and pick the best based on lookahead.
    
    # Let's implement a greedy that tries to include ALL moduli.
    # If it fails, it means some moduli must be dropped.
    # We want to drop the ones with smallest 1/m (largest m).
    # So, process m in increasing order.
    # If we can't find a valid r for m, we skip m.
    # But maybe skipping a larger m allows us to keep more smaller ones?
    # No, we are processing in increasing order, so we've already fixed the smaller ones.
    # If we can't place m, we skip it.
    
    # To improve, we can use backtracking: if we get stuck, try changing a previous r.
    # But backtracking is expensive.
    
    # Let's try a local search on the residues.
    # 1. Start with a greedy solution (possibly with some moduli skipped).
    # 2. Try to add a skipped modulus by changing residues of existing moduli.
    # 3. Or, try to swap residues to free up space for a new modulus.
    
    # Actually, let's just use a simple randomized greedy with multiple restarts.
    
    best_score = 0.0
    best_solution = []
    
    import random
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Randomize the order of residues tried for each modulus
        # Or randomize the choice among valid residues
        
        selected = []
        # Process moduli in increasing order
        for m in candidates:
            valid_rs = []
            for r in range(m):
                valid = True
                for (m2, r2) in selected:
                    g = gcd(m, m2)
                    if r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    valid_rs.append(r)
            
            if not valid_rs:
                continue
            
            # Pick a residue. 
            # To maximize future flexibility, pick the one that blocks the fewest future options.
            # Cost: for each future modulus m2, count how many residues are blocked.
            # A residue r blocks residue r2 in m2 if r2 % gcd(m, m2) == r % gcd(m, m2).
            # Number of blocked residues in m2 is m2 / gcd(m, m2).
            # So the cost is sum over future m2 of (m2 / gcd(m, m2)) for each blocked r2?
            # No, it's simpler: for a given r, the number of blocked residues in m2 is exactly m2/g.
            # So the cost is independent of r! 
            # Wait, is it?
            # r % g == r2 % g. The number of r2 in [0, m2) satisfying this is m2/g.
            # Yes, it's independent of r.
            # So the "blocking" cost is the same for all valid r.
            # This means we should pick r randomly or based on some other heuristic.
            
            # Maybe we should pick r to be "spread out"?
            # Or just pick the first valid one?
            
            # Let's pick randomly from valid_rs to diversify.
            r = random.choice(valid_rs)
            selected.append((m, r))
        
        # Calculate score
        score = sum(1.0/m for m, r in selected)
        if score > best_score:
            best_score = score
            best_solution = selected[:]
    
    if not best_solution:
        return [[20, 0]]
    
    return [[m, r] for m, r in best_solution]
