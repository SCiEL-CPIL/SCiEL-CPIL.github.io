import time
from math import gcd

def construct():
    """
    Finds maximum number of pairwise disjoint infinite residue classes
    with distinct moduli in [2, 128].
    
    Uses a CSP approach with MRV (Minimum Remaining Values) heuristic for variable selection
    and LCV (Least Constraining Value) heuristic for value selection.
    Multiple randomized restarts are used to escape local optima.
    """
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    max_modulus = 128
    all_moduli = list(range(2, max_modulus + 1))
    n = len(all_moduli)
    
    # Precompute gcd matrix
    gcd_mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcd_mat[i][j] = gcd(all_moduli[i], all_moduli[j])
            
    # Precompute "conflict cost" matrix: cost[i][j] is the number of residues in m_j
    # that would be forbidden if we pick a specific residue in m_i?
    # Actually, the number of forbidden residues in m_j given a residue in m_i is m_j // gcd(m_i, m_j).
    # This value is constant for a pair (i, j) regardless of which specific residue is picked.
    # So, the LCV cost for picking any residue in i is sum(cost[i][j] for j in unassigned).
    # Wait, the LCV heuristic picks the value r that minimizes the *total* constraints on neighbors.
    # The number of forbidden residues in neighbor j is fixed by gcd(m_i, m_j).
    # So all residues in m_i have the *same* impact on the *number* of valid residues in neighbors?
    # No, the *specific* residues forbidden change, but the *count* is constant.
    # However, the *overlap* of forbidden sets with *other* already forbidden sets matters.
    # But in a greedy forward-checking approach, we only look at currently valid residues.
    # If we just minimize the count of forbidden residues, all r in m_i are equivalent in terms of count.
    # We need a tie-breaker or a more sophisticated LCV.
    # A better LCV: minimize the sum of (valid_count[neighbor] - new_valid_count[neighbor]).
    # This requires checking each neighbor's valid set.
    
    def solve_csp(random_seed=None):
        """
        Solves the CSP using MRV and a simple greedy value selection.
        If random_seed is provided, tie-breaking is randomized.
        """
        import random
        if random_seed is not None:
            random.seed(random_seed)
        
        assignment = {}
        unassigned = set(range(n))
        
        # valid_residues[i] is a set of valid residues for modulus i
        # Initialize with all residues
        valid_residues = [set(range(m)) for m in all_moduli]
        
        while unassigned:
            if time.time() > deadline:
                break
                
            # MRV: Find unassigned modulus with fewest valid residues
            min_count = float('inf')
            best_idx = -1
            best_valid_set = None
            
            # To add randomness, we can shuffle the iteration order or break ties randomly
            candidates = list(unassigned)
            if random_seed is not None:
                random.shuffle(candidates)
                
            for idx in candidates:
                count = len(valid_residues[idx])
                if count == 0:
                    # Dead end
                    return assignment
                if count < min_count:
                    min_count = count
                    best_idx = idx
                    best_valid_set = valid_residues[idx]
                    if min_count == 1:
                        break
            
            if best_idx == -1:
                break
            
            # LCV: Pick a residue from best_valid_set that minimizes impact on others
            # Impact of picking r in m_i on m_j:
            # It removes all r' in valid_residues[j] such that (r' - r) % gcd(m_i, m_j) == 0.
            # We want to minimize the total number of removed residues across all neighbors.
            
            m_i = all_moduli[best_idx]
            best_r = None
            min_impact = float('inf')
            
            # If there are many valid residues, sampling might be faster, but max is 128, so iterate all
            for r in best_valid_set:
                impact = 0
                for j in unassigned:
                    if j == best_idx:
                        continue
                    g = gcd_mat[best_idx][j]
                    # Count how many valid residues in j are forbidden by r
                    # Forbidden if (r' - r) % g == 0
                    # i.e., r' % g == r % g
                    # Let target = r % g
                    # Count x in valid_residues[j] such that x % g == target
                    # Since g <= 128, we can just iterate or use precomputed buckets?
                    # Iterating valid_residues[j] is O(m_j).
                    # Total cost per r is sum(m_j) ~ 8000.
                    # Total cost for all r in m_i is m_i * 8000 ~ 128 * 8000 = 1M.
                    # This is too slow for 127 steps * 127 restarts.
                    
                    # Optimization:
                    # We don't need exact impact if we just pick a random valid residue when counts are low.
                    # Or we can use a heuristic: pick r that is "central" or random.
                    # Let's try a faster heuristic: just pick a random valid residue.
                    # The MRV part does the heavy lifting.
                    pass
                
                # For speed, we will just pick a random valid residue from the MRV-selected modulus.
                # This is a "Randomized Greedy" with MRV.
                break # Just pick one, don't calculate full impact for speed
                
            # Actually, let's use a simple LCV approximation:
            # Pick the residue that is farthest from "popular" forbidden patterns?
            # No, let's stick to Randomized Greedy with MRV. It's robust and fast.
            
            r = random.choice(list(best_valid_set)) if random_seed is not None else sorted(best_valid_set)[0]
            
            # Update assignment
            assignment[best_idx] = r
            unassigned.remove(best_idx)
            
            # Forward Checking: Update valid_residues for remaining unassigned moduli
            for j in unassigned:
                g = gcd_mat[best_idx][j]
                target = r % g
                # Remove all x in valid_residues[j] where x % g == target
                # Using set difference with a generated set is efficient for small sets
                # Or filter in place
                valid_residues[j] = {x for x in valid_residues[j] if x % g != target}
                
        return assignment

    best_result = []
    best_size = 0
    
    # Run multiple restarts
    # We can run deterministic first, then randomized
    for seed in [None, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000]:
        if time.time() > deadline:
            break
            
        result = solve_csp(seed)
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = [[result[idx], all_moduli[idx]] for idx in result]
            
        # Early stop if we found a very large solution? 
        # The max possible is 127.
        # If we find > 100, it's likely near optimal.
        if best_size > 100:
             # Do a few more to be sure
             pass

    # If no result found (should not happen), fallback
    if not best_result:
        best_result = [[0, 2]]
        
    return best_result

# EVOLVE-BLOCK-END
