"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from functools import lru_cache

def construct():
    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)  # 127 moduli
    
    # Precompute for each modulus m, the list of residues 0..m-1
    residues = {m: list(range(m)) for m in moduli}
    
    # For each pair of moduli (m1, m2), precompute compatibility:
    # residue r1 of m1 is compatible with residue r2 of m2 iff gcd(m1,m2) does NOT divide (r1-r2)
    # i.e., (r1 - r2) % gcd(m1, m2) != 0
    
    # We'll build the conflict graph implicitly.
    # For branch and bound, we process moduli in a fixed order but use dynamic filtering.
    
    # Precompute gcd for all pairs
    gcd_table = {}
    for i, m1 in enumerate(moduli):
        for j, m2 in enumerate(moduli):
            if i != j:
                gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    # For each modulus m and each residue r of m, and for each other modulus m2,
    # precompute the set of residues of m2 that are compatible with (m, r)
    # compatible(m, r, m2) = {r2 in 0..m2-1 : (r - r2) % gcd(m, m2) != 0}
    
    # To make this efficient, note that (r - r2) % g == 0 iff r2 % g == r % g
    # So incompatible residues of m2 are those with r2 % g == r % g, where g = gcd(m, m2)
    
    # Precompute for each m, m2, and residue r of m: the bitset of compatible residues of m2
    # Actually, let's think of it differently for the B&B.
    
    # B&B approach:
    # We assign residues to moduli one by one.
    # State: for each unassigned modulus, the set of remaining candidate residues.
    # When we assign residue r to modulus m, for each unassigned modulus m2,
    #   we remove from m2's candidates all r2 such that (r - r2) % gcd(m, m2) == 0
    
    # Upper bound: number of unassigned moduli that still have at least one candidate.
    
    # To speed up, use bitsets (Python ints) for candidate sets.
    
    # Precompute for each modulus m, for each residue r of m, and for each modulus m2:
    #   mask_incompatible[m][r][m2] = bitset of residues of m2 that are incompatible with (m, r)
    
    # This is too much memory potentially. Let's compute on the fly but cache.
    
    # Alternative: for each pair (m, m2) and each residue r of m, compute the incompatible mask for m2.
    # incompatible mask for m2 given (m, r): residues r2 where r2 % g == r % g, g = gcd(m, m2)
    
    # Let's precompute this.
    # For each m, for each r in 0..m-1, for each m2:
    #   g = gcd(m, m2)
    #   rem = r % g
    #   incompatible residues of m2: those r2 in 0..m2-1 with r2 % g == rem
    
    # Build mask as integer bitset.
    
    # Precompute masks
    # mask_compat[m_idx][r][m2_idx] -> bitset of compatible residues of m2
    # But that's huge. Instead, compute incompatible mask on the fly using precomputed per-(m,m2,g,rem) masks.
    
    # For each modulus m2 and each g that divides m2, and each rem in 0..g-1:
    #   mask_incompat[m2][g][rem] = bitset of residues r2 in 0..m2-1 with r2 % g == rem
    
    mask_incompat = {}
    for m2 in moduli:
        mask_incompat[m2] = {}
        for g in range(1, m2 + 1):
            if m2 % g != 0:
                continue
            # Actually g = gcd(m, m2) so g must divide m2. But g can be any divisor of m2 that is also a divisor of some m.
            # Let's just precompute for all g that divide m2.
            mask_incompat[m2][g] = {}
            for rem in range(g):
                mask = 0
                for r2 in range(m2):
                    if r2 % g == rem:
                        mask |= (1 << r2)
                mask_incompat[m2][g][rem] = mask
    
    # Now for B&B:
    # Order moduli to process. Let's use a fixed order but we can try different orderings.
    # Actually, for maximum independent set, the order matters for B&B efficiency.
    # Let's use the order of increasing number of residues (i.e., increasing modulus) or some heuristic.
    
    # Let's process moduli in order of increasing modulus (2, 3, 4, ..., 128)
    # But maybe better to process more constrained ones first.
    
    # For B&B, we maintain:
    #   current_assignment: list of (m, r) chosen so far
    #   candidate_masks: dict m -> bitset of available residues for m
    
    # Upper bound: count of moduli with non-zero candidate mask + current count
    
    # Let's implement B&B with memoization on (tuple of candidate masks) - but that's too large.
    # Instead, just recursive B&B with good pruning.
    
    best_solution = []
    best_size = 0
    
    def update_best(assignment):
        nonlocal best_solution, best_size
        if len(assignment) > best_size:
            best_size = len(assignment)
            best_solution = list(assignment)
    
    def count_nonzero(masks):
        return sum(1 for m in moduli if masks[m] != 0)
    
    def bb(assignment, masks, idx):
        nonlocal best_size
        # idx is the index in moduli order we're considering
        if idx >= n:
            update_best(assignment)
            return
        
        m = moduli[idx]
        mask = masks[m]
        
        if mask == 0:
            # Cannot assign m, skip it (it's not in the independent set)
            bb(assignment, masks, idx + 1)
            return
        
        # Upper bound: current + number of moduli from idx onward with non-zero mask
        ub = len(assignment) + count_nonzero({m2: masks[m2] for m2 in moduli[idx:]})
        if ub <= best_size:
            return
        
        # Try each residue of m
        # Dynamic ordering: try residues that leave the most options? Or just iterate.
        r = 0
        while mask:
            # Get lowest set bit
            lsb = mask & (-mask)
            r = lsb.bit_length() - 1
            mask_without = mask & (mask - 1)
            
            # Assign r to m
            new_masks = dict(masks)
            new_masks[m] = 0  # m is now assigned
            
            # For each unassigned modulus m2 > idx, update its mask
            feasible = True
            for j in range(idx + 1, n):
                m2 = moduli[j]
                if new_masks[m2] == 0:
                    continue
                g = math.gcd(m, m2)
                rem = r % g
                # Remove incompatible residues
                incompat = mask_incompat[m2][g][rem]
                new_masks[m2] &= ~incompat
                if new_masks[m2] == 0:
                    # This assignment makes m2 infeasible, but that's ok, we just skip m2 later
                    # Actually it's fine, m2 just won't be in the set
                    pass
            
            bb(assignment + [(m, r)], new_masks, idx + 1)
            
            mask = mask_without
    
    # Initial masks: all residues available
    init_masks = {m: (1 << m) - 1 for m in moduli}
    
    # This naive B&B might be too slow for 127 moduli. Let's add more pruning.
    # Actually, the upper bound count_nonzero is expensive. Let's maintain a counter.
    
    # Let's try a different approach: since the problem structure is special,
    # maybe we can find a near-optimal solution greedily and use it as a lower bound.
    
    # Greedy: process moduli in some order, pick the residue that maximizes future options.
    # Or just pick any valid residue.
    
    # Let's first try the B&B but with a time limit, and also try to get a good initial lower bound.
    
    # Actually, let me reconsider the problem. The current solution gives 7 classes.
    # The maximum is likely much higher. Let's think about what's achievable.
    
    # For modulus 2: residue 0 or 1
    # For modulus 3: residue 0, 1, or 2
    # Two classes are disjoint if gcd(m1,m2) doesn't divide (r1-r2)
    
    # If we pick residues randomly, the probability of conflict for a pair is 1/gcd(m1,m2)
    # So for large moduli with small gcd, conflicts are rare.
    
    # The key constraint is between moduli that share large gcds.
    
    # Let's just run the B&B and see. But 127 moduli is a lot.
    
    # Alternative: use a SAT-like approach or just a smarter greedy + local search.
    
    # Given the time constraint, let's implement a greedy with random restarts and local improvement.
    
    import random
    
    def greedy_solution():
        # Process moduli in order of decreasing "constraint" (those with many conflicts first)
        # Or just in order 2..128
        assignment = {}
        # For each modulus, track which residues are still available
        available = {m: set(range(m)) for m in moduli}
        
        # Order moduli by number of potential conflicts (sum of gcd with others)
        # Moduli with large gcds with many others are more constrained
        order = sorted(moduli, key=lambda m: -sum(math.gcd(m, m2) for m2 in moduli if m2 != m))
        
        for m in order:
            if not available[m]:
                continue
            # Pick the residue that leaves the most options for others
            best_r = None
            best_score = -1
            for r in available[m]:
                # Count how many future moduli still have options after picking r
                score = 0
                for m2 in order:
                    if m2 == m or m2 not in available:
                        continue
                    if not available[m2]:
                        continue
                    g = math.gcd(m, m2)
                    rem = r % g
                    # Count residues of m2 that would be removed
                    removed = sum(1 for r2 in available[m2] if r2 % g == rem)
                    score += len(available[m2]) - removed
                if score > best_score:
                    best_score = score
                    best_r = r
            if best_r is not None:
                assignment[m] = best_r
                # Update available for others
                for m2 in order:
                    if m2 == m:
                        continue
                    if not available[m2]:
                        continue
                    g = math.gcd(m, m2)
                    rem = best_r % g
                    available[m2] = {r2 for r2 in available[m2] if r2 % g != rem}
        
        return assignment
    
    # Try multiple random orderings
    best_assign = None
    best_count = 0
    
    for trial in range(200):
        random.shuffle(moduli)  # This modifies moduli, let's use a copy
        break
    
    # Let's do proper random restarts
    moduli_list = list(range(2, MAX_M + 1))
    
    best_assign = None
    best_count = 0
    
    for trial in range(500):
        order = list(moduli_list)
        random.shuffle(order)
        
        assignment = {}
        available = {m: set(range(m)) for m in moduli_list}
        
        for m in order:
            if not available[m]:
                continue
            # Pick residue randomly from available, or use heuristic
            # Use heuristic: pick residue that minimizes immediate damage
            best_r = None
            best_score = float('inf')
            for r in available[m]:
                damage = 0
                for m2 in order:
                    if m2 == m or not available[m2]:
                        continue
                    g = math.gcd(m, m2)
                    rem = r % g
                    removed = sum(1 for r2 in available[m2] if r2 % g == rem)
                    damage += removed
                if damage < best_score:
                    best_score = damage
                    best_r = r
            if best_r is not None:
                assignment[m] = best_r
                for m2 in order:
                    if m2 == m:
                        continue
                    if not available[m2]:
                        continue
                    g = math.gcd(m, m2)
                    rem = best_r % g
                    available[m2] = {r2 for r2 in available[m2] if r2 % g != rem}
        
        count = len(assignment)
        if count > best_count:
            best_count = count
            best_assign = dict(assignment)
    
    # Convert to required format
    result = [[r, m] for m, r in best_assign.items()]
    return result

# EVOLVE-BLOCK-END
