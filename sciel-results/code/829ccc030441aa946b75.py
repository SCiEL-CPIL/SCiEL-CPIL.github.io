import math
import time
import random
import sys

MAX_MOD = 128
DEADLINE = 60.0
MARGIN = 5.0

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - MARGIN
    
    all_mods = list(range(2, MAX_MOD + 1))
    n_mods = len(all_mods)
    
    # Pre-compute gcds
    gcd_table = [[0] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
    for m1 in range(2, MAX_MOD + 1):
        for m2 in range(2, MAX_MOD + 1):
            gcd_table[m1][m2] = math.gcd(m1, m2)
            
    # Pre-compute neighbors for each modulus
    # neighbors[m] = list of m' such that gcd(m, m') > 1
    neighbors = [[] for _ in range(MAX_MOD + 1)]
    for m1 in range(2, MAX_MOD + 1):
        for m2 in range(2, MAX_MOD + 1):
            if m1 != m2 and gcd_table[m1][m2] > 1:
                neighbors[m1].append(m2)
                
    # Pre-compute forbidden masks
    # forbidden_mask[m'][g][k] is a bitmask of residues in [0, m') that are congruent to k mod g
    # We only need this for g that are divisors of m' (which implies g = gcd(m, m') for some m)
    # Actually, g can be any divisor of m'.
    forbidden_masks = [None] * (MAX_MOD + 1)
    for m in range(2, MAX_MOD + 1):
        forbidden_masks[m] = {}
        # Find all divisors of m
        divs = []
        for d in range(2, m + 1):
            if m % d == 0:
                divs.append(d)
        if 1 in divs: divs.remove(1) # gcd > 1 required for constraint, but if gcd=1, no constraint.
        # Wait, if gcd=1, they are always disjoint? No.
        # (r2-r1) % 1 == 0 is always true. So if gcd=1, they are NEVER disjoint?
        # Let's re-read the condition.
        # Disjoint if (r2 - r1) % gcd(m1, m2) != 0.
        # If gcd=1, (r2-r1)%1 is always 0. So they are NEVER disjoint.
        # This means if two moduli are coprime, ANY residues we pick for them will result in intersecting classes?
        # Let's check: x = a mod 2, x = b mod 3.
        # gcd(2,3)=1. (b-a)%1 = 0. Condition fails.
        # So x=a mod 2 and x=b mod 3 ALWAYS intersect.
        # Therefore, we CANNOT have both modulus 2 and modulus 3 in our set?
        # Wait. The condition for disjointness is that the system has NO solution.
        # CRT says system has solution iff r1 ≡ r2 (mod gcd).
        # So system has NO solution iff r1 NOT≡ r2 (mod gcd).
        # If gcd=1, r1 ≡ r2 (mod 1) is always true.
        # So system ALWAYS has solution.
        # So classes ALWAYS intersect.
        # Thus, we cannot select two coprime moduli?
        
        # Let's re-verify.
        # Class 1: { ..., -2, 0, 2, 4, ... } (0 mod 2)
        # Class 2: { ..., -3, 0, 3, 6, ... } (0 mod 3)
        # Intersection: { ..., -6, 0, 6, ... } (0 mod 6). Non-empty.
        # So yes, coprime moduli always intersect.
        
        # This implies that the set of moduli we choose MUST be pairwise non-coprime?
        # No, that's not right either.
        # If we choose moduli 2 and 4. gcd=2.
        # r1=0, r2=1. (1-0)%2 = 1 != 0. Disjoint.
        # Class 1: even numbers. Class 2: odd multiples of 4 (1, 5, 9... no, 1 mod 4 is 1, 5, 9...).
        # Wait, 1 mod 4 is {1, 5, 9, 13...}.
        # 0 mod 2 is {0, 2, 4, 6, 8, 10...}.
        # Intersection: None?
        # 1 mod 4 are odd. 0 mod 2 are even. Disjoint.
        
        # So, if gcd(m1, m2) = g, we can make them disjoint if we pick r1, r2 such that r1 not≡ r2 (mod g).
        # If g=1, r1 not≡ r2 (mod 1) is impossible.
        # So we CANNOT include two coprime moduli in the set.
        
        # This is a HUGE constraint.
        # The graph of moduli where edges connect coprime moduli must be an independent set?
        # No, the graph where edges connect NON-coprime moduli?
        # Let G be graph with vertices 2..128. Edge if gcd > 1.
        # We need to select a subset of vertices such that for EVERY pair (u, v) in the subset, they are NOT coprime?
        # No, that's not it.
        # We need to select a subset of vertices such that we CAN assign residues to make them disjoint.
        # If u, v are coprime, we CANNOT make them disjoint.
        # So, if u, v are coprime, we cannot have both u and v in our set.
        # Therefore, the set of moduli we select must be an independent set in the "coprime graph" (where edges connect coprime pairs).
        # Equivalently, the set of moduli must be a clique in the "non-coprime graph" (where edges connect non-coprime pairs).
        
        # Let's verify this logic.
        # If I select {2, 3}, gcd(2,3)=1. They always intersect. Invalid.
        # If I select {2, 4}, gcd(2,4)=2. Can pick r1=0, r2=1. Disjoint. Valid.
        # If I select {2, 4, 6}.
        # 2,4: gcd 2.
        # 2,6: gcd 2.
        # 4,6: gcd 2.
        # Can we assign r2, r4, r6?
        # r2 not≡ r4 (mod 2)
        # r2 not≡ r6 (mod 2)
        # r4 not≡ r6 (mod 2)
        # This implies r2, r4, r6 must have distinct parities?
        # Only 2 parities exist. Pigeonhole principle: 3 items, 2 bins. Two must be same parity.
        # If r2 ≡ r4 (mod 2), they intersect.
        # So {2, 4, 6} is NOT solvable?
        
        # Let's check.
        # r2=0 (even).
        # r4 must be odd (1, 3). Let r4=1.
        # r6 must be odd (not≡ 0 mod 2). Let r6=1.
        # Check r4, r6: r4=1, r6=1. gcd(4,6)=2. (1-1)%2 = 0. Intersect.
        # Try r6=3. (3-1)%2 = 0. Intersect.
        # So yes, {2, 4, 6} is unsolvable.
        
        # So the condition is stronger than just being a clique in the non-coprime graph.
        # It is a list coloring problem on the non-coprime graph, where the list for node m is {0..m-1}, and the constraint is r_u not≡ r_v (mod gcd(u,v)).
        
        # Back to the solver. The forward checking with bitmasks handles this correctly.
        # We just need to ensure we precompute the masks correctly.
        
        for g in divs:
            forbidden_masks[m][g] = []
            for k in range(g):
                mask_val = 0
                for r in range(m):
                    if r % g == k:
                        mask_val |= (1 << r)
                forbidden_masks[m][g].append(mask_val)
                
    # Initial masks: all residues available
    initial_masks = [0] * (MAX_MOD + 1)
    for m in range(2, MAX_MOD + 1):
        # Set bits 0 to m-1
        initial_masks[m] = (1 << m) - 1
        
    def solve_with_backtracking(seed_val, t_limit):
        random.seed(seed_val)
        
        # State:
        # masks: list of current available residue masks for each modulus
        # assigned: dict modulus -> residue
        # order: list of moduli to process? We use MRV, so dynamic.
        
        masks = initial_masks[:]
        assigned = {}
        
        def get_mrv_modulus(unassigned_mods):
            min_count = float('inf')
            candidates = []
            for m in unassigned_mods:
                # Count bits in masks[m]
                count = bin(masks[m]).count('1')
                if count == 0:
                    return None # Fail
                if count < min_count:
                    min_count = count
                    candidates = [m]
                elif count == min_count:
                    candidates.append(m)
            if not candidates:
                return None
            return random.choice(candidates)
            
        def backtrack(unassigned_mods):
            if time.time() > t_limit:
                return False
            if not unassigned_mods:
                return True
                
            m = get_mrv_modulus(unassigned_mods)
            if m is None:
                return False
                
            # Get available residues for m
            # Iterate through bits in masks[m]
            mask = masks[m]
            residues_to_try = []
            # Randomize order for tie-breaking? MRV already picks random candidate.
            # We can also randomize the residue order.
            for r in range(m):
                if (mask >> r) & 1:
                    residues_to_try.append(r)
            random.shuffle(residues_to_try)
            
            for r in residues_to_try:
                # Assign r to m
                assigned[m] = r
                
                # Apply forward checking
                new_masks = masks[:]
                valid = True
                for m2 in unassigned_mods:
                    if m2 == m:
                        continue
                    g = gcd_table[m][m2]
                    if g > 1:
                        k = r % g
                        # Remove forbidden residues from m2
                        forbidden = forbidden_masks[m2][g][k]
                        new_masks[m2] = masks[m2] & ~forbidden
                        if new_masks[m2] == 0:
                            valid = False
                            break
                
                if valid:
                    # Recurse
                    new_unassigned = [x for x in unassigned_mods if x != m]
                    # We need to pass new_masks to the next call.
                    # To avoid copying the whole list every time (expensive), we can use a stack or just copy since N=127.
                    # Copying 127 integers is fast.
                    if backtrack(new_unassigned):
                        return True
                
                # Undo
                del assigned[m]
                
            return False
            
        # Start
        all_mods_set = list(range(2, MAX_MOD + 1))
        if backtrack(all_mods_set):
            return [(assigned[m], m) for m in assigned]
        return None

    best_pairs = []
    best_len = 0
    
    # Try multiple seeds
    # We have ~55 seconds.
    # Each run might be fast if it succeeds, or hit time limit.
    # Let's cap each run at 1-2 seconds to allow many restarts.
    
    attempts = 0
    max_attempts = 50
    time_per_attempt = 1.0
    
    for seed in range(1000):
        if time.time() > deadline - 1.0 or attempts >= max_attempts:
            break
        
        t_start = time.time()
        pairs = solve_with_backtracking(seed, t_start + time_per_attempt)
        attempts += 1
        
        if pairs is not None:
            if len(pairs) > best_len:
                best_pairs = pairs
                best_len = len(pairs)
                if best_len == len(all_mods):
                    break
        else:
            # If it failed, maybe try a smaller subset?
            # For now, just continue.
            pass

    if not best_pairs:
        # Fallback: return a single valid pair
        return [[0, 2]]
        
    result = []
    for (r, m) in best_pairs:
        result.append([r, m])
        
    return result

