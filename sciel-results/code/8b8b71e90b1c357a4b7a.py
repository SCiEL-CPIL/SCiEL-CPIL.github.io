import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Moduli in [20, 200], sorted by 1/m descending (small moduli first)
    moduli = list(range(20, 201))
    moduli.sort(key=lambda m: -1.0/m)  # descending 1/m
    n = len(moduli)
    mod_index = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd for each pair
    gcd_table = [[1]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # For each modulus i and residue r, precompute a bitset of (j, r_j) pairs
    # that conflict with (i, r). But we need per-modulus bitsets for efficiency.
    # 
    # Strategy: For modulus i with residue r, and another modulus j with gcd g:
    #   conflict if (r - r_j) % g == 0, i.e., r_j ≡ r (mod g)
    #   So r_j in {r mod g, r mod g + g, r mod g + 2g, ...} within [0, m_j)
    #
    # For each modulus j, precompute for each residue class c mod g (where g = gcd(m_i, m_j)),
    # a bitset of residues r_j in [0, m_j) such that r_j ≡ c (mod g).
    #
    # Actually, let's precompute: for each pair (i, j) with gcd g > 1,
    #   conflict_mask[i][j] = list of g bitsets, where conflict_mask[i][j][c] is a bitset
    #   of residues r_j in [0, m_j) with r_j ≡ c (mod g).
    # Then for (i, r), the forbidden residues for j are conflict_mask[i][j][r % g].
    
    # Precompute conflict masks
    # conflict_masks[i][j] = list of g bitmasks (for g = gcd(m_i, m_j))
    # Each bitmask is an integer where bit r_j is set if r_j ≡ c (mod g)
    conflict_masks = {}
    for i in range(n):
        conflict_masks[i] = [None] * n
        for j in range(n):
            if i == j:
                continue
            g = gcd_table[i][j]
            if g <= 1:
                conflict_masks[i][j] = None  # no conflict possible
            else:
                m_j = moduli[j]
                masks = [0] * g
                for r_j in range(m_j):
                    c = r_j % g
                    masks[c] |= (1 << r_j)
                conflict_masks[i][j] = masks
    
    # For quick access: given modulus index i and residue r, and modulus index j,
    # the forbidden residues for j is conflict_masks[i][j][r % gcd_table[i][j]]
    
    # Upper bound: sum of 1/m for all remaining moduli (admissible since we can always
    # pick at most one residue per modulus, and the constraint only restricts which)
    # Actually this isn't tight. A better bound: for each remaining modulus, 1/m is achievable
    # if there exists ANY valid residue. But computing that is expensive.
    # Use the simple bound: sum of 1/m for all unprocessed moduli.
    
    suffix_sum = [0.0] * (n + 1)
    for i in range(n-1, -1, -1):
        suffix_sum[i] = suffix_sum[i+1] + 1.0/moduli[i]
    
    best_score = Fraction(0, 1)
    best_classes = []
    
    def search(pos, chosen, current_score, forbidden_bitsets):
        """
        pos: index in moduli to consider next
        chosen: list of (m, r) chosen so far
        current_score: Fraction, sum of 1/m for chosen
        forbidden_bitsets: list of n integers, where forbidden_bitsets[j] has bit r_j set
                          if residue r_j for modulus j is forbidden
        """
        nonlocal best_score, best_classes
        
        if time.time() > deadline:
            return
        
        # Upper bound: current_score + sum of 1/m for all remaining moduli
        # But we can only achieve 1/m for modulus j if at least one residue is not forbidden
        ub = float(current_score)
        for j in range(pos, n):
            # Check if any residue is available for modulus j
            all_mask = (1 << moduli[j]) - 1
            if (forbidden_bitsets[j] & all_mask) != all_mask:
                ub += 1.0 / moduli[j]
        
        if ub <= float(best_score):
            return
        
        if pos == n:
            if current_score > best_score:
                best_score = current_score
                best_classes = [list(c) for c in chosen]
            return
        
        m = moduli[pos]
        # Try each residue for modulus m
        all_mask = (1 << m) - 1
        available = all_mask & ~forbidden_bitsets[pos]
        
        if available == 0:
            # No valid residue for this modulus, skip it
            search(pos + 1, chosen, current_score, forbidden_bitsets)
            return
        
        # Try residues. We want to try all of them, but for efficiency,
        # note that different residues for the same modulus have the same "cost" (1/m)
        # but different effects on future moduli. We should try the one that
        # forbids the fewest future options first (or use the one that's best).
        # For correctness of B&B, we need to try all. But we can prune:
        # if two residues r1, r2 for modulus m have the same effect on all future
        # moduli, we only need to try one.
        
        # For speed, let's try residues in order and use the bound to prune.
        # Iterate through available residues
        r = 0
        while r < m:
            if time.time() > deadline:
                return
            
            if available & (1 << r):
                # Try choosing (m, r)
                new_forbidden = list(forbidden_bitsets)  # copy
                new_chosen = chosen + [[m, r]]
                new_score = current_score + Fraction(1, m)
                
                # Update forbidden bitsets for future moduli
                for j in range(pos + 1, n):
                    g = gcd_table[pos][j]
                    if g > 1:
                        c = r % g
                        new_forbidden[j] |= conflict_masks[pos][j][c]
                
                search(pos + 1, new_chosen, new_score, new_forbidden)
            
            r += 1
        
        # Also try skipping this modulus
        search(pos + 1, chosen, current_score, forbidden_bitsets)
    
    # Initial forbidden bitsets: all zeros
    initial_forbidden = [0] * n
    search(0, [], Fraction(0, 1), initial_forbidden)
    
    # If B&B didn't find anything (shouldn't happen), fallback to greedy
    if not best_classes:
        # Simple greedy: pick smallest moduli with any valid residue
        classes = []
        used = set()
        forbidden = [0] * n
        for pos in range(n):
            m = moduli[pos]
            all_mask = (1 << m) - 1
            available = all_mask & ~forbidden[pos]
            if available:
                r = available & (-available)  # lowest set bit
                r = r.bit_length() - 1
                classes.append([m, r])
                for j in range(pos + 1, n):
                    g = gcd_table[pos][j]
                    if g > 1:
                        c = r % g
                        forbidden[j] |= conflict_masks[pos][j][c]
        best_classes = classes
    
    return best_classes
