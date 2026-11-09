import time, math, random
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    def now(): return time.time()
    
    all_m = list(range(2, 129))
    n = len(all_m)
    
    # Precompute gcd matrix
    G = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = math.gcd(all_m[i], all_m[j])
            G[i][j] = g
            G[j][i] = g
    
    # Two moduli i, j with gcd g are "compatible" if there exist residues
    # r_i in [0, m_i) and r_j in [0, m_j) such that r_i % g != r_j % g.
    # This is ALWAYS possible as long as g < m_i and g < m_j (which is always true
    # since g = gcd(m_i, m_j) <= min(m_i, m_j), and if g == m_i then m_i | m_j,
    # but then we need r_i % m_i != r_j % m_i, which requires m_i >= 2, always true).
    # Wait - actually the constraint is: for ALL pairs in the chosen set, the residues
    # must satisfy the pairwise condition. The key insight is that we need to assign
    # residues such that for every pair (i,j) with gcd g, r_i % g != r_j % g.
    #
    # This is a constraint satisfaction problem. The maximum number of moduli we can
    # pick is at most 127 (all of them), but the residue constraints may prevent it.
    #
    # Key observation: if we pick moduli that are all pairwise coprime, then gcd=1 for
    # all pairs, and r_i % 1 = 0 = r_j % 1 for all residues, so NO two coprime moduli
    # can be in the set! Wait, that means we can't have any two coprime moduli.
    #
    # Actually wait: gcd(m_i, m_j) = 1 means the constraint is r_i % 1 != r_j % 1,
    # but 0 != 0 is false. So two coprime moduli CANNOT both be in the set.
    #
    # So we need all pairs to have gcd >= 2. This means the set of moduli must be
    # such that every pair shares a common factor >= 2.
    #
    # This is equivalent to: the moduli must form a "2-wise intersecting" family in
    # terms of prime factors. By the Erdős–Ko–Rado type results, the largest such
    # families are typically those sharing a common prime factor.
    #
    # Strategy: pick all moduli divisible by 2 (evens from 2 to 128): that's 63 moduli.
    # Then gcd of any two evens is at least 2. Now we need to assign residues such that
    # for every pair, r_i % gcd(m_i, m_j) != r_j % gcd(m_i, m_j).
    #
    # For all-even moduli, let's try r_i = 0 for all. Then r_i % g = 0 for all g,
    # which violates the constraint. So we need different residues.
    #
    # Better approach: use branch-and-bound / backtracking with bitsets.
    # But with up to 127 nodes, this is intractable in general.
    #
    # Alternative: focus on a structured family. All multiples of 2: 63 moduli.
    # For these, gcd(m_i, m_j) >= 2. We need r_i % gcd(m_i,m_j) != r_j % gcd(m_i,m_j).
    #
    # Let's try: for modulus m (even), set r = m//2. Then for two evens m1, m2 with
    # gcd g: r1 % g = (m1//2) % g, r2 % g = (m2//2) % g.
    # Since g | m1 and g | m2, m1//2 mod g: m1 = g*k1, so m1//2 = g*k1/2 if k1 even,
    # or (g*k1)//2. This is getting complicated.
    #
    # Let me just do a smart backtracking with good ordering and pruning.
    # Use the fact that we can order moduli by descending number of constraints.
    
    # Actually, let me reconsider. The maximum possible is bounded by the size of the
    # largest subset of {2,...,128} where every pair has gcd >= 2.
    # All evens: 63 moduli, all pairs have gcd >= 2. Can we do better by mixing?
    # If we include an odd number m, it must share a factor >= 2 with every other
    # modulus in the set. So all other moduli must be divisible by some prime factor of m.
    # E.g., if m=3, all others must be divisible by 3. Multiples of 3 in [2,128]: 42.
    # But 3 itself is odd and not even, so we can't mix with evens easily.
    #
    # Best bet: all evens (63 moduli). Now solve the CSP for residues.
    
    # For the all-even family, let's use a greedy + local search approach
    # with a specific residue assignment strategy.
    
    # For even modulus m, the key constraint with another even modulus m' is:
    # r % gcd(m,m') != r' % gcd(m,m')
    # 
    # If we set r = 0 for m=2 (only option that works well), then for any other even m',
    # gcd(2, m') = 2, so we need r' % 2 != 0, i.e., r' must be odd.
    #
    # For m=4: gcd(2,4)=2, need r_4 % 2 != 0 -> r_4 odd. gcd(4,4) N/A.
    # For m=6: gcd(2,6)=2, need r_6 % 2 != 0. gcd(4,6)=2, need r_6 % 2 != r_4 % 2.
    #   If r_4 is odd (say 1), then r_6 % 2 != 1, so r_6 % 2 = 0, but we also need
    #   r_6 % 2 != 0 from m=2. Contradiction! So r_4 and r_6 can't both work with r_2=0.
    #
    # Wait, let me recheck. m=2, r_2=0. m=4, r_4 must be odd (from gcd(2,4)=2: 0%2=0,
    # need r_4%2 != 0). m=6, r_6 must be odd (from gcd(2,6)=2: 0%2=0, need r_6%2 != 0).
    # Now gcd(4,6)=2: need r_4%2 != r_6%2. But both are odd, so both %2=1. Contradiction!
    #
    # So we can't have m=2, m=4, m=6 all together with any valid residue assignment?
    # Let me verify: we need r_2%2 != r_4%2, r_2%2 != r_6%2, r_4%2 != r_6%2.
    # r_2 can be 0 or 1 (mod 2). If r_2=0: r_4%2=1, r_6%2=1, but r_4%2 != r_6%2 fails.
    # If r_2=1: r_4%2=0, r_6%2=0, but r_4%2 != r_6%2 fails.
    # So indeed {2,4,6} is impossible!
    #
    # This means the problem is much harder than I thought. Let me use a proper
    # backtracking solver with bitset optimization.
    
    # Backtracking with constraint propagation
    # Order moduli to maximize pruning
    
    # First, determine which subsets of moduli are even possible (all pairs gcd>=2)
    # Then solve the CSP.
    
    # Let's use a different approach: start with a greedy construction and then
    # use local search to maximize the count.
    
    # Greedy: pick moduli one by one, trying to maintain feasibility.
    # For each candidate modulus, check if there exists a residue consistent with
    # all already-chosen moduli.
    
    def can_assign(i, chosen, G, all_m):
        """Check if modulus i can be added with some residue to the chosen set."""
        m_i = all_m[i]
        # For each residue r in [0, m_i), check compatibility with all chosen
        # This is expensive, so use the gcd structure.
        # The constraint for each chosen (j, r_j) is: r % gcd(m_i, m_j) != r_j % gcd(m_i, m_j)
        # So r must avoid specific residue classes modulo gcd(m_i, m_j) for each j.
        
        # Collect all forbidden conditions
        # For each j: r mod g != r_j mod g, where g = gcd(m_i, m_j)
        # This means r mod g is in {0,...,g-1} \ {r_j mod g}
        
        # Use CRT-like reasoning or just check
        # For small g values, this is manageable.
        
        # Simple approach: try all residues
        for r in range(m_i):
            ok = True
            for (j, r_j) in chosen:
                g = G[i][j]
                if g < 2:
                    return False  # coprime, impossible
                if (r % g) == (r_j % g):
                    ok = False
                    break
            if ok:
                return True
        return False
    
    def find_residue(i, chosen, G, all_m):
        """Find a valid residue for modulus i given the chosen set."""
        m_i = all_m[i]
        for r in range(m_i):
            ok = True
            for (j, r_j) in chosen:
                g = G[i][j]
                if g < 2:
                    return None
                if (r % g) == (r_j % g):
                    ok = False
                    break
            if ok:
                return r
        return None
    
    # Order moduli by difficulty (most constrained first)
    # A modulus is more constrained if it has many small gcds with others
    order = sorted(range(n), key=lambda i: (
        sum(1 for j in range(n) if j != i and G[i][j] == 1),  # coprime count
        all_m[i]  # smaller modulus = fewer residue options
    ))
    
    # Greedy construction
    chosen = []
    for i in order:
        if now() - t0 > DEADLINE:
            break
        r = find_residue(i, chosen, G, all_m)
        if r is not None:
            chosen.append((i, r))
    
    # Local search: try to add more moduli by removing and re-adding
    def try_add(chosen, G, all_m):
        included = set(i for i, _ in chosen)
        for i in range(n):
            if i in included:
                continue
            r = find_residue(i, chosen, G, all_m)
            if r is not None:
                return (i, r)
        return None
    
    def local_search(chosen, G, all_m, max_iters=500):
        for it in range(max_iters):
            if now() - t0 > DEADLINE:
                break
            # Try to add
            add = try_add(chosen, G, all_m)
            if add is not None:
                chosen.append(add)
                continue
            # Try 1-swap: remove one, try to add two
            if len(chosen) < 2:
                break
            removed = random.choice(chosen)
            chosen2 = [x for x in chosen if x != removed]
            # Try to add back removed and one more
            improved = False
            for i in range(n):
                if i == removed[0]:
                    continue
                if i in set(x[0] for x in chosen2):
                    continue
                r = find_residue(i, chosen2, G, all_m)
                if r is not None:
                    chosen2.append((i, r))
                    # Also try to add removed back
                    r2 = find_residue(removed[0], chosen2, G, all_m)
                    if r2 is not None:
                        chosen2.append((removed[0], r2))
                    if len(chosen2) > len(chosen):
                        chosen = chosen2
                        improved = True
                        break
            if not improved:
                # Try removing a different one
                pass
        return chosen
    
    random.seed(42)
    chosen = local_search(chosen, G, all_m)
    
    # Try multiple random restarts
    best = chosen
    best_len = len(chosen)
    
    for restart in range(50):
        if now() - t0 > DEADLINE - 5:
            break
        random.shuffle(order)
        chosen_r = []
        for i in order:
            if now() - t0 > DEADLINE - 5:
                break
            r = find_residue(i, chosen_r, G, all_m)
            if r is not None:
                chosen_r.append((i, r))
        chosen_r = local_search(chosen_r, G, all_m, max_iters=100)
        if len(chosen_r) > best_len:
            best = chosen_r
            best_len = len(chosen_r)
    
    # Verify
    def verify(chosen):
        for a in range(len(chosen)):
            for b in range(a + 1, len(chosen)):
                i, r = chosen[a]
                j, r2 = chosen[b]
                g = G[i][j]
                if g < 2:
                    return False
                if (r % g) == (r2 % g):
                    return False
        return True
    
    if not verify(best):
        # Fallback: just moduli 2 and 4
        best = [(0, 0), (2, 1)]  # m=2 r=0, m=4 r=1: gcd=2, 0%2=0, 1%2=1, OK
    
    result = [[r, all_m[i]] for (i, r) in best]
    return result
