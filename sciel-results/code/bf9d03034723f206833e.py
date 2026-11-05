import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin under 90s kill

    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def compatible(r1, m1, r2, m2):
        g = gcd(m1, m2)
        return (r1 - r2) % g != 0

    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    pf_counts = {m: num_prime_factors(m) for m in all_moduli}

    def find_valid_residue(m, chosen, rng, limit=200):
        # Forward-checking: find a residue r in [0,m) such that for every
        # (r_k, m_k) in chosen, gcd(m, m_k) does not divide (r - r_k).
        # Equivalently, r mod g_k != r_k mod g_k for each k.
        # Build forbidden sets per modulus via CRT-like intersection.
        # Simpler: just try residues, but order them intelligently.
        # Collect constraints: for each (r_k, m_k), g = gcd(m, m_k),
        # forbidden residue class: r ≡ r_k (mod g).
        constraints = []
        for r_k, m_k in chosen:
            g = gcd(m, m_k)
            if g > 1:
                constraints.append((g, r_k % g))
        
        if not constraints:
            return 0

        # Try to find r by checking constraints.
        # Use a small search: try r in a smart order.
        # Heuristic: try residues that are NOT congruent to forbidden values.
        # For small m, just brute force.
        if m <= 64:
            for r in range(m):
                ok = True
                for g, fk in constraints:
                    if r % g == fk:
                        ok = False
                        break
                if ok:
                    return r
            return None
        else:
            # Larger m: sample + targeted
            tried = set()
            # First try 0, m-1, and some structured values
            candidates = [0, m - 1, m // 2, (m // 2) + 1]
            for r in candidates:
                if r in tried or r >= m:
                    continue
                tried.add(r)
                ok = True
                for g, fk in constraints:
                    if r % g == fk:
                        ok = False
                        break
                if ok:
                    return r
            # Random sampling
            for _ in range(limit):
                r = rng.randint(0, m - 1)
                if r in tried:
                    continue
                tried.add(r)
                ok = True
                for g, fk in constraints:
                    if r % g == fk:
                        ok = False
                        break
                if ok:
                    return r
            # Fallback: full scan (bounded)
            for r in range(m):
                if r in tried:
                    continue
                ok = True
                for g, fk in constraints:
                    if r % g == fk:
                        ok = False
                        break
                if ok:
                    return r
            return None

    def greedy_build(order, rng, residue_mode='smart'):
        chosen = []
        for m in order:
            r = find_valid_residue(m, chosen, rng)
            if r is not None:
                chosen.append((r, m))
        return chosen

    def local_search_improve(pairs, rng, max_iters=20000):
        """Local search: try to add more moduli by swapping residues or removing+re-adding."""
        best = list(pairs)
        best_len = len(best)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            current = list(best)
            used = {m for _, m in current}
            candidates = [m for m in all_moduli if m not in used]
            if not candidates:
                break
            
            # Strategy 1: try to add a candidate directly
            rng.shuffle(candidates)
            added_any = False
            for m_new in candidates[:15]:
                r_new = find_valid_residue(m_new, current, rng)
                if r_new is not None:
                    current.append((r_new, m_new))
                    added_any = True
                    if len(current) > best_len:
                        best = list(current)
                        best_len = len(best)
                    break
            
            # Strategy 2: swap a residue of an existing pair to free space
            if not added_any and current:
                idx = rng.randint(0, len(current) - 1)
                r_old, m_old = current[idx]
                # Try a few alternative residues for this modulus
                for _ in range(10):
                    r_new = rng.randint(0, m_old - 1)
                    if r_new == r_old:
                        continue
                    # Check compatibility with all others
                    ok = True
                    for i, (r2, m2) in enumerate(current):
                        if i == idx:
                            continue
                        if not compatible(r_new, m_old, r2, m2):
                            ok = False
                            break
                    if ok:
                        current[idx] = (r_new, m_old)
                        # Now try to add a candidate
                        used2 = {m for _, m in current}
                        cands2 = [m for m in all_moduli if m not in used2]
                        if cands2:
                            rng.shuffle(cands2)
                            for m_add in cands2[:10]:
                                r_add = find_valid_residue(m_add, current, rng)
                                if r_add is not None:
                                    current.append((r_add, m_add))
                                    if len(current) > best_len:
                                        best = list(current)
                                        best_len = len(best)
                                    break
                        break  # only try one swap per iteration
        
        return best

    def backtrack_csp(order, deadline, max_branch=6):
        """CSP backtracking with forward checking and dynamic variable ordering.
        Process moduli in order; at each step, pick a residue that doesn't conflict.
        Use branching factor limit to keep it tractable."""
        best = []
        
        def bt(chosen, remaining):
            nonlocal best
            if time.time() > deadline:
                return
            if len(chosen) > len(best):
                best = list(chosen)
            if not remaining:
                return
            
            # Dynamic ordering: pick the most constrained variable first
            # (the one with fewest valid residues given current chosen)
            best_idx = -1
            best_residues = None
            min_count = float('inf')
            
            for i, m in enumerate(remaining):
                # Count valid residues quickly (sampled)
                constraints = []
                for r_k, m_k in chosen:
                    g = gcd(m, m_k)
                    if g > 1:
                        constraints.append((g, r_k % g))
                if not constraints:
                    count = m
                else:
                    # Estimate: for each constraint (g, fk), fraction forbidden = 1/g
                    # Combined forbidden fraction approx 1 - prod(1 - 1/g)
                    # But this is rough. Just sample.
                    count = 0
                    sample_size = min(m, 50)
                    for _ in range(sample_size):
                        r = random.randint(0, m - 1)
                        ok = True
                        for g, fk in constraints:
                            if r % g == fk:
                                ok = False
                                break
                        if ok:
                            count += 1
                    count = int(m * count / sample_size)
                
                if count < min_count:
                    min_count = count
                    best_idx = i
                    if count == 0:
                        break
            
            if best_idx == -1:
                return
            if min_count == 0:
                return
            
            m = remaining[best_idx]
            remaining_new = remaining[:best_idx] + remaining[best_idx+1:]
            
            # Find valid residues for this m
            valid_rs = []
            constraints = []
            for r_k, m_k in chosen:
                g = gcd(m, m_k)
                if g > 1:
                    constraints.append((g, r_k % g))
            
            if not constraints:
                # All residues valid; pick a few
                valid_rs = [0, m - 1] + [random.randint(0, m-1) for _ in range(min(max_branch - 2, m - 2))]
            else:
                # Find up to max_branch valid residues
                if m <= 64:
                    for r in range(m):
                        ok = True
                        for g, fk in constraints:
                            if r % g == fk:
                                ok = False
                                break
                        if ok:
                            valid_rs.append(r)
                            if len(valid_rs) >= max_branch:
                                break
                else:
                    # Sample
                    for _ in range(min(m, 200)):
                        r = random.randint(0, m - 1)
                        ok = True
                        for g, fk in constraints:
                            if r % g == fk:
                                ok = False
                                break
                        if ok and r not in valid_rs:
                            valid_rs.append(r)
                            if len(valid_rs) >= max_branch:
                                break
            
            for r in valid_rs:
                chosen.append((r, m))
                bt(chosen, remaining_new)
                chosen.pop()
                if time.time() > deadline:
                    return
            # Also try skipping this modulus
            bt(chosen, remaining_new)
        
        bt([], list(order))
        return best

    # Main search
    rng = random.Random(12345)
    best_solution = []
    best_len = 0

    # Phase 1: Multi-strategy greedy
    primes = [m for m in all_moduli if all(m % i != 0 for i in range(2, int(math.isqrt(m)) + 1))]
    composites = [m for m in all_moduli if m not in primes]
    
    orders = [
        sorted(all_moduli, key=lambda m: -pf_counts[m]),  # most prime factors first
        sorted(all_moduli, key=lambda m: pf_counts[m]),   # fewest first
        sorted(all_moduli),
        sorted(all_moduli, reverse=True),
        primes + sorted(composites, key=lambda m: -pf_counts[m]),
        sorted(composites, key=lambda m: -pf_counts[m]) + primes,
    ]
    
    # Add some random orders
    for _ in range(20):
        shuffled = all_moduli.copy()
        rng.shuffle(shuffled)
        orders.append(shuffled)

    for order in orders:
        if time.time() > deadline:
            break
        sol = greedy_build(order, rng)
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol

    # Phase 2: Local search improvement on best greedy solution
    if best_solution:
        improved = local_search_improve(best_solution, rng, max_iters=15000)
        if len(improved) > best_len:
            best_len = len(improved)
            best_solution = improved

    # Phase 3: CSP backtracking with forward checking
    # Use the most-constrained-first ordering
    csp_order = sorted(all_moduli, key=lambda m: -pf_counts[m])
    csp_sol = backtrack_csp(csp_order, deadline, max_branch=4)
    if len(csp_sol) > best_len:
        best_len = len(csp_sol)
        best_solution = csp_sol

    # Phase 4: More local search if time remains
    if best_solution and time.time() < deadline - 5:
        improved2 = local_search_improve(best_solution, rng, max_iters=10000)
        if len(improved2) > best_len:
            best_len = len(improved2)
            best_solution = improved2

    # Ensure nonempty
    if not best_solution:
        best_solution = [(0, 2)]

    return [[r, m] for r, m in best_solution]
