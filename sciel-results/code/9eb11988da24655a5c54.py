import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    max_mod = 128
    all_moduli = list(range(2, max_mod + 1))
    
    best_count = 0
    best_solution = []
    
    # Precompute gcds for speed
    gcd_table = [[0] * (max_mod + 1) for _ in range(max_mod + 1)]
    for i in range(2, max_mod + 1):
        for j in range(i, max_mod + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def get_gcd(m1, m2):
        return gcd_table[m1][m2]
    
    def is_compatible(r_new, m_new, selected):
        """Check if residue r_new mod m_new is disjoint from all in selected."""
        for r_old, m_old in selected:
            g = get_gcd(m_new, m_old)
            if g == 1:
                return False  # Always intersect if gcd=1
            if (r_new - r_old) % g == 0:
                return False
        return True
    
    def valid_residues(m, selected):
        """Return list of residues r in [0,m) that are disjoint from all selected."""
        out = []
        for r in range(m):
            ok = True
            for r_k, m_k in selected:
                g = get_gcd(m, m_k)
                if g == 1:
                    ok = False
                    break
                if (r - r_k) % g == 0:
                    ok = False
                    break
            if ok:
                out.append(r)
        return out
    
    def count_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    div_counts = {m: count_divisors(m) for m in all_moduli}
    
    # Strategy 1: Single-prime constructions (strong baseline)
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
        if time.time() > deadline:
            break
        multiples = [m for m in all_moduli if m % p == 0]
        k = min(len(multiples), p)
        sol = [[i % p, multiples[i]] for i in range(k)]
        if len(sol) > best_count:
            best_count = len(sol)
            best_solution = [list(x) for x in sol]
    
    # Strategy 2: Greedy with most-constrained-first heuristic
    def most_constrained_greedy(order):
        remaining = list(order)
        selected = []
        while remaining:
            if time.time() > deadline:
                break
            best_m = None
            best_vr = None
            best_count_local = None
            for m in remaining:
                vr = valid_residues(m, selected)
                c = len(vr)
                if c == 0:
                    continue
                if best_count_local is None or c < best_count_local:
                    best_count_local = c
                    best_m = m
                    best_vr = vr
                    if c == 1:
                        break
            if best_m is None:
                break
            r = best_vr[0]
            selected.append((r, best_m))
            remaining.remove(best_m)
        return [[r, m] for r, m in selected]
    
    # Strategy 3: Randomized greedy with multiple orderings
    def randomized_greedy(order):
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            vr = valid_residues(m, selected)
            if not vr:
                continue
            r = random.choice(vr)
            selected.append((r, m))
        return [[r, m] for r, m in selected]
    
    # Run most-constrained greedy with different orderings
    orders = [
        all_moduli[:],
        all_moduli[::-1],
        sorted(all_moduli, key=lambda x: -div_counts[x]),
        sorted(all_moduli, key=lambda x: div_counts[x]),
        sorted(all_moduli, key=lambda x: (-div_counts[x], x)),
        sorted(all_moduli, key=lambda x: (div_counts[x], -x)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        sol = most_constrained_greedy(order[:])
        if len(sol) > best_count:
            best_count = len(sol)
            best_solution = [list(x) for x in sol]
    
    # Strategy 4: Local search to improve current best
    def local_search(sol, max_rounds=100):
        sol = [list(x) for x in sol]
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            # Try adding any unselected modulus
            selected_set = set(m for _, m in sol)
            for m in all_moduli:
                if m in selected_set:
                    continue
                vr = valid_residues(m, sol)
                if vr:
                    sol.append([vr[0], m])
                    improved = True
                    break
            if improved:
                continue
            # Try 1-for-1 swap: remove one, add up to two
            for i in range(len(sol)):
                if time.time() > deadline:
                    break
                base = sol[:i] + sol[i+1:]
                selected_set = set(m for _, m in base)
                added = 0
                for m in all_moduli:
                    if m in selected_set:
                        continue
                    vr = valid_residues(m, base)
                    if vr:
                        base.append([vr[0], m])
                        selected_set.add(m)
                        added += 1
                if len(base) > len(sol):
                    sol = base
                    improved = True
                    break
            if not improved:
                break
        return sol
    
    # Apply local search to best solution found so far
    if best_solution:
        improved = local_search(best_solution)
        if len(improved) > best_count:
            best_count = len(improved)
            best_solution = improved
    
    # Strategy 5: Multiple random restarts with mixed strategies
    restart_count = 0
    while time.time() < deadline - 5.0:
        restart_count += 1
        strategy = restart_count % 4
        
        if strategy == 0:
            order = all_moduli[:]
            random.shuffle(order)
            sol = randomized_greedy(order)
        elif strategy == 1:
            order = sorted(all_moduli, key=lambda x: random.random())
            sol = randomized_greedy(order)
        elif strategy == 2:
            order = all_moduli[:]
            random.shuffle(order)
            sol = most_constrained_greedy(order)
        else:
            # Start from single-prime solution and try to extend
            p = random.choice([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31])
            multiples = [m for m in all_moduli if m % p == 0]
            k = min(len(multiples), p)
            sol = [[i % p, multiples[i]] for i in range(k)]
            # Try to add more
            selected_set = set(m for _, m in sol)
            for m in all_moduli:
                if m in selected_set:
                    continue
                vr = valid_residues(m, sol)
                if vr:
                    sol.append([vr[0], m])
                    selected_set.add(m)
        
        if len(sol) > best_count:
            best_count = len(sol)
            best_solution = [list(x) for x in sol]
        
        # Apply local search to this solution
        improved = local_search(sol, max_rounds=50)
        if len(improved) > best_count:
            best_count = len(improved)
            best_solution = improved
    
    if not best_solution:
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    return best_solution
