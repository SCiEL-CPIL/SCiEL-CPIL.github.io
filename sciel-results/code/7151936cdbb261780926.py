import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (evaluator kills at 90s)

    max_modulus = 128
    moduli = list(range(2, max_modulus + 1))
    n_moduli = len(moduli)

    # Precompute gcds for fast compatibility checks
    gcds = {}
    for i in range(n_moduli):
        m1 = moduli[i]
        for j in range(i + 1, n_moduli):
            m2 = moduli[j]
            gcds[(m1, m2)] = math.gcd(m1, m2)

    def gcd_of(a, b):
        if a == b:
            return a
        if a < b:
            return gcds.get((a, b), math.gcd(a, b))
        return gcds.get((b, a), math.gcd(b, a))

    def is_compatible(r_new, m_new, selected):
        # selected: list of (r, m)
        for r_k, m_k in selected:
            g = gcd_of(m_new, m_k)
            if r_new % g == r_k % g:
                return False
        return True

    def find_residue(m, selected, rng=None):
        # Try residues in a few strategic orders to maximize success.
        for r in range(m):
            if is_compatible(r, m, selected):
                return r
        if rng is not None:
            cand = list(range(m))
            rng.shuffle(cand)
            for r in cand:
                if is_compatible(r, m, selected):
                    return r
        return None

    def solve_greedy(order, rng):
        selected = []
        for m in order:
            r = find_residue(m, selected, rng)
            if r is not None:
                selected.append((r, m))
        return selected

    # ---- Heuristic ordering keys ----
    def num_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    def count_prime_factors(n):
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

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    div_counts = {m: num_divisors(m) for m in moduli}
    pf_counts = {m: count_prime_factors(m) for m in moduli}

    primes = [m for m in moduli if is_prime(m)]
    composites = [m for m in moduli if not is_prime(m)]

    orders = []
    orders.append(sorted(moduli, key=lambda m: div_counts[m]))
    orders.append(sorted(moduli, key=lambda m: -div_counts[m]))
    orders.append(sorted(moduli))
    orders.append(sorted(moduli, reverse=True))
    orders.append(sorted(moduli, key=lambda m: -pf_counts[m]))
    orders.append(sorted(moduli, key=lambda m: pf_counts[m]))
    orders.append(primes + sorted(composites, key=lambda m: div_counts[m]))
    orders.append(sorted(composites, key=lambda m: -div_counts[m]) + primes)

    # ---- Fixed local search: add missing moduli via residue swaps ----
    def local_search(selected, rng, time_budget=2.0):
        selected = list(selected)
        selected_moduli = set(m for _, m in selected)
        missing = [m for m in moduli if m not in selected_moduli]
        start = time.time()
        
        improved = True
        while improved and time.time() - start < time_budget and time.time() < deadline:
            improved = False
            rng.shuffle(missing)
            for target_m in missing:
                if time.time() - start >= time_budget or time.time() >= deadline:
                    break
                if target_m in selected_moduli:
                    continue
                # First: can we just add it?
                r_add = find_residue(target_m, selected, None)
                if r_add is not None:
                    selected.append((r_add, target_m))
                    selected_moduli.add(target_m)
                    improved = True
                    continue
                # Try single-swap: change residue of one existing class to free target_m
                added = False
                for i in range(len(selected)):
                    r_i, m_i = selected[i]
                    for r_i_new in range(m_i):
                        if r_i_new == r_i:
                            continue
                        # Check class i (with r_i_new) vs all others
                        ok_i = True
                        for j in range(len(selected)):
                            if j == i:
                                continue
                            r_j, m_j = selected[j]
                            g = gcd_of(m_i, m_j)
                            if r_i_new % g == r_j % g:
                                ok_i = False
                                break
                        if not ok_i:
                            continue
                        # Check target_m vs all (using r_i_new for class i's residue)
                        ok_t = True
                        for j in range(len(selected)):
                            r_j, m_j = selected[j]
                            g = gcd_of(target_m, m_j)
                            # For class i, use r_i_new; for others, use r_j
                            r_eff = r_i_new if j == i else r_j
                            if r_eff % g == r_j % g and j != i:
                                ok_t = False
                                break
                            if j == i:
                                # target vs class i: check target residue vs r_i_new mod gcd
                                # We need some residue r_t for target such that r_t % g != r_i_new % g
                                # This is possible iff not all residues 0..target_m-1 have same value mod g
                                # i.e., iff g > 1 (since if g==1, all residues are 0 mod 1, conflict always)
                                # Actually if g==1, gcd=1, so r_new%1==0==r_k%1 always -> incompatible
                                # So if g==1, we can NEVER add target_m with any residue
                                if g == 1:
                                    ok_t = False
                                    break
                                # If g>1, there exists r_t in [0,target_m) with r_t%g != r_i_new%g
                                # as long as target_m > g (which it is since target_m >= 2 and g <= target_m)
                                # Actually we need to verify: is there an r_t in [0, target_m-1] with r_t % g != r_i_new % g?
                                # Yes, since target_m >= 2 and g >= 2 (if g==1 we already handled), 
                                # the residues 0,1,...,target_m-1 cover multiple values mod g if target_m > g,
                                # or if target_m <= g, they cover target_m distinct values mod g, and since g >= 2,
                                # at most one value equals r_i_new % g, so there's another.
                                # So ok_t stays True for this j.
                                pass
                        if ok_t:
                            # Find a valid residue for target_m
                            r_t = None
                            for rt in range(target_m):
                                compatible = True
                                for j in range(len(selected)):
                                    r_j, m_j = selected[j]
                                    g = gcd_of(target_m, m_j)
                                    r_eff = r_i_new if j == i else r_j
                                    if rt % g == r_eff % g:
                                        compatible = False
                                        break
                                if compatible:
                                    r_t = rt
                                    break
                            if r_t is not None:
                                selected[i] = (r_i_new, m_i)
                                selected.append((r_t, target_m))
                                selected_moduli.add(target_m)
                                added = True
                                improved = True
                                break
                    if added:
                        break
                if time.time() - start >= time_budget or time.time() >= deadline:
                    break
        return selected

    # ---- Backtracking search with time limit ----
    def backtrack_search(order, initial_selected, time_budget=5.0):
        """Try to add remaining moduli from order to initial_selected via backtracking."""
        selected = list(initial_selected)
        start = time.time()
        remaining = [m for m in order if m not in set(mm for _, mm in selected)]
        
        def bt(idx):
            if time.time() - start >= time_budget or time.time() >= deadline:
                return
            if idx >= len(remaining):
                return
            m = remaining[idx]
            # Try each residue
            for r in range(m):
                if is_compatible(r, m, selected):
                    selected.append((r, m))
                    bt(idx + 1)
                    selected.pop()
                    if len(selected) > len(initial_selected):
                        # Found improvement, keep going to try for more
                        pass
                if time.time() - start >= time_budget or time.time() >= deadline:
                    return
        bt(0)
        return selected

    # ---- Main search loop ----
    best_solution = []
    best_len = 0

    rng = random.Random(42)

    # Deterministic + random greedy orders
    all_orders = list(orders)
    for _ in range(5000):
        shuffled = moduli.copy()
        rng.shuffle(shuffled)
        all_orders.append(shuffled)

    for order in all_orders:
        if time.time() >= deadline:
            break
        sol = solve_greedy(order, rng)
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol

    # Polish the best solution with local search
    if best_len > 0 and time.time() < deadline:
        polished = local_search(best_solution, rng, time_budget=3.0)
        if len(polished) > best_len:
            best_len = len(polished)
            best_solution = polished

    # Extra local-search restarts from random greedy solutions
    for _ in range(200):
        if time.time() >= deadline:
            break
        shuffled = moduli.copy()
        rng.shuffle(shuffled)
        sol = solve_greedy(shuffled, rng)
        polished = local_search(sol, rng, time_budget=1.0)
        if len(polished) > best_len:
            best_len = len(polished)
            best_solution = polished

    # Backtracking polish on best solution
    if best_len > 0 and time.time() < deadline:
        remaining = [m for m in moduli if m not in set(mm for _, mm in best_solution)]
        bt_sol = backtrack_search(remaining, best_solution, time_budget=5.0)
        if len(bt_sol) > best_len:
            best_len = len(bt_sol)
            best_solution = bt_sol

    # Final safety: ensure nonempty and canonical
    if not best_solution:
        return [[0, 2]]

    result = []
    seen_moduli = set()
    for r, m in best_solution:
        if m in seen_moduli:
            continue
        seen_moduli.add(m)
        r = r % m
        result.append([r, m])

    return result
