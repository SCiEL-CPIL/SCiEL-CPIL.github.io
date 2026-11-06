import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MOD_MAX = 128
    moduli = list(range(2, MOD_MAX + 1))

    # Precompute gcd table
    gcd_table = [[0] * (MOD_MAX + 1) for _ in range(MOD_MAX + 1)]
    for i in range(2, MOD_MAX + 1):
        for j in range(i, MOD_MAX + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def disjoint(r1, m1, r2, m2):
        g = gcd_table[m1][m2]
        if g <= 1:
            return False
        return (r1 % g) != (r2 % g)

    def is_valid_solution(sol):
        # sol is list of [r, m]
        for i in range(len(sol)):
            for j in range(i + 1, len(sol)):
                r1, m1 = sol[i]
                r2, m2 = sol[j]
                if not disjoint(r1, m1, r2, m2):
                    return False
        # Check distinct moduli
        mods = [m for _, m in sol]
        if len(mods) != len(set(mods)):
            return False
        # Check canonical
        for r, m in sol:
            if not (0 <= r < m):
                return False
        return True

    # Generate best star solution
    best_star = []
    for p in range(2, MOD_MAX + 1):
        # Check if p is prime? Not strictly necessary, but primes are best for stars
        # Actually, any g works.
        multiples = [m for m in moduli if m % p == 0]
        if len(multiples) < p:
            continue
        # We can pick p multiples.
        # Assign residues 0, 1, ..., p-1
        # Sort multiples to be deterministic
        multiples.sort()
        sol = [[i, multiples[i]] for i in range(p)]
        if is_valid_solution(sol) and len(sol) > len(best_star):
            best_star = sol

    # best_star should be size 11 (p=11)
    best = [list(x) for x in best_star]

    # Local search to try to improve
    def local_search(sol, max_rounds=500):
        sol = [list(x) for x in sol]
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            # Try adding a modulus
            selected_mods = set(m for _, m in sol)
            for m in moduli:
                if m in selected_mods:
                    continue
                # Check if m can be added with some residue
                found = False
                for r in range(m):
                    ok = True
                    for r_k, m_k in sol:
                        if not disjoint(r, m, r_k, m_k):
                            ok = False
                            break
                    if ok:
                        sol.append([r, m])
                        improved = True
                        found = True
                        break
                if found:
                    break
            
            if not improved:
                # Try swapping
                for i in range(len(sol)):
                    if time.time() > deadline:
                        break
                    r_old, m_old = sol[i]
                    # Try replacing m_old with a new m_new
                    # And adjusting residue
                    # This is complex, skip for now if star is strong
                    
            if not improved:
                break
        return sol

    # Run local search on best star
    best = local_search(best, max_rounds=200)

    # If we have time, try random restarts
    import random
    for _ in range(100):
        if time.time() > deadline - 5.0:
            break
        # Random greedy
        random.shuffle(moduli)
        sol = []
        for m in moduli:
            # Find a valid residue
            found = False
            for r in range(m):
                ok = True
                for r_k, m_k in sol:
                    if not disjoint(r, m, r_k, m_k):
                        ok = False
                        break
                if ok:
                    sol.append([r, m])
                    found = True
                    break
        if len(sol) > len(best):
            best = sol
        # Local search on random
        sol_ls = local_search(sol, max_rounds=50)
        if len(sol_ls) > len(best):
            best = sol_ls

    return best
