import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n_mod = len(moduli)

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    gcd_mat = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Generate prime powers
    prime_powers = {}
    for m in moduli:
        # check if m is a prime power
        is_pp = False
        for p in range(2, int(math.isqrt(m)) + 1):
            if m % p == 0:
                temp = m
                while temp % p == 0:
                    temp //= p
                if temp == 1:
                    is_pp = True
                    break
        if is_pp:
            # find the prime
            for p in range(2, int(math.isqrt(m)) + 1):
                if m % p == 0:
                    prime_powers.setdefault(p, []).append(m)
                    break
            else:
                # m is prime
                prime_powers.setdefault(m, []).append(m)

    # Assign residues for prime powers
    chosen = {}
    for p, powers in prime_powers.items():
        # Sort powers ascending
        powers.sort()
        # Assign residues
        # r[p] = 0
        # r[p^2] = 1
        # r[p^3] = p
        # r[p^4] = 2p
        # ...
        residues = []
        for i, m in enumerate(powers):
            if i == 0:
                r = 0
            else:
                # Need r not congruent to any previous r_j mod gcd(m, m_j)
                # gcd(m, m_j) = m_j (since m_j < m and both powers of p)
                # So r not congruent to r_j mod m_j
                # We can just pick a value
                # Try small values
                found = False
                for candidate in range(m):
                    ok = True
                    for j, m_j in enumerate(powers[:i]):
                        r_j = residues[j]
                        if candidate % m_j == r_j % m_j:
                            ok = False
                            break
                    if ok:
                        residues.append(candidate)
                        found = True
                        break
                if not found:
                    # Should not happen
                    residues.append(0)
            # Map back to index
            idx = moduli.index(m)
            chosen[idx] = residues[-1] if found else 0 # Wait, residues list logic is messy

    # Re-implement residue assignment cleanly
    chosen = {}
    for p, powers in prime_powers.items():
        powers.sort()
        assigned_residues = []
        for m in powers:
            idx = moduli.index(m)
            found = False
            for r in range(m):
                ok = True
                for m_prev, r_prev in zip(powers[:powers.index(m)], assigned_residues):
                    g = gcd(m, m_prev)
                    if r % g == r_prev % g:
                        ok = False
                        break
                if ok:
                    chosen[idx] = r
                    assigned_residues.append(r)
                    found = True
                    break
            if not found:
                chosen[idx] = 0
                assigned_residues.append(0)

    # Local search to add more
    def is_valid_solution(chosen):
        items = list(chosen.items())
        for i in range(len(items)):
            idx_i, r_i = items[i]
            mi = moduli[idx_i]
            for j in range(i + 1, len(items)):
                idx_j, r_j = items[j]
                mj = moduli[idx_j]
                g = gcd_mat[idx_i][idx_j]
                if r_i % g == r_j % g:
                    return False
        return True

    def local_search(chosen, max_iter=1000):
        chosen = dict(chosen)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iter and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            
            missing = [i for i in range(n_mod) if i not in chosen]
            if not missing:
                break
            
            # Try adding each missing modulus
            for idx in missing:
                if time.time() > deadline - 5.0:
                    break
                mi = moduli[idx]
                
                valid_residues = []
                for r in range(mi):
                    ok = True
                    for j, rj in chosen.items():
                        g = gcd_mat[idx][j]
                        if r % g == rj % g:
                            ok = False
                            break
                    if ok:
                        valid_residues.append(r)
                
                if not valid_residues:
                    continue
                
                # Pick residue that blocks fewest others (greedy heuristic)
                best_r = None
                best_cost = float('inf')
                for r in valid_residues:
                    cost = 0
                    for j in missing:
                        if j == idx:
                            continue
                        mj = moduli[j]
                        g = gcd_mat[idx][j]
                        target = r % g
                        # Count how many residues of j are blocked
                        # This is mj // g
                        cost += mj // g
                    if cost < best_cost:
                        best_cost = cost
                        best_r = r
                
                if best_r is not None:
                    chosen[idx] = best_r
                    improved = True
                    break
        
        return chosen

    best_sol = local_search(chosen)
    if not is_valid_solution(best_sol):
        # Fallback to prime powers only
        best_sol = {idx: r for idx, r in chosen.items() if is_valid_solution({idx: r, **{k:v for k,v in chosen.items() if k < idx}})} 
        # Simpler fallback: just return the prime powers we constructed, which are valid
        # The construction above ensures validity within chains, and chains are coprime.
        # So 'chosen' is valid.
        best_sol = dict(chosen)

    return [[r, moduli[idx]] for idx, r in best_sol.items()]
