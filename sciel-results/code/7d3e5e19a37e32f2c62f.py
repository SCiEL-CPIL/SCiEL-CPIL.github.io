import math
import time

def construct():
    start = time.time()
    deadline = start + 85  # safety margin under 90s

    N = 256

    # Sieve for primes
    def sieve(limit):
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = False
        return [i for i in range(2, limit + 1) if is_p[i]]

    primes = sieve(N)
    prime_set = set(primes)
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    # Prime powers: largest p^k <= N for each prime p
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power

    # Precompute prime factorization (as frozenset of distinct primes) for all numbers 1..N
    F = [None] * (N + 1)
    F[1] = frozenset()
    for i in range(2, N + 1):
        x = i
        factors = set()
        d = 2
        while d * d <= x:
            if x % d == 0:
                factors.add(d)
                while x % d == 0:
                    x //= d
            else:
                d += 1
        if x > 1:
            factors.add(x)
        F[i] = frozenset(factors)

    # Precompute coprime table
    coprime = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            c = math.gcd(i, j) == 1
            coprime[i][j] = c
            coprime[j][i] = c

    # Base solution: 1 + all prime powers
    base_numbers = [1] + [prime_powers[p] for p in primes]
    base_sum = sum(base_numbers)

    # Generate candidate moves
    moves = []
    
    def get_mask(ps):
        mask = 0
        for p in ps:
            mask |= (1 << prime_index[p])
        return mask

    best_c_for_ps = {}
    
    for c in range(2, N + 1):
        if c in prime_set:
            continue
        ps = F[c]
        if len(ps) < 2:
            continue
        if ps not in best_c_for_ps or c > best_c_for_ps[ps]:
            best_c_for_ps[ps] = c

    for ps, c in best_c_for_ps.items():
        base_cost = sum(prime_powers[p] for p in ps)
        gain = c - base_cost
        if gain > 0:
            mask = get_mask(ps)
            moves.append((mask, c, gain, ps))

    moves.sort(key=lambda x: -x[2])
    
    n_moves = len(moves)
    suffix_gains = [0] * (n_moves + 1)
    for i in range(n_moves - 1, -1, -1):
        suffix_gains[i] = suffix_gains[i+1] + moves[i][2]

    best_total_gain = 0
    best_selected_moves = []
    
    def solve(idx, used_mask, current_gain, selected):
        nonlocal best_total_gain, best_selected_moves
        
        if time.time() > deadline - 2:
            return

        if current_gain + suffix_gains[idx] <= best_total_gain:
            return

        if idx == n_moves:
            if current_gain > best_total_gain:
                best_total_gain = current_gain
                best_selected_moves = selected[:]
            return

        solve(idx + 1, used_mask, current_gain, selected)

        mask, c, gain, ps = moves[idx]
        if mask & used_mask == 0:
            selected.append((mask, c, gain, ps))
            solve(idx + 1, used_mask | mask, current_gain + gain, selected)
            selected.pop()

    solve(0, 0, 0, [])

    final_sol = set(base_numbers)
    
    for mask, c, gain, ps in best_selected_moves:
        for p in ps:
            final_sol.remove(prime_powers[p])
        
    for mask, c, gain, ps in best_selected_moves:
        final_sol.add(c)
        
    result = [int(x) for x in final_sol]
    
    def verify(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    if not verify(result):
        result = [int(x) for x in base_numbers]

    # Local search with precomputed coprime table
    def try_improve(sol):
        sol_set = set(sol)
        sol_list = list(sol)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for num in sol_list[:]:
                others_mask = 0
                for y in sol_list:
                    if y != num:
                        others_mask |= (1 << (y - 1))
                
                for x in range(N, num, -1):
                    if x in sol_set:
                        continue
                    ok = True
                    for y in sol_list:
                        if y != num and not coprime[x][y]:
                            ok = False
                            break
                    if ok:
                        sol_set.remove(num)
                        sol_set.add(x)
                        sol_list.remove(num)
                        sol_list.append(x)
                        improved = True
                        break
        return sol_list

    result = try_improve(result)
    
    # Try pure additions: add numbers not in solution that are coprime to all
    def try_additions(sol):
        sol_set = set(sol)
        sol_list = list(sol)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in range(N, 1, -1):
                if x in sol_set:
                    continue
                ok = True
                for y in sol_list:
                    if not coprime[x][y]:
                        ok = False
                        break
                if ok:
                    sol_set.add(x)
                    sol_list.append(x)
                    improved = True
                    break
        return sol_list

    result = try_additions(result)
    
    # Final local search pass
    result = try_improve(result)
    
    if not verify(result):
        final_sol = set(base_numbers)
        for mask, c, gain, ps in best_selected_moves:
            for p in ps:
                final_sol.remove(prime_powers[p])
        for mask, c, gain, ps in best_selected_moves:
            final_sol.add(c)
        result = [int(x) for x in final_sol]
    
    return result
