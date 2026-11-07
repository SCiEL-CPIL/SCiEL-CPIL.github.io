import math
import time
import random

def construct():
    start = time.time()
    deadline = start + 80  # safety margin under 90s

    N = 256

    # Sieve for primes up to N
    def sieve(limit):
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = False
        return [i for i in range(2, limit + 1) if is_p[i]]

    primes = sieve(N)
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    # Distinct prime factors for each number 1..N
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

    # Base solution: 1 + all primes (always valid, good fallback)
    base_numbers = [1] + list(primes)
    base_sum = sum(base_numbers)

    def time_left():
        return deadline - time.time()

    # ---- Exact MILP solve (weighted set packing) ----
    # vars: x[n] for n in 1..N  (index n-1)
    # constraint per prime p: sum_{n: p in F[n]} x[n] <= 1
    # objective: maximize sum n*x[n]
    def solve_milp():
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
        except Exception:
            return None

        n_vars = N  # n in 1..N
        # objective c (minimize -sum n*x[n])
        c = [-(n) for n in range(1, N + 1)]

        # Build constraint matrix A (num_primes x n_vars)
        A = []
        for p in primes:
            row = [0.0] * n_vars
            for n in range(1, N + 1):
                if p in F[n]:
                    row[n - 1] = 1.0
            A.append(row)

        A = [A]  # list of matrices
        lb = [0.0] * n_vars
        ub = [1.0] * n_vars
        # per-prime upper bound 1
        constr = LinearConstraint(A, lb=[0.0] * num_primes, ub=[1.0] * num_primes)
        integrality = [1] * n_vars
        bounds = Bounds(lb, ub)

        try:
            res = milp(c=c, constraints=constr, integrality=integrality, bounds=bounds)
        except Exception:
            return None

        if res.status != 0:
            return None

        x = res.x
        selected = []
        for n in range(1, N + 1):
            if x[n - 1] is not None and x[n - 1] > 0.5:
                selected.append(n)
        return selected

    result = None
    if time_left() > 30:
        result = solve_milp()

    if result is None:
        result = base_numbers[:]
    else:
        result = [int(x) for x in result]

    # ---- Local search polish ----
    def verify(lst):
        s = set(lst)
        if len(s) != len(lst):
            return False
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    def try_improve(sol):
        sol_set = set(sol)
        sol_list = list(sol)
        cur_sum = sum(sol_list)
        improved = True
        while improved and time_left() > 15:
            improved = False

            # 1) Try to ADD a number not in sol that is coprime to all in sol
            #    (only possible if it uses primes not already used)
            used_primes = set()
            for n in sol_list:
                used_primes |= F[n]
            for n in range(N, 1, -1):
                if n in sol_set:
                    continue
                if F[n] & used_primes:
                    continue
                # n uses only unused primes -> coprime to all
                sol_set.add(n)
                sol_list.append(n)
                cur_sum += n
                used_primes |= F[n]
                improved = True
                break

            if not improved and time_left() > 15:
                # 2) Swap: remove a, add b (b not in sol), b coprime to all except a, b>a
                for a in sol_list:
                    others = [y for y in sol_list if y != a]
                    for b in range(N, a, -1):
                        if b in sol_set:
                            continue
                        ok = True
                        for y in others:
                            if math.gcd(b, y) != 1:
                                ok = False
                                break
                        if ok:
                            sol_set.remove(a)
                            sol_set.add(b)
                            sol_list.remove(a)
                            sol_list.append(b)
                            cur_sum += (b - a)
                            improved = True
                            break
                    if improved:
                        break

        return sol_list

    if time_left() > 15:
        candidate = try_improve(result)
        if verify(candidate) and sum(candidate) >= sum(result):
            result = candidate

    # Final verification; fallback if broken
    if not verify(result):
        result = base_numbers[:]

    # Ensure pure Python ints, list type
    result = [int(x) for x in result]
    return result
