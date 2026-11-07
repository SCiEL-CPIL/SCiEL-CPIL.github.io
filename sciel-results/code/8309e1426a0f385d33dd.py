import math
import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin for 90s limit

    N = 256

    # Precompute prime factors for each number 1..N
    prime_factors = [set() for _ in range(N + 1)]
    prime_factors[1] = set()
    for i in range(2, N + 1):
        x = i
        d = 2
        while d * d <= x:
            if x % d == 0:
                prime_factors[i].add(d)
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            prime_factors[i].add(x)

    def is_coprime_to_set(x, S):
        if x == 1:
            return True
        pf_x = prime_factors[x]
        for y in S:
            if y == 1:
                continue
            if pf_x & prime_factors[y]:
                return False
        return True

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(N)
    all_numbers = list(range(1, N + 1))

    # Product move: replace subset of primes {p1,...,pk} with their product P
    # if P <= N, P is coprime to S\subset, and P > sum(subset)
    def try_improve_product_moves(S, s_sum):
        prims_in_set = [p for p in S if p in primes]
        n = len(prims_in_set)

        # Size 2
        for i in range(n):
            for j in range(i+1, n):
                p1, p2 = prims_in_set[i], prims_in_set[j]
                prod = p1 * p2
                if prod > N:
                    continue
                rest = S - {p1, p2}
                if is_coprime_to_set(prod, rest):
                    gain = prod - (p1 + p2)
                    if gain > 0:
                        S.discard(p1)
                        S.discard(p2)
                        S.add(prod)
                        return True, s_sum + gain
        # Size 3
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = prims_in_set[i], prims_in_set[j], prims_in_set[k]
                    prod = p1 * p2 * p3
                    if prod > N:
                        continue
                    rest = S - {p1, p2, p3}
                    if is_coprime_to_set(prod, rest):
                        gain = prod - (p1 + p2 + p3)
                        if gain > 0:
                            S.discard(p1)
                            S.discard(p2)
                            S.discard(p3)
                            S.add(prod)
                            return True, s_sum + gain
        # Size 4
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1, p2, p3, p4 = prims_in_set[i], prims_in_set[j], prims_in_set[k], prims_in_set[l]
                        prod = p1 * p2 * p3 * p4
                        if prod > N:
                            continue
                        rest = S - {p1, p2, p3, p4}
                        if is_coprime_to_set(prod, rest):
                            gain = prod - (p1 + p2 + p3 + p4)
                            if gain > 0:
                                S.discard(p1)
                                S.discard(p2)
                                S.discard(p3)
                                S.discard(p4)
                                S.add(prod)
                                return True, s_sum + gain
        return False, s_sum

    def try_add(S, s_sum, x):
        if x in S:
            return S, s_sum, False
        if is_coprime_to_set(x, S):
            S.add(x)
            return S, s_sum + x, True
        return S, s_sum, False

    def try_swap(S, s_sum, out_x, in_y):
        if out_x not in S or in_y in S or in_y == out_x:
            return S, s_sum, False
        temp_set = S - {out_x}
        if is_coprime_to_set(in_y, temp_set):
            S.discard(out_x)
            S.add(in_y)
            return S, s_sum - out_x + in_y, True
        return S, s_sum, False

    def local_search(S, s_sum, max_time):
        # Phase A: Product moves
        while time.time() - start_time < max_time:
            improved, s_sum = try_improve_product_moves(S, s_sum)
            if not improved:
                break

        # Phase B: Greedy add
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            for x in all_numbers:
                if x not in S:
                    S, s_sum, did_add = try_add(S, s_sum, x)
                    if did_add:
                        improved = True
                        break

        # Phase C: Single swap
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            s_list = list(S)
            candidates_out = sorted(s_list, key=lambda x: x)
            candidates_in = [y for y in all_numbers if y not in S]
            candidates_in.sort(reverse=True)
            for x in candidates_out:
                if time.time() - start_time > max_time - 1:
                    break
                for y in candidates_in:
                    if y <= x:
                        break
                    S, s_sum, did_swap = try_swap(S, s_sum, x, y)
                    if did_swap:
                        improved = True
                        break
                if improved:
                    break

        # Phase D: Multi-swap (replace 2 with 1, or 3 with 1)
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            s_list = list(S)
            # Replace 2 elements with 1
            for i in range(len(s_list)):
                if time.time() - start_time > max_time - 1:
                    break
                for j in range(i + 1, len(s_list)):
                    x1, x2 = s_list[i], s_list[j]
                    if x1 + x2 >= N:
                        continue
                    temp_set = S - {x1, x2}
                    for y in all_numbers:
                        if y in S or y <= x1 + x2:
                            continue
                        if is_coprime_to_set(y, temp_set):
                            S.discard(x1)
                            S.discard(x2)
                            S.add(y)
                            s_sum += y - x1 - x2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            # Replace 3 elements with 1
            if not improved:
                s_list = list(S)
                for i in range(len(s_list)):
                    if time.time() - start_time > max_time - 1:
                        break
                    for j in range(i + 1, len(s_list)):
                        for k in range(j + 1, len(s_list)):
                            x1, x2, x3 = s_list[i], s_list[j], s_list[k]
                            if x1 + x2 + x3 >= N:
                                continue
                            temp_set = S - {x1, x2, x3}
                            for y in all_numbers:
                                if y in S or y <= x1 + x2 + x3:
                                    continue
                                if is_coprime_to_set(y, temp_set):
                                    S.discard(x1)
                                    S.discard(x2)
                                    S.discard(x3)
                                    S.add(y)
                                    s_sum += y - x1 - x2 - x3
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        return S, s_sum

    # Main search
    best_sum = 0
    best_set = set()

    # Initial solution: all primes
    current_set = set(primes)
    current_sum = sum(current_set)
    current_set, current_sum = local_search(current_set, current_sum, time_limit - 5)
    if current_sum > best_sum:
        best_sum = current_sum
        best_set = set(current_set)

    # Random restarts
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        num_restarts = int(remaining_time / 4)
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit - 3:
                break
            # Perturbed start: random subset of primes + greedy fill
            random_set = set()
            random_sum = 0
            shuffled = primes[:]
            random.shuffle(shuffled)
            for p in shuffled:
                if is_coprime_to_set(p, random_set):
                    random_set.add(p)
                    random_sum += p
            # Also try starting with some composites
            if random.random() < 0.3:
                extra = random.choice([6, 10, 14, 15, 21, 22, 26, 33, 34, 35, 38, 39, 46, 51, 55, 57, 58, 62, 65, 69, 74, 77, 82, 85, 86, 87, 91, 93, 94, 95])
                if is_coprime_to_set(extra, random_set):
                    random_set.add(extra)
                    random_sum += extra
            random_set, random_sum = local_search(random_set, random_sum, time_limit - 3)
            if random_sum > best_sum:
                best_sum = random_sum
                best_set = set(random_set)

    result = list(best_set)
    result = [int(x) for x in result]
    return result
