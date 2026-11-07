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
    
    # Heuristic Initial Solution
    # For each prime, pick the largest power <= 256
    initial_set = set()
    for p in primes:
        x = p
        while x * p <= N:
            x *= p
        initial_set.add(x)
        
    # Specific Merges
    # 5 and 7 -> 245
    if 125 in initial_set and 49 in initial_set:
        initial_set.remove(125)
        initial_set.remove(49)
        initial_set.add(245)
    # 11 and 13 -> 143
    if 11 in initial_set and 13 in initial_set:
        initial_set.remove(11)
        initial_set.remove(13)
        initial_set.add(143)
        
    current_set = initial_set
    current_sum = sum(current_set)
    
    def local_search(S, s_sum, max_time):
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            
            # Phase 1: Add
            for x in all_numbers:
                if x not in S and is_coprime_to_set(x, S):
                    S.add(x)
                    s_sum += x
                    improved = True
                    break
            
            # Phase 2: Swap 1 for 1
            if not improved:
                s_list = list(S)
                candidates_in = [y for y in all_numbers if y not in S]
                candidates_in.sort(reverse=True)
                for a in s_list:
                    temp_set = S - {a}
                    for b in candidates_in:
                        if b <= a:
                            break
                        if is_coprime_to_set(b, temp_set):
                            S.discard(a)
                            S.add(b)
                            s_sum += b - a
                            improved = True
                            break
                    if improved:
                        break
                        
            # Phase 3: Swap 2 for 1
            if not improved:
                s_list = list(S)
                for i in range(len(s_list)):
                    if time.time() - start_time > max_time - 1:
                        break
                    for j in range(i + 1, len(s_list)):
                        a, c = s_list[i], s_list[j]
                        if a + c >= N:
                            continue
                        temp_set = S - {a, c}
                        for b in all_numbers:
                            if b in S or b <= a + c:
                                continue
                            if is_coprime_to_set(b, temp_set):
                                S.discard(a)
                                S.discard(c)
                                S.add(b)
                                s_sum += b - a - c
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                        
            # Phase 4: Swap 1 for 2
            if not improved:
                s_list = list(S)
                for a in s_list:
                    if time.time() - start_time > max_time - 1:
                        break
                    temp_set = S - {a}
                    # Find two numbers b, d not in S, coprime to temp_set and each other, sum > a
                    # This is expensive, limit candidates
                    candidates = [y for y in all_numbers if y not in S and is_coprime_to_set(y, temp_set)]
                    candidates.sort(reverse=True)
                    found = False
                    for i in range(len(candidates)):
                        if time.time() - start_time > max_time - 1:
                            break
                        b = candidates[i]
                        if b <= a:
                            break
                        for j in range(i + 1, len(candidates)):
                            d = candidates[j]
                            if b + d <= a:
                                break
                            if is_coprime_to_set(d, temp_set | {b}):
                                S.discard(a)
                                S.add(b)
                                S.add(d)
                                s_sum += b + d - a
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if improved:
                        break
                        
        return S, s_sum

    best_sum = current_sum
    best_set = set(current_set)
    
    current_set, current_sum = local_search(current_set, current_sum, time_limit - 5)
    if current_sum > best_sum:
        best_sum = current_sum
        best_set = set(current_set)
        
    # Random Restarts
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        num_restarts = int(remaining_time / 4)
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit - 3:
                break
            # Perturb
            random_set = set(best_set)
            # Remove a random element
            if random_set:
                random_set.remove(random.choice(list(random_set)))
            # Greedy fill
            random_sum = sum(random_set)
            for x in all_numbers:
                if x not in random_set and is_coprime_to_set(x, random_set):
                    random_set.add(x)
                    random_sum += x
            
            random_set, random_sum = local_search(random_set, random_sum, time_limit - 3)
            if random_sum > best_sum:
                best_sum = random_sum
                best_set = set(random_set)

    result = list(best_set)
    result = [int(x) for x in result]
    return result
