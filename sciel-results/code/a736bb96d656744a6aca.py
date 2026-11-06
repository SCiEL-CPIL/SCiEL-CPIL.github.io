import math
import time
import heapq

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    LIMIT = 10**18

    def get_divisors_from_factors(factors):
        divisors = [1]
        for p, e in factors:
            new_divs = []
            for d in divisors:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divisors = new_divs
        divisors.sort()
        return divisors

    def tau_perp(n, factors=None):
        if factors is None:
            factors = []
            temp = n
            p = 2
            while p * p <= temp:
                if temp % p == 0:
                    e = 0
                    while temp % p == 0:
                        temp //= p
                        e += 1
                    factors.append((p, e))
                p += 1 if p == 2 else 2
            if temp > 1:
                factors.append((temp, 1))
        divisors = get_divisors_from_factors(factors)
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
        return score, len(divisors)

    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in range(101, 300):
        if is_prime(p):
            small_primes.append(p)

    primes_15 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n_base = 1
    for p in primes_15:
        n_base *= p
    if n_base <= LIMIT:
        score_base, _ = tau_perp(n_base, [(p, 1) for p in primes_15])
        best_n = n_base
        best_score = score_base
    else:
        best_n = 1
        best_score = 0

    if time.time() - start_time > DEADLINE:
        return best_n

    # Best-first search
    init_state = tuple((p, 1) for p in primes_15)
    init_n = n_base
    init_score, _ = tau_perp(init_n, list(init_state))
    
    heap = [(-init_score, init_n, init_state)]
    heapq.heapify(heap)
    visited = {init_state}
    
    max_expansions = 30000
    expansions = 0
    
    while heap and expansions < max_expansions:
        if time.time() - start_time > DEADLINE:
            break
        
        neg_score, n, state = heapq.heappop(heap)
        current_score = -neg_score
        expansions += 1
        
        if current_score > best_score:
            best_score = current_score
            best_n = n
        
        state_list = list(state)
        current_primes = set(p for p, e in state_list)
        
        # 1. Increment exponent of existing prime
        for i, (p, e) in enumerate(state_list):
            if n * p > LIMIT:
                continue
            new_state = list(state_list)
            new_state[i] = (p, e + 1)
            new_state.sort()
            key = tuple(new_state)
            if key not in visited:
                visited.add(key)
                new_n = n * p
                s, _ = tau_perp(new_n, new_state)
                heapq.heappush(heap, (-s, new_n, key))
        
        # 2. Decrement exponent (if e > 1)
        for i, (p, e) in enumerate(state_list):
            if e <= 1:
                continue
            new_state = list(state_list)
            new_e = e - 1
            if new_e == 0:
                new_state.pop(i)
            else:
                new_state[i] = (p, new_e)
            new_state.sort()
            key = tuple(new_state)
            if key not in visited:
                visited.add(key)
                new_n = n // p
                s, _ = tau_perp(new_n, new_state)
                heapq.heappush(heap, (-s, new_n, key))
        
        # 3. Add new prime
        for p in small_primes:
            if p in current_primes:
                continue
            if n * p > LIMIT:
                continue
            new_state = state_list + [(p, 1)]
            new_state.sort()
            key = tuple(new_state)
            if key not in visited:
                visited.add(key)
                new_n = n * p
                s, _ = tau_perp(new_n, new_state)
                heapq.heappush(heap, (-s, new_n, key))
        
        # 4. Remove prime (if e == 1)
        for i, (p, e) in enumerate(state_list):
            if e != 1:
                continue
            new_state = list(state_list)
            new_state.pop(i)
            key = tuple(new_state)
            if key not in visited:
                visited.add(key)
                new_n = n // p
                s, _ = tau_perp(new_n, new_state)
                heapq.heappush(heap, (-s, new_n, key))
        
        # 5. Swap: dec one, inc another existing
        for i, (p_i, e_i) in enumerate(state_list):
            if e_i <= 1:
                continue
            for j, (p_j, e_j) in enumerate(state_list):
                if i == j:
                    continue
                new_state = list(state_list)
                new_e_i = e_i - 1
                if new_e_i == 0:
                    new_state.pop(i)
                    if j > i:
                        j -= 1
                    if j < len(new_state):
                        new_state[j] = (p_j, e_j + 1)
                else:
                    new_state[i] = (p_i, new_e_i)
                    new_state[j] = (p_j, e_j + 1)
                new_state.sort()
                key = tuple(new_state)
                if key not in visited:
                    visited.add(key)
                    new_n = n * p_j // p_i
                    if new_n > LIMIT or new_n < 1:
                        continue
                    s, _ = tau_perp(new_n, new_state)
                    heapq.heappush(heap, (-s, new_n, key))
        
        # 6. Swap: dec one, add new prime
        for i, (p_i, e_i) in enumerate(state_list):
            if e_i <= 1:
                continue
            for p_new in small_primes:
                if p_new in current_primes:
                    continue
                new_state = list(state_list)
                new_e_i = e_i - 1
                if new_e_i == 0:
                    new_state.pop(i)
                else:
                    new_state[i] = (p_i, new_e_i)
                new_state.append((p_new, 1))
                new_state.sort()
                key = tuple(new_state)
                if key not in visited:
                    visited.add(key)
                    new_n = n * p_new // p_i
                    if new_n > LIMIT or new_n < 1:
                        continue
                    s, _ = tau_perp(new_n, new_state)
                    heapq.heappush(heap, (-s, new_n, key))

    return best_n
