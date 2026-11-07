# EVOLVE-BLOCK-START
import math
import random

def construct():
    n = 256
    # Precompute prime factors for each number
    prime_factors = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        if not prime_factors[i]:  # i is prime
            for j in range(i, n + 1, i):
                prime_factors[j].append(i)
    
    def get_greedy_solution():
        used_primes = set()
        result = [1]
        result_set = {1}
        for m in range(n, 1, -1):
            conflict = False
            for p in prime_factors[m]:
                if p in used_primes:
                    conflict = True
                    break
            if not conflict:
                result.append(m)
                result_set.add(m)
                for p in prime_factors[m]:
                    used_primes.add(p)
        return result, result_set

    def get_prime_set(result):
        ps = set()
        for m in result:
            for p in prime_factors[m]:
                ps.add(p)
        return ps

    def local_search(result, result_set):
        current_primes = get_prime_set(result)
        improved = True
        while improved:
            improved = False
            indices = list(range(len(result)))
            random.shuffle(indices)
            for i in indices:
                m1 = result[i]
                p1 = set(prime_factors[m1])
                others_primes = current_primes - p1
                
                best_m2 = 0
                best_gain = 0
                
                for m2 in range(n, 1, -1):
                    if m2 == m1 or m2 in result_set:
                        continue
                    conflict = False
                    for p in prime_factors[m2]:
                        if p in others_primes:
                            conflict = True
                            break
                    if not conflict:
                        if m2 > m1:
                            gain = m2 - m1
                            if gain > best_gain:
                                best_gain = gain
                                best_m2 = m2
                if best_m2 > 0:
                    # Update structures
                    result[i] = best_m2
                    result_set.remove(m1)
                    result_set.add(best_m2)
                    current_primes -= p1
                    for p in prime_factors[best_m2]:
                        current_primes.add(p)
                    improved = True
                    break
        return result, result_set

    # Iterated Local Search
    best_result, best_result_set = get_greedy_solution()
    best_sum = sum(best_result)
    
    for _ in range(100): # Increase iterations slightly
        current, current_set = get_greedy_solution()
        # Perturb
        for _ in range(5):
            i = random.randint(0, len(current) - 1)
            m1 = current[i]
            p1 = set(prime_factors[m1])
            others_primes = get_prime_set(current) - p1
            
            candidates = []
            for m2 in range(n, 1, -1):
                if m2 == m1 or m2 in current_set:
                    continue
                conflict = False
                for p in prime_factors[m2]:
                    if p in others_primes:
                        conflict = True
                        break
                if not conflict:
                    candidates.append(m2)
            
            if candidates:
                m2 = random.choice(candidates)
                current[i] = m2
                current_set.remove(m1)
                current_set.add(m2)
        
        # Local Search
        current, current_set = local_search(current, current_set)
        curr_sum = sum(current)
        
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_result = current
            best_result_set = current_set

    return best_result
# EVOLVE-BLOCK-END
