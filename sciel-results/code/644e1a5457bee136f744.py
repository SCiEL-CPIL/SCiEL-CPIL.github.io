"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    n = 256
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Precompute prime factors for each number 1..n
    # factors[i] = frozenset of prime factors of i
    def get_prime_factors(x):
        if x <= 1:
            return frozenset()
        factors = set()
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors.add(d)
                x //= d
            d += 1
        if x > 1:
            factors.add(x)
        return frozenset(factors)
    
    factors = [frozenset() for _ in range(n + 1)]
    for i in range(2, n + 1):
        factors[i] = get_prime_factors(i)
    
    # Compatibility check: two numbers are compatible if their prime factor sets are disjoint
    def compatible(a, b):
        if a == b:
            return True
        return len(factors[a] & factors[b]) == 0
    
    # Check if a candidate can be added to a set of selected numbers
    def can_add(candidate, selected_set):
        if candidate in selected_set:
            return False
        for s in selected_set:
            if not compatible(candidate, s):
                return False
        return True
    
    # Greedy initial solution with multiple orderings
    def greedy_solution(order):
        selected = []
        selected_set = set()
        for a in order:
            if time.time() > deadline:
                break
            if can_add(a, selected_set):
                selected.append(a)
                selected_set.add(a)
        return selected
    
    # Local search with single and double swaps
    def local_search(selected, time_limit):
        selected_set = set(selected)
        current_sum = sum(selected)
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            
            # Try single swaps: remove i, add j if compatible and improves sum
            for i in list(selected):
                others = [s for s in selected if s != i]
                others_set = set(others)
                for j in range(n, 1, -1):
                    if j in selected_set:
                        continue
                    if can_add(j, others_set):
                        new_sum = current_sum - i + j
                        if new_sum > current_sum:
                            selected = others + [j]
                            selected_set = set(selected)
                            current_sum = new_sum
                            improved = True
                            break
                if improved:
                    break
            
            # Try double swaps: remove i, j, add k, l if compatible and improves sum
            if not improved and time.time() < time_limit - 2.0:
                for i_idx in range(len(selected)):
                    for j_idx in range(i_idx + 1, len(selected)):
                        i, j = selected[i_idx], selected[j_idx]
                        others = [s for k, s in enumerate(selected) if k != i_idx and k != j_idx]
                        others_set = set(others)
                        for k in range(n, 1, -1):
                            if k in selected_set:
                                continue
                            if not can_add(k, others_set):
                                continue
                            for l in range(n, 1, -1):
                                if l in selected_set or l == k:
                                    continue
                                if not can_add(l, others_set) or not compatible(k, l):
                                    continue
                                new_sum = current_sum - i - j + k + l
                                if new_sum > current_sum:
                                    selected = others + [k, l]
                                    selected_set = set(selected)
                                    current_sum = new_sum
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return selected
    
    # Prime power heuristic: for each prime, take the largest power <= n
    def prime_power_solution():
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        
        primes = [i for i in range(2, n + 1) if is_prime[i]]
        result_set = {1}
        
        for p in primes:
            power = p
            while power * p <= n:
                power *= p
            # Check if power is compatible with all in result_set
            # Since all are prime powers of distinct primes, they should be compatible
            can_add_flag = True
            for s in result_set:
                if s != 1 and not compatible(power, s):
                    can_add_flag = False
                    break
            if can_add_flag:
                result_set.add(power)
        
        return list(result_set)
    
    # Run multiple strategies and keep the best
    best = []
    best_sum = 0
    
    # Strategy 1: Greedy descending
    order1 = list(range(n, 0, -1))
    result1 = greedy_solution(order1)
    if sum(result1) > best_sum:
        best = result1
        best_sum = sum(result1)
    
    # Strategy 2: Prime power solution
    result2 = prime_power_solution()
    if sum(result2) > best_sum:
        best = result2
        best_sum = sum(result2)
    
    # Strategy 3: Sort by value/number_of_prime_factors
    def num_prime_factors(x):
        if x <= 1:
            return 0
        count = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    order3 = sorted(range(n, 0, -1), key=lambda x: (-x / max(1, num_prime_factors(x)), -x))
    result3 = greedy_solution(order3)
    if sum(result3) > best_sum:
        best = result3
        best_sum = sum(result3)
    
    # Local search on the best solution found so far
    if time.time() < deadline - 5.0:
        best = local_search(best, deadline)
        best_sum = sum(best)
    
    # Ensure 1 is included (it's always compatible and adds to sum)
    if 1 not in best:
        best.append(1)
        best_sum += 1
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
