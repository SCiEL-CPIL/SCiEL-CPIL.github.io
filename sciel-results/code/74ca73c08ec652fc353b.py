import math
import time
import sys
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    
    small_primes = [2, 3, 5, 7, 11, 13]
    
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    large_primes = [p for p in range(17, N + 1) if is_prime[p]]
    
    def get_partitions(elements):
        if not elements:
            yield []
            return
        first = elements[0]
        rest = elements[1:]
        for partition in get_partitions(rest):
            for i in range(len(partition)):
                new_partition = [g[:] for g in partition]
                new_partition[i].append(first)
                yield new_partition
            new_partition = [g[:] for g in partition]
            new_partition.append([first])
            yield new_partition

    partitions = list(get_partitions(small_primes))
    
    best_sum = 0
    best_solution = [1]
    
    group_values_cache = {}
    
    def get_group_values(group):
        if not group:
            return [1]
        key = tuple(sorted(group))
        if key in group_values_cache:
            return group_values_cache[key]
        
        vals = []
        def dfs(idx, current_val):
            if current_val > N:
                return
            if idx == len(group):
                vals.append(current_val)
                return
            p = group[idx]
            power = 1
            while current_val * power <= N:
                dfs(idx + 1, current_val * power)
                power *= p
        dfs(0, 1)
        group_values_cache[key] = vals
        return vals

    for partition in partitions:
        if time.time() > deadline:
            break
            
        group_vals = [get_group_values(g) for g in partition]
        if any(len(v) == 0 for v in group_vals):
            continue
            
        for combo in product(*group_vals):
            # Sort a's descending for pruning
            sorted_a = sorted(combo, reverse=True)
            
            # Solve assignment
            memo = {}
            def solve(idx, used):
                if idx == len(sorted_a):
                    return 0, {}
                key = (idx, frozenset(used))
                if key in memo:
                    return memo[key]
                
                a = sorted_a[idx]
                # Option m=1
                val1, assign1 = solve(idx + 1, used)
                best_val = a + val1
                best_assign = assign1.copy()
                
                # Option m=p
                for p in large_primes:
                    if p in used:
                        continue
                    if a * p > N:
                        break
                    gain = a * p
                    new_used = used | {p}
                    val2, assign2 = solve(idx + 1, new_used)
                    if gain + val2 > best_val:
                        best_val = gain + val2
                        best_assign = assign2.copy()
                        best_assign[idx] = p
                
                memo[key] = (best_val, best_assign)
                return best_val, best_assign

            total_sum, assignment = solve(0, frozenset())
            
            if total_sum > best_sum:
                best_sum = total_sum
                nums = []
                for i, a in enumerate(sorted_a):
                    m = assignment.get(i, 1)
                    nums.append(a * m)
                best_solution = [1] + nums

    return best_solution

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

if __name__ == "__main__":
    res = construct()
    print(f"Sum: {sum(res)}, Count: {len(res)}")
