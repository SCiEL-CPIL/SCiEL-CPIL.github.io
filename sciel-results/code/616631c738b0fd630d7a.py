"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute lcm for all pairs
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)
    
    def get_lcm(a, b):
        if a <= b:
            return lcm_table[(a, b)]
        else:
            return lcm_table[(b, a)]
    
    def has_bad_triple_with_x(selected, x):
        """Check if adding x to selected creates a bad triple."""
        n = len(selected)
        for i in range(n):
            a = selected[i]
            lax = get_lcm(a, x)
            for j in range(i + 1, n):
                b = selected[j]
                if get_lcm(a, b) == lax and get_lcm(b, x) == lax:
                    return True
        return False
    
    def is_valid_set(selected):
        """Full validation that no bad triple exists in selected."""
        n = len(selected)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = selected[i], selected[j], selected[k]
                    if get_lcm(a, b) == get_lcm(a, c) == get_lcm(b, c):
                        return False
        return True
    
    def greedy(order):
        """Greedy construction with given element order."""
        selected = []
        selected_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if x not in selected_set:
                if not has_bad_triple_with_x(selected, x):
                    selected.append(x)
                    selected_set.add(x)
        return selected
    
    # Try multiple orderings and keep the best
    best = []
    
    # Ordering 1: ascending
    s1 = greedy(nums)
    if len(s1) > len(best):
        best = list(s1)
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Ordering 2: descending
    s2 = greedy(nums[::-1])
    if len(s2) > len(best):
        best = list(s2)
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Ordering 3: odd first, then even
    s3 = greedy(sorted(nums, key=lambda x: (x & 1, x)))
    if len(s3) > len(best):
        best = list(s3)
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Ordering 4: by number of prime factors (more factors = more constraints, add later)
    def omega(x):
        c = 0
        n = x
        d = 2
        while d * d <= n:
            while n % d == 0:
                c += 1
                n //= d
            d += 1
        if n > 1:
            c += 1
        return c
    
    s4 = greedy(sorted(nums, key=omega))
    if len(s4) > len(best):
        best = list(s4)
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Ordering 5: powers of 2 first (they rarely form bad triples with each other)
    def is_pow2(x):
        return (x & (x - 1)) == 0
    
    s5 = greedy(sorted(nums, key=lambda x: (not is_pow2(x), x)))
    if len(s5) > len(best):
        best = list(s5)
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Ordering 6: odd descending, then even
    s6 = greedy(sorted(nums, key=lambda x: (x & 1, -x)))
    if len(s6) > len(best):
        best = list(s6)
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Local search: try adding remaining elements
    improved = list(best)
    improved_set = set(improved)
    remaining = [x for x in nums if x not in improved_set]
    
    for x in remaining:
        if time.time() > deadline:
            break
        if not has_bad_triple_with_x(improved, x):
            improved.append(x)
            improved_set.add(x)
    
    if len(improved) > len(best):
        best = improved
    
    # Swap-based local search: try removing one element and adding two
    current_size = len(best)
    improvement_found = True
    while improvement_found and time.time() < deadline - 10:
        improvement_found = False
        best_set = set(best)
        
        for i in range(len(best)):
            if time.time() > deadline - 10:
                break
            removed = best[i]
            temp_list = best[:i] + best[i+1:]
            temp_set = best_set - {removed}
            
            for a in range(1, N + 1):
                if time.time() > deadline - 10:
                    break
                if a in temp_set:
                    continue
                if not has_bad_triple_with_x(temp_list, a):
                    temp_list_a = temp_list + [a]
                    temp_set_a = temp_set | {a}
                    
                    for b in range(a + 1, N + 1):
                        if time.time() > deadline - 10:
                            break
                        if b in temp_set_a:
                            continue
                        if not has_bad_triple_with_x(temp_list_a, b):
                            new_set = temp_list_a + [b]
                            if len(new_set) > current_size:
                                best = new_set
                                current_size = len(best)
                                best_set = set(best)
                                improvement_found = True
                                break
                    if improvement_found:
                        break
            if improvement_found:
                break
    
    # Final validation
    if not is_valid_set(best):
        # Fallback to greedy
        best = greedy(nums)
    
    if not best:
        best = [1]
    
    return best
# EVOLVE-BLOCK-END
