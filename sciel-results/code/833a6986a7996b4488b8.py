import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 60.0  # safety margin

    def elapsed():
        return time.time() - start_time

    # The seed returns [1, ..., 22] (since isqrt(512)=22), size 22
    # We need to maximize |S| such that lcm(a,b) <= 512 for all a,b in S, S subset [1,512]

    # Key insight: If we include n, we can include all divisors of n.
    # But lcm(a,b) <= 512 is the constraint.
    # 
    # A number x is "compatible" with y if lcm(x,y) <= 512.
    # Note: lcm(x,y) = x*y / gcd(x,y).
    # 
    # Strategy: Greedy or combinatorial search.
    # Since 512 = 2^9, numbers with large odd parts are problematic.
    # 
    # Let's think about which numbers can be in the set.
    # If x has an odd prime factor p, then for any y in the set, 
    # lcm(x,y) must be <= 512.
    # 
    # Powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512. All pairwise lcms are powers of 2 up to 512.
    # So all powers of 2 up to 512 can be in the set. That's 10 numbers.
    # 
    # What about numbers like 3? lcm(3, 2^k) = 3*2^k. We need 3*2^k <= 512, so 2^k <= 170.67, k <= 7.
    # So 3 can coexist with 1,2,4,8,16,32,64,128 but not 256 or 512.
    # 
    # This is a combinatorial optimization problem. Let's use a greedy/local search approach.
    # 
    # First, let's identify candidate numbers. A number n in [1,512] can potentially be in the set
    # if there exists at least one compatible set containing it. But for maximization, we want the largest clique
    # in the compatibility graph where edge exists if lcm(a,b) <= 512.
    # 
    # Actually, it's simpler: we want the largest subset where ALL pairwise lcms are <= 512.
    # This is a clique in a graph where vertices are 1..512 and edges connect pairs with lcm <= 512.
    # 
    # For efficiency, let's use a greedy approach with local search.
    # 
    # Observation: If we take all divisors of 512, that's just powers of 2 (10 elements).
    # Can we do better? Yes, by including numbers like 3, 5, etc., but we may need to exclude some powers of 2.
    
    # Let's try a different approach: characterize valid sets.
    # A set S is valid iff for all a,b in S, lcm(a,b) <= 512.
    # 
    # Key: lcm(a,b) <= 512 means that for every prime p, max(v_p(a), v_p(b)) * p^... 
    # Actually, lcm(a,b) = product over primes p of p^max(v_p(a), v_p(b)).
    # So we need product of p^max(v_p(a), v_p(b)) <= 512 for all pairs.
    
    # Greedy: start with empty set, add numbers that are compatible with all current members.
    # But order matters. Let's try multiple orderings.
    
    # Better: Use local search. Start with a good initial set, then try to swap/add.
    
    # Initial set: all powers of 2 up to 512
    initial_set = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
    
    def is_compatible(a, b):
        return (a * b) // gcd(a, b) <= 512
    
    def is_valid_set(s):
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                if not is_compatible(lst[i], lst[j]):
                    return False
        return True
    
    best_set = set(initial_set)
    best_size = len(best_set)
    
    # Try to add elements greedily
    # For each number from 1 to 512, try to add it if compatible with all in current set
    # Try multiple orderings
    
    import random
    random.seed(42)
    
    def try_add_elements(current_set, order):
        s = set(current_set)
        for n in order:
            if n in s:
                continue
            # Check if n is compatible with all in s
            compatible = True
            for m in s:
                if not is_compatible(n, m):
                    compatible = False
                    break
            if compatible:
                s.add(n)
        return s
    
    # Try natural order
    for n in range(1, 513):
        if n not in best_set:
            compatible = all(is_compatible(n, m) for m in best_set)
            if compatible:
                best_set.add(n)
    
    best_size = len(best_set)
    
    # Try random orderings
    numbers = list(range(1, 513))
    for trial in range(50):
        if elapsed() > deadline - 5:
            break
        random.shuffle(numbers)
        s = set(initial_set)
        for n in numbers:
            if n in s:
                continue
            compatible = all(is_compatible(n, m) for m in s)
            if compatible:
                s.add(n)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Local search: try to swap elements
    for trial in range(100):
        if elapsed() > deadline - 5:
            break
        improved = False
        current = set(best_set)
        # Try removing one element and adding one or two others
        for rem in list(current):
            temp = current - {rem}
            for add in range(1, 513):
                if add in temp:
                    continue
                if all(is_compatible(add, m) for m in temp):
                    new_set = temp | {add}
                    if len(new_set) > best_size:
                        best_set = new_set
                        best_size = len(best_set)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    if elapsed() > deadline - 2:
        return list(best_set)
    
    # Second phase of local search: try swapping two elements
    for trial in range(50):
        if elapsed() > deadline - 3:
            break
        improved = False
        current = list(best_set)
        for i in range(len(current)):
            for j in range(i+1, len(current)):
                rem1, rem2 = current[i], current[j]
                temp = set(best_set) - {rem1, rem2}
                for add1 in range(1, 513):
                    if add1 in temp:
                        continue
                    if not all(is_compatible(add1, m) for m in temp):
                        continue
                    for add2 in range(add1+1, 513):
                        if add2 in temp or add2 == add1:
                            continue
                        if not all(is_compatible(add2, m) for m in temp):
                            continue
                        if not is_compatible(add1, add2):
                            continue
                        new_set = temp | {add1, add2}
                        if len(new_set) > best_size:
                            best_set = new_set
                            best_size = len(new_set)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return list(best_set)
