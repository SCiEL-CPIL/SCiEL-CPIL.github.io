import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Precompute conflict information
    # For each pair (b, c), compute which a values divide b*c
    # This helps us quickly check if adding x creates a conflict
    
    # For each x in 2..256, precompute the set of pairs (b,c) such that x divides b*c
    # But that's expensive. Instead, let's precompute for each x, the "bad pairs"
    
    # Actually, let's think differently:
    # When checking if x can be added to S, we need:
    # 1. For all pairs (b,c) in S: x does NOT divide b*c
    # 2. For all b in S: x does NOT divide x*b (always false since x|x*b)
    #    Wait, that's wrong. Let me re-read the constraint.
    # 
    # "a does not divide b*c" for three distinct members a,b,c
    # So for a fixed x being added, we need to check:
    # - For all distinct b,c in S: x does not divide b*c
    # - For all distinct b,c in S: b does not divide x*c  (i.e., x*c % b != 0)
    # - For all distinct b,c in S: c does not divide x*b  (i.e., x*b % c != 0)
    
    # Precompute: for each x, the set of (b,c) pairs where x divides b*c
    # This is O(N^3) which is 255^3 ~ 16M, might be too slow
    
    # Alternative: precompute for each x, the set of numbers that "conflict" with x
    # in the sense that if both are in S, they create a conflict with some third element
    
    # Let's use a different approach: precompute the "conflict graph"
    # Two elements u,v conflict if there exists w such that u|v*w or v|u*w or w|u*v
    # But this depends on w being in S, so it's not purely pairwise
    
    # Let's just optimize the can_add function using precomputed tables
    
    # Precompute for each x in 2..256:
    # divisors[x] = set of d such that d divides x
    # multiples[x] = set of m such that x divides m (within 2..256)
    
    # For the check "x divides b*c":
    # We can factor x and check if b*c has enough prime factors
    
    # Let's precompute prime factorizations
    def factorize(num):
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 0) + 1
        return factors
    
    # Precompute factorizations for all numbers
    factorizations = {}
    for i in range(2, N + 1):
        factorizations[i] = factorize(i)
    
    # Precompute for each x, the set of (b,c) pairs where x divides b*c
    # This is expensive but let's try a smarter approach
    
    # For can_add(x, S), we need to check:
    # 1. No pair (b,c) in S has x | b*c
    # 2. No b in S has b | x*c for any c in S, c != b
    # 3. No c in S has c | x*b for any b in S, b != c
    
    # For check 1: x | b*c means the prime factorization of x must be "covered" by b*c
    # We can precompute for each x, a function that given b and c, checks if x | b*c
    
    # Let's precompute a table: divides_bc[x][b][c] = True if x | b*c
    # But that's 255 * 255 * 255 = 16M entries, too much memory
    
    # Alternative: for each x, precompute the set of b values such that for some c, x | b*c
    # Actually, let's just optimize the inner loop
    
    def can_add_fast(x, S_list):
        """Check if x can be added to set S_list (list of ints)"""
        L = len(S_list)
        
        # Check 1: for all pairs (b,c) in S_list, x does not divide b*c
        for i in range(L):
            b = S_list[i]
            for j in range(i + 1, L):
                c = S_list[j]
                if (b * c) % x == 0:
                    return False
        
        # Check 2: for all pairs (b,c) in S_list, b does not divide x*c
        # and c does not divide x*b
        # We can combine: for all ordered pairs (b,c) with b != c, 
        # check b | x*c and c | x*b
        for i in range(L):
            b = S_list[i]
            for j in range(L):
                if i == j:
                    continue
                c = S_list[j]
                if (x * c) % b == 0:
                    return False
                # Note: we also need c | x*b, but that's the same check with b and c swapped
                # Since we're iterating all ordered pairs, we'll catch it
        
        return True
    
    # Greedy Construction with multiple strategies
    def greedy_add_largest(S, candidates):
        """Add elements from largest to smallest"""
        for x in sorted(candidates, reverse=True):
            S_list = list(S)
            if can_add_fast(x, S_list):
                S.add(x)
        return S
    
    def greedy_add_smallest(S, candidates):
        """Add elements from smallest to largest"""
        for x in sorted(candidates):
            S_list = list(S)
            if can_add_fast(x, S_list):
                S.add(x)
        return S
    
    def greedy_randomized(S, candidates, seed=42):
        """Add elements in random order"""
        rng = random.Random(seed)
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        for x in shuffled:
            S_list = list(S)
            if can_add_fast(x, S_list):
                S.add(x)
        return S
    
    # Try multiple greedy strategies and keep the best
    best_set = set()
    best_size = 0
    
    # Strategy 1: Largest first
    S1 = greedy_add_largest(set(), numbers)
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = set(S1)
    
    # Strategy 2: Smallest first
    S2 = greedy_add_smallest(set(), numbers)
    if len(S2) > best_size:
        best_size = len(S2)
        best_set = set(S2)
    
    # Strategy 3: Randomized with different seeds
    for seed in [1, 2, 3, 4, 5]:
        S3 = greedy_randomized(set(), numbers, seed)
        if len(S3) > best_size:
            best_size = len(S3)
            best_set = set(S3)
    
    current_set = set(best_set)
    
    # Local Search with swap and 2-swap moves
    def local_search(S, max_iter=2000):
        """Improve S using swap moves"""
        S = set(S)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() - start_time < deadline:
            improved = False
            iter_count += 1
            S_list = list(S)
            remaining = [x for x in numbers if x not in S]
            
            # Try to add any element
            for x in remaining:
                if can_add_fast(x, S_list):
                    S.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove a, add b
            for a in S_list:
                if time.time() - start_time > deadline:
                    return S
                S_without_a = S - {a}
                S_without_a_list = list(S_without_a)
                for b in remaining:
                    if can_add_fast(b, S_without_a_list):
                        S.remove(a)
                        S.add(b)
                        improved = True
                        break
                if improved:
                    break
        
        return S
    
    # Run local search from the best greedy result
    current_set = local_search(current_set)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = set(current_set)
    
    # Try 2-opt: remove two elements, add two different elements
    def two_opt_search(S, max_iter=500):
        """Improve S using 2-swap moves"""
        S = set(S)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() - start_time < deadline:
            improved = False
            iter_count += 1
            S_list = list(S)
            remaining = [x for x in numbers if x not in S]
            
            # Try all pairs to remove and all pairs to add
            # This is O(|S|^2 * |remaining|^2) which might be too slow
            # Let's be more selective: try removing the two "most restrictive" elements
            # and adding the two "least restrictive" remaining elements
            
            # For now, just try a few random 2-swaps
            if len(S_list) >= 2 and len(remaining) >= 2:
                for _ in range(10):  # Try 10 random 2-swaps
                    if time.time() - start_time > deadline:
                        return S
                    remove_pair = random.sample(S_list, 2)
                    add_pair = random.sample(remaining, 2)
                    
                    S_new = S - set(remove_pair)
                    S_new_list = list(S_new)
                    
                    # Check if both can be added
                    valid = True
                    for x in add_pair:
                        if not can_add_fast(x, S_new_list):
                            valid = False
                            break
                        S_new_list.append(x)
                    
                    if valid:
                        S = S_new | set(add_pair)
                        improved = True
                        break
        
        return S
    
    if time.time() - start_time < deadline:
        current_set = two_opt_search(current_set)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    # Simulated annealing style: accept worse solutions sometimes to escape local optima
    def simulated_annealing(S, max_iter=1000, start_temp=10.0, end_temp=0.1):
        """SA-style search"""
        S = set(S)
        best_S = set(S)
        best_size = len(S)
        current_temp = start_temp
        temp_decay = (end_temp / start_temp) ** (1.0 / max_iter)
        
        for iter_count in range(max_iter):
            if time.time() - start_time > deadline:
                break
            current_temp *= temp_decay
            
            S_list = list(S)
            remaining = [x for x in numbers if x not in S]
            
            # Generate a random move
            move_type = random.random()
            new_S = set(S)
            
            if move_type < 0.3 and remaining:
                # Add a random element
                x = random.choice(remaining)
                if can_add_fast(x, S_list):
                    new_S.add(x)
            elif move_type < 0.7 and S_list:
                # Swap: remove a, add b
                a = random.choice(S_list)
                b = random.choice(remaining) if remaining else None
                if b is not None:
                    S_without_a = S - {a}
                    S_without_a_list = list(S_without_a)
                    if can_add_fast(b, S_without_a_list):
                        new_S = S_without_a | {b}
            elif move_type < 0.9 and len(S_list) >= 2 and len(remaining) >= 2:
                # 2-swap
                remove_pair = random.sample(S_list, 2)
                add_pair = random.sample(remaining, 2)
                S_new = S - set(remove_pair)
                S_new_list = list(S_new)
                valid = True
                for x in add_pair:
                    if not can_add_fast(x, S_new_list):
                        valid = False
                        break
                    S_new_list.append(x)
                if valid:
                    new_S = S_new | set(add_pair)
            else:
                # Remove a random element (to escape local optima)
                if S_list:
                    a = random.choice(S_list)
                    new_S = S - {a}
            
            # Accept or reject based on simulated annealing
            old_size = len(S)
            new_size = len(new_S)
            
            if new_size >= old_size:
                S = new_S
                if new_size > best_size:
                    best_size = new_size
                    best_S = set(S)
            else:
                # Worse solution: accept with probability exp((new_size - old_size) / temp)
                delta = new_size - old_size  # negative
                prob = min(1.0, max(0.0, (delta / max(current_temp, 0.01))))
                if random.random() < prob:
                    S = new_S
        
        return best_S
    
    if time.time() - start_time < deadline:
        sa_result = simulated_annealing(current_set)
        if len(sa_result) > best_size:
            best_size = len(sa_result)
            best_set = set(sa_result)
    
    # Final: try to add any remaining elements to best_set
    best_list = list(best_set)
    remaining = [x for x in numbers if x not in best_set]
    for x in remaining:
        if can_add_fast(x, best_list):
            best_set.add(x)
            best_list.append(x)
    
    # Verify the result is valid
    def is_valid_set(S):
        S_list = list(S)
        L = len(S_list)
        for i in range(L):
            a = S_list[i]
            for j in range(i + 1, L):
                b = S_list[j]
                for k in range(j + 1, L):
                    c = S_list[k]
                    if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                        return False
        return True
    
    if not is_valid_set(best_set):
        # Fallback to a known valid set
        return list(range(129, 257))  # Numbers from 129 to 256
    
    return sorted(best_set)
