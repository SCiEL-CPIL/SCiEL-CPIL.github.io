import time
import sys
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 256
    
    # Precompute conflicts for each a:
    # conflict[a] is a set of (b, c) pairs with a < b <= c such that a | (b+c)
    # This is static for a given n.
    # However, checking if a specific pair is in conflict is O(1) if we use a set.
    # But we can be smarter.
    
    # For a given set S, and a candidate a, we want to check if adding a is valid.
    # Condition: For all b, c in S with b <= c, if a < b, then a must not divide b+c.
    # Also, if b < a, then we need to check when a is the "small" one?
    # The condition is symmetric in the sense of the set elements, but the role of 'a' is specific.
    # "For every a < b <= c in the set, a must NOT divide b+c".
    # This means a is the smallest of the three.
    
    # So if we add x to S:
    # 1. x must not act as 'a' for any existing pair (b,c) in S.
    #    i.e., for all b,c in S with x < b <= c, x must not divide b+c.
    # 2. Existing elements in S must not act as 'a' for pairs involving x.
    #    i.e., for all a in S with a < x, and for all c in S with a < c <= x (wait, c can be x or other)
    #    Actually, if a < b <= c.
    #    If x is involved, it can be b or c.
    #    Case 2a: x is b. a < x <= c. a in S, c in S.
    #             Need a not divide x+c.
    #    Case 2b: x is c. a < b <= x. a in S, b in S.
    #             Need a not divide b+x.
    
    # Let's precompute a helper structure for fast validity checking.
    # For each a, let divisors_of_sum[a] be a dict or set?
    
    # Alternative: Just check validity directly but optimized.
    
    def is_valid_set(S):
        S_list = sorted(S)
        len_S = len(S_list)
        # Map value to index for O(1) lookup? 
        # Or just use set membership for c.
        S_set = set(S_list)
        
        for i in range(len_S):
            a = S_list[i]
            # Check pairs b, c from S_list[i+1:]
            # b <= c
            for j in range(i + 1, len_S):
                b = S_list[j]
                # c can be b or any element > b
                # We need to check if there exists c in S_list[j:] such that (b+c) % a == 0
                # (b+c) % a == 0 => c % a == (-b) % a
                # Let target_res = (-b) % a
                # We need to know if any c in S_list[j:] has c % a == target_res
                
                # This is still O(N^3) if done naively.
                # For N=256, N^3 is ~16M ops. In Python, this might be slow if done repeatedly.
                # But we only do it a few times per search step.
                
                # Optimization: Precompute for each a, and each residue r, the sorted list of elements in S with that residue?
                # Since S changes, this is hard to maintain.
                
                # Let's just do the direct check. 16M ops in C is fast, in Python it's ~1-2s.
                # We might do this 1000 times. 2000s is too slow.
                # We need a faster check.
                
                pass
        
        return True

    # Better Validity Check:
    # For a given S, create a frequency array or bitset for residues?
    
    def is_valid_set_fast(S):
        S_list = sorted(S)
        S_set = set(S_list)
        
        # For each a in S:
        #   For each b in S, b > a:
        #     Check if there is c in S, c >= b, such that (b+c) % a == 0.
        
        # Precompute for each a, the set of residues of elements in S?
        # No, let's iterate a, then b, then solve for c.
        # c = k*a - b.
        # We need c in S and c >= b.
        # k is such that c > a.
        
        for a in S_list:
            # Iterate b in S, b > a
            for b in S_list:
                if b <= a:
                    continue
                # Find c in S such that c >= b and (b+c) % a == 0
                # c % a == (-b) % a
                target_res = (-b) % a
                # Iterate k starting from ceil((b + target_res) / a)?
                # Min c is b.
                # c = k*a - b
                # k*a >= b + b = 2b => k >= 2b/a
                # Also c >= b.
                
                # Range of k:
                # c <= 256 => k*a - b <= 256 => k <= (256+b)/a
                # c >= b => k*a - b >= b => k >= 2b/a
                
                k_min = int((2 * b + a - 1) / a) # ceil(2b/a)
                k_max = int((256 + b) / a)
                
                for k in range(k_min, k_max + 1):
                    c = k * a - b
                    if c >= b and c in S_set:
                        return False
        return True

    # This is O(N^3) worst case, but with early termination and small N, it's okay for a few calls.
    # But for local search, we call it many times.
    
    # Let's precompute `conflict_pairs[a]` = set of (b,c)
    # Then `is_valid(S)` is:
    # for a in S:
    #   for (b,c) in conflict_pairs[a]:
    #     if b in S and c in S: return False
    
    conflict_pairs = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b, n + 1):
                if (b + c) % a == 0:
                    conflict_pairs[a].add((b, c))

    def is_valid_set_conflict(S):
        S_set = set(S)
        for a in S:
            for (b, c) in conflict_pairs[a]:
                if b in S_set and c in S_set:
                    return False
        return True

    # This is faster if conflict_pairs are small.
    # For large a, there are few conflicts.
    # For small a, many conflicts.
    
    # Strategy:
    # 1. Find best interval [L, n] that is valid.
    # 2. Local search to add/swap.
    
    best_set = set()
    best_size = 0
    
    # Find best interval
    for L in range(1, n + 1):
        if time.time() > deadline - 10:
            break
        S_test = set(range(L, n + 1))
        if is_valid_set_conflict(S_test):
            if len(S_test) > best_size:
                best_size = len(S_test)
                best_set = S_test

    if not best_set:
        return list(range(192, 257))

    current_set = set(best_set)
    current_size = len(current_set)
    
    # Local Search
    # We will try to add elements or swap.
    # To make it efficient, we can precompute for each x not in S, the "cost" of adding it (number of conflicts created).
    # But conflicts are binary.
    
    # Simple Hill Climbing with Random Restarts
    rng = random.Random(42)
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            S = set(best_set)
        else:
            # Perturb best_set
            S = set(best_set)
            # Remove a few random elements
            for _ in range(rng.randint(1, 5)):
                if S:
                    x = rng.choice(list(S))
                    S.remove(x)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Try adding any element
            candidates = [x for x in range(1, n + 1) if x not in S]
            rng.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline - 2:
                    break
                test_S = S | {x}
                if is_valid_set_conflict(test_S):
                    S = test_S
                    improved = True
                    break
            
            # If no add worked, try swaps
            if not improved:
                S_list = list(S)
                rng.shuffle(S_list)
                for x in S_list:
                    if time.time() > deadline - 2:
                        break
                    test_S = S - {x}
                    # Try to add any y
                    candidates_y = [y for y in range(1, n + 1) if y not in test_S and y != x]
                    rng.shuffle(candidates_y)
                    for y in candidates_y:
                        if time.time() > deadline - 2:
                            break
                        test_S2 = test_S | {y}
                        if is_valid_set_conflict(test_S2):
                            S = test_S2
                            improved = True
                            break
                    if improved:
                        break
        
        if len(S) > current_size:
            current_size = len(S)
            best_set = S

    return list(best_set)

# EVOLVE-BLOCK-END
