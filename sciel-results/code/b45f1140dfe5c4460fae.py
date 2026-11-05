import random
import time
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    n = 192
    
    # Helper to check if a set S (list of ints) is valid
    # Returns True if valid
    def is_valid(S):
        if len(S) < 4:
            return True
        S_arr = np.array(S)
        # Check all elements
        for idx, x in enumerate(S):
            # Get all other elements
            others = np.delete(S_arr, idx)
            if len(others) < 3:
                continue
            
            # Generate all sums of 3 distinct elements from 'others'
            # This is O(m^3) if done naively.
            # m is up to 47. 47^3 ~ 100k. 
            # Doing this for every x (48 times) is 4.8M ops per check.
            # If we do 1000 checks, that's 4.8B ops. Too slow.
            
            # We need a faster check.
            pass
        return True # Placeholder

    # We need a faster validity check.
    # Let's stick to the incremental check in the search.
    
    def can_add_to(S, x):
        """
        Check if x can be added to valid set S.
        Returns True if S U {x} is valid.
        """
        if x in S:
            return False
        
        S_list = list(S)
        
        # 1. Check if x divides sum of any 3 distinct elements in S_list
        if len(S_list) >= 3:
            # Optimize: Use modulo arithmetic
            # We need to check if there exist a,b,c in S_list such that (a+b+c) % x == 0
            # We can iterate pairs (a,b) and check if (x - (a+b)%x) % x is in S_list?
            # But we need distinct a,b,c.
            
            # For small x, we can use a count array of residues.
            # For large x, we can just iterate?
            
            # Let's try a simple loop first, it might be fast enough for n~40
            found = False
            # To speed up, we can sort S_list
            S_sorted = sorted(S_list)
            for i in range(len(S_sorted)):
                for j in range(i+1, len(S_sorted)):
                    # We need k > j such that (S[i] + S[j] + S[k]) % x == 0
                    # S[k] % x == (x - (S[i] + S[j]) % x) % x
                    target_res = (- (S_sorted[i] + S_sorted[j])) % x
                    # We need to know if there is a k > j with S_sorted[k] % x == target_res
                    # This is hard to do fast without precomputation.
                    
                    # Given the constraints (60s), maybe we can just run a greedy with a very fast heuristic?
                    pass
            
            # Let's assume the current can_add is the bottleneck.
            # I will implement a Local Search that reuses computations or is just efficient enough.
            
            # Actually, let's just use the current can_add logic but make the search smarter.
            # The current code runs 10000 random greedies.
            # A local search starting from a good greedy is likely better.
            
            return True # Placeholder for logic below

    # Let's implement a specific fast check for the Local Search.
    # We will maintain S as a list.
    # Check 1: x divides sum of 3 in S.
    # Check 2: For each a in S, a does not divide (x + s1 + s2) for s1,s2 in S\{a}.
    
    def is_valid_addition(S, x):
        # S is a list of ints
        # Check 1: x divides sum of 3 distinct in S
        if len(S) >= 3:
            # Use a set of residues?
            # Iterate i < j < k
            # If this is too slow, we break.
            # But n is small (max 48). 48C3 = 17296.
            # 17296 checks is very fast in Python (~1ms).
            # So Check 1 is fast.
            
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    s_ij = S[i] + S[j]
                    for k in range(j+1, len(S)):
                        if (s_ij + S[k]) % x == 0:
                            return False
        
        # Check 2: For each a in S, a does not divide x + s1 + s2
        # where s1, s2 are distinct from a and each other, and in S.
        for a_idx, a in enumerate(S):
            # We need to check sums x + s1 + s2 where s1, s2 in S \ {a}
            # This is equivalent to checking if there exist s1, s2 in S \ {a}
            # such that (x + s1 + s2) % a == 0
            # i.e., (s1 + s2) % a == (-x) % a
            target = (-x) % a
            
            # Iterate pairs in S \ {a}
            # To avoid index confusion, create a list of others
            others = [S[i] for i in range(len(S)) if i != a_idx]
            if len(others) < 2:
                continue
            
            # Check if any pair in 'others' sums to target mod a
            # We can use a set of residues seen so far.
            seen = set()
            for val in others:
                res = val % a
                need = (target - res) % a
                if need in seen:
                    return False
                seen.add(res)
        
        return True

    # Initial Greedy: Largest to Smallest
    def greedy_init():
        S = []
        for x in range(192, 0, -1):
            if is_valid_addition(S, x):
                S.append(x)
        return S

    # Local Search
    def local_search(S, steps=5000):
        S = list(S)
        best_S = list(S)
        best_size = len(S)
        
        for step in range(steps):
            if time.time() - start_time > time_limit:
                break
            
            # Try to add a random element
            candidates = [x for x in range(1, 193) if x not in S]
            if not candidates:
                break
            
            x = random.choice(candidates)
            
            # Check if we can add x
            if is_valid_addition(S, x):
                S.append(x)
                if len(S) > best_size:
                    best_S = list(S)
                    best_size = len(S)
                continue
            
            # Try to swap: remove y from S, add x
            # Pick a random y from S
            y_idx = random.randint(0, len(S) - 1)
            y = S[y_idx]
            
            # Create S' = S \ {y}
            S_prime = [S[i] for i in range(len(S)) if i != y_idx]
            
            # Check if x can be added to S_prime
            if is_valid_addition(S_prime, x):
                S = S_prime + [x]
                # Size is same, but we might have improved "potential"
                # We don't update best_size here unless we find a larger set later
                # But swapping can unblock future additions.
        
        return best_S

    # Run multiple local searches from different seeds
    best_set = []
    
    # Seed 1: Greedy Largest
    S1 = greedy_init()
    if len(S1) > len(best_set):
        best_set = S1
    
    # Run local search on S1
    S1_ls = local_search(S1)
    if len(S1_ls) > len(best_set):
        best_set = S1_ls
        
    # Seed 2: Greedy Smallest
    S2 = []
    for x in range(1, 193):
        if is_valid_addition(S2, x):
            S2.append(x)
    if len(S2) > len(best_set):
        best_set = S2
        
    S2_ls = local_search(S2)
    if len(S2_ls) > len(best_set):
        best_set = S2_ls
        
    # Seed 3: Random Greedy
    random.seed(42)
    for i in range(5):
        S_rand = []
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        for x in candidates:
            if is_valid_addition(S_rand, x):
                S_rand.append(x)
        
        S_rand_ls = local_search(S_rand, steps=1000)
        if len(S_rand_ls) > len(best_set):
            best_set = S_rand_ls
            
    return best_set
