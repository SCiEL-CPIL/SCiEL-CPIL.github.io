"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd

def construct():
    start_time = time.time()
    # Evaluator kills at 90s. Use a safety margin of 10s.
    deadline = start_time + 80
    
    LIMIT = 512
    N = LIMIT
    
    def check_time():
        return time.time() > deadline

    # --- Precomputation ---
    # 1. Divisors for each number (useful for structural analysis)
    divisors = [set() for _ in range(N + 1)]
    for d in range(1, N + 1):
        for m in range(d, N + 1, d):
            divisors[m].add(d)
            
    # 2. Compatibility Matrix: compat[i][j] = True if lcm(i,j) <= N
    # This is the core constraint.
    # Storing as a list of bytearrays for memory efficiency and speed.
    compat = [bytearray(N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            l = i * j // gcd(i, j)
            if l <= N:
                compat[i][j] = 1
                compat[j][i] = 1

    # 3. Precompute compatible neighbors for each node
    # neighbors[i] = set of j such that lcm(i,j) <= N
    # We can derive this from compat, but sets are faster for intersection logic in some contexts.
    # However, for "is_compatible_with_all", iterating over the set S and checking compat[x][y] is O(|S|).
    # Let's stick to the compat matrix for the main checks.

    def is_compatible(x, S):
        """Check if x is compatible with all elements in S (set or list)."""
        for y in S:
            if not compat[x][y]:
                return False
        return True

    def greedy_add(S, candidates=None):
        """
        Add elements from candidates to S if compatible.
        If candidates is None, use range(1, N+1).
        Returns a new set.
        """
        S = set(S)
        if candidates is None:
            candidates = range(1, N + 1)
        else:
            candidates = list(candidates)
        
        # Sort candidates by a heuristic: prefer numbers with higher "connectivity" (more neighbors)
        # or simply ascending order? 
        # Ascending order tends to find small numbers first, which are more compatible.
        # Let's try sorting by number of compatible neighbors in descending order.
        # Actually, just ascending is a strong baseline. Let's stick to simple iteration for speed,
        # but maybe try a few different orders in different restarts.
        
        for x in candidates:
            if x in S:
                continue
            if is_compatible(x, S):
                S.add(x)
        return S

    def local_search_add(S):
        """
        Try to add elements to S.
        If no single element can be added, try removing one element u and then greedily adding.
        If the resulting set is larger, accept.
        """
        S = set(S)
        improved = True
        while improved and not check_time():
            improved = False
            
            # Phase 1: Direct addition
            added = False
            # Try adding in ascending order
            for x in range(1, N + 1):
                if x not in S:
                    if is_compatible(x, S):
                        S.add(x)
                        improved = True
                        added = True
                        break
            if added:
                continue

            # Phase 2: 1-for-1+ swap
            # Remove one element, then greedily add as many as possible.
            elems = sorted(S)
            for u in elems:
                if u == 1:
                    continue # 1 is compatible with everything, removing it is rarely beneficial for size
                if check_time():
                    break
                
                trial = set(S)
                trial.discard(u)
                
                # Greedily add to trial
                # To be efficient, we can just iterate 1..N
                for x in range(1, N + 1):
                    if x not in trial:
                        if is_compatible(x, trial):
                            trial.add(x)
                
                if len(trial) > len(S):
                    S = trial
                    improved = True
                    break
        
        return S

    def local_search_2for3(S):
        """
        Try removing two elements and adding up to three (or more via greedy).
        """
        S = set(S)
        improved = True
        while improved and not check_time():
            improved = False
            elems = sorted(S)
            for i in range(len(elems)):
                if check_time():
                    break
                for j in range(i + 1, len(elems)):
                    if check_time():
                        break
                    u, v = elems[i], elems[j]
                    if u == 1 or v == 1:
                        continue
                    
                    trial = set(S)
                    trial.discard(u)
                    trial.discard(v)
                    
                    # Greedily add
                    for x in range(1, N + 1):
                        if x not in trial:
                            if is_compatible(x, trial):
                                trial.add(x)
                    
                    if len(trial) > len(S):
                        S = trial
                        improved = True
                        break
                if improved:
                    break
        return S

    # --- Main Search Loop ---
    best_S = set()
    best_size = 0

    # Strategy 1: Structural Seeds based on Divisors
    # Numbers with many divisors are "hubs" in the compatibility graph.
    # Starting with the divisors of a highly composite number is a strong seed.
    nums_by_divs = sorted(range(1, N + 1), key=lambda x: -len(divisors[x]))
    
    # Try top 10 highly composite numbers as seeds
    for m in nums_by_divs[:10]:
        if check_time():
            break
        seed = set(divisors[m])
        S = greedy_add(seed)
        S = local_search_add(S)
        if not check_time():
            S = local_search_2for3(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    # Strategy 2: Randomized Restart with Greedy + Local Search
    # Use different random seeds to explore different basins of attraction.
    if not check_time():
        random.seed(42)
        for _ in range(30): # Limit restarts to avoid timeout
            if check_time():
                break
            
            # Create a random seed
            # Option A: Random subset of 1..50
            # Option B: Divisors of a random number
            # Option C: Random compatible set
            
            r = random.random()
            if r < 0.3:
                # Small random seed
                seed_size = random.randint(3, 10)
                seed = set(random.sample(range(1, 50), seed_size))
            elif r < 0.6:
                # Divisors of random number
                m = random.randint(1, N)
                seed = set(divisors[m])
            else:
                # Random compatible set built greedily
                seed = set([1])
                # Pick a few random numbers and add if compatible
                for _ in range(20):
                    x = random.randint(1, N)
                    if is_compatible(x, seed):
                        seed.add(x)
            
            S = greedy_add(seed)
            S = local_search_add(S)
            if not check_time():
                S = local_search_2for3(S)
            
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)

    # Strategy 3: Aggressive local search on the best solution found
    if not check_time():
        best_S = local_search_add(best_S)
        if not check_time():
            best_S = local_search_2for3(best_S)
        
        # One more round of 1-for-1 just in case
        if not check_time():
             best_S = local_search_add(best_S)

    # Final verification
    final_list = sorted(best_S)
    if len(final_list) == 0:
        return [1]
        
    valid = True
    for i in range(len(final_list)):
        for j in range(i + 1, len(final_list)):
            if not compat[final_list[i]][final_list[j]]:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to a known valid set: divisors of 120 (or similar)
        # 120 is divisible by 1,2,3,4,5,6,8,10,12,15,20,24,30,40,60,120
        # lcm of any pair of divisors of 120 divides 120, so <= 512.
        return sorted(divisors[120])
    
    return final_list

# EVOLVE-BLOCK-END
