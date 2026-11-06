"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 85  # seconds, leave 5s margin
    
    def check_deadline():
        return (time.time() - start_time) > TIME_LIMIT
    
    N = 512
    
    # Precompute divisors for all numbers up to N
    # divisors[m] = set of divisors of m
    divisors = [set() for _ in range(N + 1)]
    for d in range(1, N + 1):
        for m in range(d, N + 1, d):
            divisors[m].add(d)
            
    # Precompute LCM for all pairs? 
    # lcm[a][b] might be too big for memory if stored as 2D list of 512x512 integers? 
    # 512*512 = 262144 integers. That's fine.
    # But we can compute on the fly or precompute a boolean matrix valid_lcm[a][b] = (lcm(a,b) <= N)
    # Let's precompute a boolean compatibility matrix for speed in local search.
    # compat[i][j] = True if lcm(i,j) <= N
    compat = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            l = i // g * j
            if l <= N:
                compat[i][j] = True
                compat[j][i] = True
            else:
                compat[i][j] = False
                compat[j][i] = False
    
    def lcm(a, b):
        g = math.gcd(a, b)
        return a // g * b
        
    def is_compatible(x, s):
        """Check if x is compatible with all elements in set s"""
        for y in s:
            if not compat[x][y]:
                return False
        return True

    def greedy_build(seed_set):
        """Greedy addition of elements"""
        current = set(seed_set)
        current.add(1)
        # Sort candidates by heuristic: prefer those with smaller max LCM
        # Or just iterate and add if possible. 
        # To be deterministic and fast, we iterate 2..N and add if compatible.
        # This might not be optimal but is fast.
        # Better: sort by "potential" 
        # Let's try adding in order of increasing value first, as smaller numbers tend to have smaller LCMs.
        for x in range(2, N + 1):
            if x not in current:
                if is_compatible(x, current):
                    current.add(x)
        return current

    def local_search_1for1(current):
        """Try to replace any element with another to increase size? 
        No, 1-for-1 keeps size same. 
        We need to increase size. 
        So we should try adding an element. 
        If we can't add, we remove one and try to add two? 
        Or remove one and add one (size same) to reach a state where we can add?
        The inspiration used "greedily add" after removal. That effectively allows 1-for-k.
        Let's implement a direct 1-for-1 swap that allows subsequent additions.
        Actually, the most effective is:
        1. Try to add any element. If success, done for this round.
        2. If no element can be added, try removing an element u.
        3. After removing u, try to add as many elements as possible (greedy).
        4. If new size > old size, accept.
        """
        current = set(current)
        improved = True
        while improved and not check_deadline():
            improved = False
            # First, try to add any element
            added = False
            for x in range(2, N + 1):
                if x not in current:
                    if is_compatible(x, current):
                        current.add(x)
                        improved = True
                        added = True
                        break
            if added:
                continue
            
            # If no direct addition, try 1-for-1+ swaps
            elems = sorted(current)
            for u in elems:
                if u == 1:
                    continue
                if check_deadline():
                    break
                trial = set(current)
                trial.discard(u)
                # Greedily add to trial
                # We want to add as many as possible, so iterate 2..N
                for x in range(2, N + 1):
                    if x not in trial:
                        if is_compatible(x, trial):
                            trial.add(x)
                
                if len(trial) > len(current):
                    current = trial
                    improved = True
                    break
        
        return current

    def local_search_2for3(current):
        """Try removing 2 elements and adding up to 3"""
        current = set(current)
        improved = True
        while improved and not check_deadline():
            improved = False
            elems = sorted(current)
            for i in range(len(elems)):
                if check_deadline():
                    break
                for j in range(i + 1, len(elems)):
                    if check_deadline():
                        break
                    u, v = elems[i], elems[j]
                    if u == 1 or v == 1:
                        continue
                    trial = set(current)
                    trial.discard(u)
                    trial.discard(v)
                    # Greedily add
                    for x in range(2, N + 1):
                        if x not in trial:
                            if is_compatible(x, trial):
                                trial.add(x)
                    
                    if len(trial) > len(current):
                        current = trial
                        improved = True
                        break
                if improved:
                    break
        return current

    best_set = set()
    best_size = 0
    
    # Strategy 1: Seeds from top divisor counts
    nums_by_divs = sorted(range(1, N + 1), key=lambda x: -len(divisors[x]))
    
    # Try top 20 highly composite numbers
    for m in nums_by_divs[:20]:
        if check_deadline():
            break
        seed = set(divisors[m])
        s = greedy_build(seed)
        s = local_search_1for1(s)
        s = local_search_2for3(s)
        
        if len(s) > best_size:
            best_set = set(s)
            best_size = len(s)

    # Strategy 2: Random restarts
    if not check_deadline():
        random.seed(42)
        for _ in range(10):
            if check_deadline():
                break
            m = random.randint(1, N)
            seed = set(divisors[m])
            # Add some random compatible numbers to diversify seed
            for _ in range(5):
                x = random.randint(1, N)
                if is_compatible(x, seed):
                    seed.add(x)
            
            s = greedy_build(seed)
            s = local_search_1for1(s)
            s = local_search_2for3(s)
            
            if len(s) > best_size:
                best_set = set(s)
                best_size = len(s)

    # Final aggressive local search on the best set found
    if not check_deadline():
        best_set = local_search_1for1(best_set)
        best_set = local_search_2for3(best_set)
        
    # Verification
    final_list = sorted(best_set)
    valid = len(final_list) > 0
    for i in range(len(final_list)):
        for j in range(i + 1, len(final_list)):
            if not compat[final_list[i]][final_list[j]]:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback
        return list(range(1, int(math.isqrt(N)) + 1))
    
    return final_list

# EVOLVE-BLOCK-END
