import time
import random

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s kill
    
    # Helper: Check if adding x to S creates a 4-AP
    def can_add(S, x):
        # x is 1st term: x, x+d, x+2d, x+3d
        max_d1 = (N - x) // 3
        for d in range(1, max_d1 + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return False
        # x is 2nd term: x-d, x, x+d, x+2d
        max_d2 = min(x - 1, (N - x) // 2)
        for d in range(1, max_d2 + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                return False
        # x is 3rd term: x-2d, x-d, x, x+d
        max_d3 = min((x - 1) // 2, N - x)
        for d in range(1, max_d3 + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                return False
        # x is 4th term: x-3d, x-2d, x-d, x
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                return False
        return True

    def has_4ap(S):
        """Check if S contains any nonconstant 4-term AP."""
        s_list = sorted(S)
        n = len(s_list)
        for i in range(n):
            for j in range(i+1, n):
                d = s_list[j] - s_list[i]
                if d == 0:
                    continue
                k = s_list[i] + 2*d
                l = s_list[i] + 3*d
                if k in S and l in S:
                    return True
        return False

    def base4_construction():
        """Generate a 4-AP-free set using base-4 digit construction."""
        S = set()
        for i in range(1, N + 1):
            # Check if i has only digits 0 and 1 in base 4
            valid = True
            n = i
            while n > 0:
                if n % 4 > 1:
                    valid = False
                    break
                n //= 4
            if valid:
                S.add(i)
        return S

    def greedy_from_seed(seed_set):
        """Run greedy from a seed set, trying to add remaining elements."""
        S = set(seed_set)
        remaining = [x for x in range(1, N + 1) if x not in S]
        random.shuffle(remaining)
        
        for x in remaining:
            if can_add(S, x):
                S.add(x)
        
        return S

    def local_search(S, max_iters=500):
        """Local search with swap operations to escape local optima."""
        S = set(S)
        current_size = len(S)
        no_improve_count = 0
        
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Try to find a swap: remove x, add y
            # Sample a few random pairs
            found_improvement = False
            for _ in range(20):
                if time.time() - start_time > time_limit - 5:
                    break
                    
                # Pick random element to remove
                if not S:
                    break
                x = random.choice(list(S))
                
                # Try to add a random element from complement
                complement = [y for y in range(1, N + 1) if y not in S]
                if not complement:
                    break
                y = random.choice(complement)
                
                # Test if removing x and adding y works
                S_test = S - {x}
                S_test.add(y)
                
                if not has_4ap(S_test) and len(S_test) > current_size:
                    S = S_test
                    current_size = len(S)
                    found_improvement = True
                    no_improve_count = 0
                    break
            
            if not found_improvement:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
        
        return S

    # Start with base-4 construction
    best_set = base4_construction()
    best_size = len(best_set)
    
    # Run greedy from the seed
    S = greedy_from_seed(best_set)
    if len(S) > best_size:
        best_size = len(S)
        best_set = set(S)
    
    # Apply local search
    S = local_search(best_set)
    if len(S) > best_size:
        best_size = len(S)
        best_set = set(S)
    
    # Run multiple random restarts with greedy
    while time.time() - start_time < time_limit - 10:
        S = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
        
        # Apply local search to this result
        S_ls = local_search(S, max_iters=200)
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_set = set(S_ls)
    
    # Final local search on the best set found
    S_final = local_search(best_set, max_iters=1000)
    if len(S_final) > best_size:
        best_set = S_final
    
    return sorted(best_set)
