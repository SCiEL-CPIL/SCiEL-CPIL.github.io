import time
import itertools

def construct():
    start_time = time.time()
    deadline = 55.0 # Safety margin

    def time_exceeded():
        return time.time() - start_time > deadline

    def is_smooth(n, prime_set):
        if n <= 1:
            return True
        for p in prime_set:
            while n % p == 0:
                n //= p
        return n == 1

    def find_solution_greedy(prime_set, target_size=12, limit=10000):
        # Precompute smooth status for sums 2..20000
        smooth_flags = [False] * (2 * limit + 1)
        for s in range(2, 2 * limit + 1):
            if is_smooth(s, prime_set):
                smooth_flags[s] = True
        
        # Greedy Approach
        current_set = []
        x = 1
        while len(current_set) < target_size and x <= limit:
            if time_exceeded():
                return None
            
            valid = True
            for y in current_set:
                if not smooth_flags[x + y]:
                    valid = False
                    break
            
            if valid:
                current_set.append(x)
            
            x += 1
            
        if len(current_set) == target_size:
            return current_set
        return None

    def find_solution_backtrack_limited(prime_set, target_size=12, limit=10000, max_nodes=5000):
        # Precompute smooth status
        smooth_flags = [False] * (2 * limit + 1)
        for s in range(2, 2 * limit + 1):
            if is_smooth(s, prime_set):
                smooth_flags[s] = True
        
        nodes_visited = [0]
        
        def backtrack(current_set, start_val):
            if time_exceeded() or nodes_visited[0] > max_nodes:
                return None
            
            if len(current_set) == target_size:
                return current_set[:]
            
            # Optimization: If remaining slots > remaining numbers, fail
            if len(current_set) + (limit - start_val + 1) < target_size:
                return None

            for x in range(start_val, limit + 1):
                nodes_visited[0] += 1
                if time_exceeded() or nodes_visited[0] > max_nodes:
                    return None
                
                valid = True
                for y in current_set:
                    if not smooth_flags[x + y]:
                        valid = False
                        break
                
                if valid:
                    res = backtrack(current_set + [x], x + 1)
                    if res is not None:
                        return res
                        
            return None
        
        return backtrack([], 1)

    # Candidate prime sets
    # We prioritize smaller sets.
    # Within same size, smaller primes are generally better.
    prime_sets = [
        {2},
        {3},
        {2, 3},
        {5},
        {2, 5},
        {3, 5},
        {2, 3, 5},
        {7},
        {2, 7},
        {3, 7},
        {5, 7},
        {2, 3, 7},
        {2, 5, 7},
        {3, 5, 7},
        {2, 3, 5, 7}
    ]
    
    # Sort by size, then by sum of primes
    prime_sets.sort(key=lambda ps: (len(ps), sum(ps)))

    for ps in prime_sets:
        if time_exceeded():
            break
            
        # Try Greedy first (very fast)
        sol = find_solution_greedy(ps)
        
        if sol is None:
            # Try Limited Backtracking
            sol = find_solution_backtrack_limited(ps)
            
        if sol is not None and len(sol) == 12:
            if len(set(sol)) == 12 and all(1 <= x <= 10000 for x in sol):
                return sol
                
    # Fallback
    return list(range(1, 13))
