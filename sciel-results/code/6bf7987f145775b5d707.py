import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    target = math.factorial(12)
    
    # Prime factorization of 12!
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    
    # Generate all divisors > 12
    divisors = []
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        d = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7) * (11 ** e11)
                        if d > 12 and d < target: # d < target to avoid single factor being the target itself? 
                            # Actually if d == target, it's a valid 1-factor solution.
                            # But we want to minimize max. 1-factor solution max is target.
                            # We can include it.
                            divisors.append(d)
                        elif d == target:
                            divisors.append(d)
    
    # Remove duplicates? The generation is unique for each exponent tuple, so no duplicates.
    divisors.sort()
    
    # Map divisor to its exponent vector
    # For speed, we can just compute exponents on the fly or precompute.
    # Precompute list of (value, exponents_tuple)
    cand_info = []
    for d in divisors:
        rem = d
        e_vec = []
        for p in primes:
            c = 0
            while rem % p == 0:
                rem //= p
                c += 1
            e_vec.append(c)
        # rem should be 1
        cand_info.append((d, tuple(e_vec)))
        
    n_cand = len(cand_info)
    
    # DP Feasibility Check for a given max_val M
    # Returns the list of factors if found, else None
    def find_decomposition(max_val):
        # Filter candidates <= max_val
        active_cands = []
        for i in range(n_cand):
            if cand_info[i][0] <= max_val:
                active_cands.append(cand_info[i])
            else:
                break
        
        if not active_cands:
            return None
        
        # DP State: (idx, e2, e3, e5, e7, e11, count)
        # We want to reach (len(active_cands), 0, 0, 0, 0, 0, c)
        # We use a dict to store visited states and the path (or just feasibility first)
        # Given the small size, we can store the path in the DP state or use backtracking.
        # Let's use a set for visited states to avoid re-exploration, and a parent dict for reconstruction.
        
        # State: (idx, e2, e3, e5, e7, e11, count)
        # Start state: (0, 10, 5, 2, 1, 1, 0)
        # Target: (len(active_cands), 0, 0, 0, 0, 0, c) for any c in 1..12
        
        visited = set()
        parent = {} # state -> (prev_state, choice) choice: 0=exclude, 1=include
        
        start_state = (0, 10, 5, 2, 1, 1, 0)
        visited.add(start_state)
        
        # We use a queue or stack for BFS/DFS? 
        # DFS is fine.
        stack = [start_state]
        
        while stack:
            state = stack.pop()
            idx, e2, e3, e5, e7, e11, count = state
            
            if idx == len(active_cands):
                if e2 == 0 and e3 == 0 and e5 == 0 and e7 == 0 and e11 == 0 and count > 0:
                    # Found a solution
                    # Reconstruct path
                    path = []
                    curr = state
                    while curr != start_state:
                        prev, choice = parent[curr]
                        if choice == 1:
                            # We included active_cands[prev_idx]
                            # prev_idx is curr_idx - 1
                            prev_idx = curr[0] - 1
                            path.append(active_cands[prev_idx][0])
                        curr = prev
                    path.reverse()
                    return path
                continue
            
            # Pruning
            if count > 12:
                continue
                
            # Check if remaining product is 1
            if e2 == 0 and e3 == 0 and e5 == 0 and e7 == 0 and e11 == 0:
                 # We have reached product 1. 
                 # We can stop here. Any further exclusions are fine.
                 # But we need to ensure we don't add more factors.
                 # So we just need to reach the end of the list with count > 0.
                 # The loop above handles the end condition.
                 # We can prune here if we just want feasibility, but for path reconstruction
                 # we need to get to the end. 
                 # Actually, if rem is 1, we must exclude all remaining.
                 # So we can jump to (len(active_cands), 0,0,0,0,0, count).
                 target_state = (len(active_cands), 0, 0, 0, 0, 0, count)
                 if target_state not in visited:
                     visited.add(target_state)
                     parent[target_state] = (state, 0) # Choice 0 (exclude) effectively, but we jumped
                     stack.append(target_state)
                 continue

            # Option 1: Exclude active_cands[idx]
            next_idx = idx + 1
            state_excl = (next_idx, e2, e3, e5, e7, e11, count)
            if state_excl not in visited:
                visited.add(state_excl)
                parent[state_excl] = (state, 0)
                stack.append(state_excl)
                
            # Option 2: Include active_cands[idx]
            val, vec = active_cands[idx]
            # Check if val divides remaining product
            if vec[0] <= e2 and vec[1] <= e3 and vec[2] <= e5 and vec[3] <= e7 and vec[4] <= e11:
                ne2 = e2 - vec[0]
                ne3 = e3 - vec[1]
                ne5 = e5 - vec[2]
                ne7 = e7 - vec[3]
                ne11 = e11 - vec[4]
                state_incl = (next_idx, ne2, ne3, ne5, ne7, ne11, count + 1)
                if state_incl not in visited:
                    visited.add(state_incl)
                    parent[state_incl] = (state, 1)
                    stack.append(state_incl)
                    
            if time.time() > deadline:
                return None
                
        return None

    # Binary search on M
    lo = 0
    hi = n_cand - 1
    best_solution = None
    
    # We can also just iterate M from smallest to largest, but binary search is faster.
    # However, the feasibility is monotonic: if M works, M+1 works.
    # So binary search is appropriate.
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        M = divisors[mid]
        
        sol = find_decomposition(M)
        
        if sol is not None:
            best_solution = sol
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_solution is None:
        return [target]
        
    return best_solution
