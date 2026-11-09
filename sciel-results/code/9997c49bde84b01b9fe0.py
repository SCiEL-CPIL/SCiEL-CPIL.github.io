import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    target = 479001600  # 12!
    
    # Generate all divisors of 12! that are > 12
    divisors = []
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        d = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7) * (11 ** e11)
                        if d > 12:
                            divisors.append(d)
    
    divisors = sorted(set(divisors))
    
    def find_solution_for_bound(B):
        """
        Finds a set of distinct divisors <= B that multiply to target.
        Returns the list of factors or None.
        """
        valid_divs = [d for d in divisors if d <= B]
        if not valid_divs:
            return None
            
        # Sort ascending
        valid_divs.sort()
        
        def dfs(rem_prod, used_set, current_factors):
            if time.time() > deadline:
                return None
                
            if rem_prod == 1:
                return list(current_factors)
                
            if len(current_factors) >= 12:
                return None
                
            if rem_prod < 13:
                return None
                
            # Pruning: Max product of remaining available factors
            slots_left = 12 - len(current_factors)
            
            # Find the largest 'slots_left' available divisors
            max_prod = 1
            count = 0
            for d in reversed(valid_divs):
                if d in used_set:
                    continue
                max_prod *= d
                count += 1
                if count == slots_left:
                    break
                    
            if max_prod < rem_prod:
                return None
                
            # Try each valid divisor
            # Optimization: We can skip d if d > rem_prod
            for d in valid_divs:
                if d > rem_prod:
                    break # Since sorted, no more divisors can divide rem_prod
                if d in used_set:
                    continue
                if rem_prod % d == 0:
                    used_set.add(d)
                    result = dfs(rem_prod // d, used_set, current_factors + [d])
                    used_set.remove(d)
                    if result is not None:
                        return result
                        
            return None
            
        used_set = set()
        return dfs(target, used_set, [])

    # Binary search on B
    lo = 0
    hi = len(divisors) - 1
    best_solution = None
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        B = divisors[mid]
        
        solution = find_solution_for_bound(B)
        
        if solution is not None:
            best_solution = solution
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_solution is None:
        return [target]
        
    return best_solution
