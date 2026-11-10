import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # LCM of divisors is 30030
    LCM = 30030
    
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    # Iterate window size W from 1 upwards
    # The range will be small, so W won't be large
    for W in range(1, 500): 
        # Check all possible starting positions L modulo LCM
        # L can be any integer, but pattern repeats every LCM
        for L in range(1, LCM + 1):
            # Check if time limit is approaching
            if time.time() - start_time > timeout:
                # Fallback: return the divisors themselves
                return divisors[:]
            
            # For this window [L, L+W], find available multiples for each divisor
            available = []
            valid_window = True
            
            for i, d in enumerate(divisors):
                # Find first multiple >= L
                # k = ceil(L / d)
                k = (L + d - 1) // d
                first_mult = k * d
                
                multiples = []
                x = first_mult
                while x <= L + W:
                    multiples.append(x)
                    x += d
                
                if not multiples:
                    valid_window = False
                    break
                available.append(multiples)
            
            if not valid_window:
                continue
            
            # Now we have available[i] list of candidates for divisor i
            # We need to pick one from each list such that all picks are distinct
            # Use backtracking, sorting by list size (most constrained first)
            indices = list(range(n))
            indices.sort(key=lambda i: len(available[i]))
            
            used = set()
            choices = [0] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                
                i = indices[idx]
                for val in available[i]:
                    if val not in used:
                        used.add(val)
                        choices[i] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                
                return False
            
            if backtrack(0):
                # Reconstruct result in original order
                result = []
                for i in range(n):
                    result.append(choices[i])
                
                # Verify constraints
                # 1. Distinct
                if len(set(result)) != n:
                    continue
                # 2. Divisible
                for i in range(n):
                    if result[i] % divisors[i] != 0:
                        continue
                # 3. Positive <= 1000000
                if any(x <= 0 or x > 1000000 for x in result):
                    continue
                    
                return result
    
    # If no solution found within W limit (unlikely), return divisors
    return divisors[:]
