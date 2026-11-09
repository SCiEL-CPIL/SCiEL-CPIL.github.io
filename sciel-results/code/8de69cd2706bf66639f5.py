import math
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 45  # Safety margin for 60s limit
    
    target = math.factorial(12)  # 479001600
    
    # Get all divisors of target
    divisors = set()
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.add(i)
            divisors.add(target // i)
    
    # Filter: must be > 12
    candidates = sorted([d for d in divisors if d > 12])
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    # Check if decomposition exists for a given max_val
    def can_decompose(rem, start_idx, max_val, memo):
        if time.time() > deadline:
            return False
        
        if rem == 1:
            return True
        
        if start_idx >= len(candidates):
            return False
            
        # Pruning: if rem is prime, it must be in the remaining candidates
        if is_prime(rem):
            if rem > max_val:
                return False
            idx = bisect.bisect_left(candidates, rem)
            if idx < len(candidates) and candidates[idx] == rem and idx >= start_idx:
                return True
            return False

        key = (rem, start_idx, max_val)
        if key in memo:
            return memo[key]
            
        # Pruning: if rem < 13, impossible
        if rem < 13:
            memo[key] = False
            return False
            
        for i in range(start_idx, len(candidates)):
            f = candidates[i]
            if f > max_val or f > rem:
                break
            if rem % f != 0:
                continue
                
            if can_decompose(rem // f, i + 1, max_val, memo):
                memo[key] = True
                return True
                
        memo[key] = False
        return False

    # Find the smallest M such that can_decompose is True
    best_M = None
    for M in candidates:
        if time.time() > deadline:
            break
        memo = {}
        if can_decompose(target, 0, M, memo):
            best_M = M
            break
            
    if best_M is None:
        return [target]
        
    # Reconstruct the solution
    def get_solution(rem, start_idx, max_val, path):
        if time.time() > deadline:
            return None
            
        if rem == 1:
            return path[:]
            
        if start_idx >= len(candidates):
            return None
            
        if is_prime(rem):
            if rem > max_val:
                return None
            idx = bisect.bisect_left(candidates, rem)
            if idx < len(candidates) and candidates[idx] == rem and idx >= start_idx:
                path.append(rem)
                return path[:]
            return None
            
        if rem < 13:
            return None
            
        for i in range(start_idx, len(candidates)):
            f = candidates[i]
            if f > max_val or f > rem:
                break
            if rem % f != 0:
                continue
                
            path.append(f)
            res = get_solution(rem // f, i + 1, max_val, path)
            if res is not None:
                return res
            path.pop()
            
        return None

    sol = get_solution(target, 0, best_M, [])
    if sol is None:
        return [target]
        
    return sol
