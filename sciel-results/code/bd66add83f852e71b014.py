import time
from math import factorial

_START = time.time()
_LIMIT = 80.0
_N = 12
_TARGET = factorial(_N)

def _remaining_time():
    return _LIMIT - (time.time() - _START)

def construct():
    # Precompute divisors of _TARGET
    divisors = []
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    d = (1 << e2) * (3 ** e3) * (5 ** e5) * (7 ** e7)
                    if d > _N:
                        divisors.append(d)
    divisors.sort()
    
    # Precompute primality for all divisors
    is_prime_map = {}
    for d in divisors:
        is_prime_map[d] = _is_prime(d)
        
    best_solution = None
    best_max = float('inf')
    
    def dfs(remaining, last, current):
        nonlocal best_solution, best_max
        
        if _remaining_time() <= 0:
            return False
            
        if remaining == 1:
            if 1 <= len(current) <= _N:
                mx = max(current) if current else 0
                if mx < best_max:
                    best_max = mx
                    best_solution = list(current)
            return True
            
        if len(current) >= _N:
            return False
            
        for d in divisors:
            if d <= last:
                continue
            if d > best_max:
                break
            if d > remaining:
                break
            
            if remaining % d == 0:
                rem = remaining // d
                
                if rem != 1:
                    # Necessary condition: rem must be formable by factors > d
                    # So rem > d
                    if rem <= d:
                        continue
                        
                    # If rem is composite, it needs at least 2 factors > d
                    # Min product (d+1)(d+2)
                    if not is_prime_map.get(rem, False):
                        if rem < (d + 1) * (d + 2):
                            continue
                            
                current.append(d)
                dfs(rem, d, current)
                current.pop()
                
                if _remaining_time() <= 0:
                    return False

    def _is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5)+1, 2):
            if n % i == 0: return False
        return True

    dfs(_TARGET, _N, [])
    
    if best_solution:
        return best_solution
    return [_TARGET]

if __name__ == "__main__":
    r = construct()
    print(r)
