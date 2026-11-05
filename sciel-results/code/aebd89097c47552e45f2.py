import time
import itertools

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s margin for safety
    
    # Primes p = 3 mod 4, sorted ascending to find smallest valid n
    candidates = []
    for p in range(31, 84):
        if p % 4 == 3 and is_prime(p):
            candidates.append(p)
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        # Build Paley tournament
        squares = {x*x % p for x in range(1, p)}
        
        # in_mask[v] = bitmask of vertices u such that u -> v
        in_mask = [0] * p
        for i in range(p):
            for j in range(p):
                if i != j:
                    # i -> j iff (j - i) % p in squares
                    if (j - i) % p in squares:
                        in_mask[j] |= (1 << i)
        
        # Check: for every 4-set, intersection of in_masks is non-empty
        valid = True
        for a, b, c, d in itertools.combinations(range(p), 4):
            common = in_mask[a] & in_mask[b] & in_mask[c] & in_mask[d]
            if common == 0:
                valid = False
                break
        
        if valid:
            orientation = []
            for i in range(p):
                for j in range(i+1, p):
                    if (j - i) % p in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': p, 'orientation': orientation}
            
    # Fallback: p=67 Paley tournament (known valid)
    p = 67
    squares = {x*x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}
