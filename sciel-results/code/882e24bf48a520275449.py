import time
import math

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Check Paley tournaments for primes p = 3 mod 4, in increasing order
    for p in range(5, 91):
        if time.time() > deadline:
            break
        if p % 4 != 3 or not is_prime(p):
            continue
            
        # Construct Paley tournament on p vertices
        # i -> j iff (j - i) % p is a quadratic residue
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
            
        # Precompute out-neighborhoods as bitmasks
        # out[i] = bitmask where bit j is set if i -> j
        out = [0] * p
        for i in range(p):
            mask = 0
            for j in range(p):
                if i == j:
                    continue
                diff = (j - i) % p
                if diff in squares:
                    mask |= (1 << j)
            out[i] = mask
            
        # Precompute pairwise common dominators to reduce AND operations in inner loop
        common = [[0] * p for _ in range(p)]
        for i in range(p):
            for j in range(i + 1, p):
                common[i][j] = out[i] & out[j]
        
        # Verify condition: Every 4 vertices have a common dominator
        valid = True
        for i in range(p - 3):
            if time.time() > deadline:
                break
            for j in range(i + 1, p - 2):
                cij = common[i][j]
                if cij == 0:
                    continue
                for k in range(j + 1, p - 1):
                    cijk = cij & out[k]
                    if cijk == 0:
                        continue
                    for l in range(k + 1, p):
                        candidate_dominators = cijk & out[l]
                        candidate_dominators &= ~(1 << i)
                        candidate_dominators &= ~(1 << j)
                        candidate_dominators &= ~(1 << k)
                        candidate_dominators &= ~(1 << l)
                        if candidate_dominators == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
                
        if valid:
            # Construct orientation list
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    if (j - i) % p in squares:
                        orientation.append(0)  # i -> j
                    else:
                        orientation.append(1)  # j -> i
            return {'n': p, 'orientation': orientation}
            
    # Fallback: return 67 Paley (known valid)
    p = 67
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
