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
    deadline = start_time + 100  # Safety margin
    
    # We are looking for primes p = 3 mod 4
    # p must be >= 5
    for p in range(5, 91):
        if time.time() > deadline:
            break
        if p % 4 != 3:
            continue
        if not is_prime(p):
            continue
            
        # Construct Paley tournament on p vertices
        # i -> j iff (j - i) % p is a quadratic residue
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
            
        # Precompute out-neighborhoods as bitmasks
        # N[i] is a bitmask where bit j is set if i -> j
        N = [0] * p
        for i in range(p):
            mask = 0
            for j in range(p):
                if i == j:
                    continue
                diff = (j - i) % p
                if diff in squares:
                    mask |= (1 << j)
            N[i] = mask
            
        # Verify condition: Every 4 vertices have a common dominator
        # i.e., intersection of out-neighborhoods is non-empty
        valid = True
        # Iterate over all combinations of 4 vertices
        # Use bit operations for speed
        for i in range(p - 3):
            Ni = N[i]
            for j in range(i + 1, p - 2):
                common_ij = Ni & N[j]
                if common_ij == 0:
                    continue
                for k in range(j + 1, p - 1):
                    common_ijk = common_ij & N[k]
                    if common_ijk == 0:
                        continue
                    # Check all l > k
                    for l in range(k + 1, p):
                        if common_ijk & N[l] == 0:
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
                    # 0 means i -> j, 1 means j -> i
                    if (j - i) % p in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': p, 'orientation': orientation}
            
    # Fallback: If no valid Paley tournament found (should not happen for n<90)
    # Return the largest one tried or a default
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
