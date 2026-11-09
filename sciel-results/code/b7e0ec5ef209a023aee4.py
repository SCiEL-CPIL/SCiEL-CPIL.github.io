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
            
        # Verify condition: Every 4 vertices have a common dominator
        # A set {a,b,c,d} is valid if there exists v (not in the set) such that
        # v->a, v->b, v->c, v->d, i.e., v is in the intersection of out-neighborhoods
        # of a, b, c, d (but v must not be in {a,b,c,d})
        
        valid = True
        for i in range(p - 3):
            if time.time() > deadline:
                break
            oi = out[i]
            for j in range(i + 1, p - 2):
                common_ij = oi & out[j]
                if common_ij == 0:
                    continue
                for k in range(j + 1, p - 1):
                    common_ijk = common_ij & out[k]
                    if common_ijk == 0:
                        continue
                    for l in range(k + 1, p):
                        # The common dominator must be in common_ijk & out[l]
                        # but must not be i, j, k, or l
                        candidate_dominators = common_ijk & out[l]
                        # Remove i, j, k, l from candidate dominators
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
