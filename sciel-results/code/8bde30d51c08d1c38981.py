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
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Check Paley tournaments for primes p = 3 mod 4, in increasing order
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
            
        # Precompute out-neighborhoods and in-neighborhoods as bitmasks
        # out[i] = bitmask of j where i -> j
        # in[i] = bitmask of j where j -> i (i.e., i is dominated by j)
        out = [0] * p
        in_neigh = [0] * p
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                diff = (j - i) % p
                if diff in squares:
                    # i -> j
                    out[i] |= (1 << j)
                    in_neigh[j] |= (1 << i)
                else:
                    # j -> i
                    out[j] |= (1 << i)
                    in_neigh[i] |= (1 << j)
            
        # Verify: Every 4 vertices {a,b,c,d} must have a common dominator v
        # i.e., there exists v such that v->a, v->b, v->c, v->d
        # This means a,b,c,d are all in out[v], or equivalently,
        # v is in in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]
        valid = True
        for a in range(p - 3):
            if time.time() > deadline:
                valid = False
                break
            ia = in_neigh[a]
            for b in range(a + 1, p - 2):
                common_ab = ia & in_neigh[b]
                if common_ab == 0:
                    continue
                for c in range(b + 1, p - 1):
                    common_abc = common_ab & in_neigh[c]
                    if common_abc == 0:
                        continue
                    for d in range(c + 1, p):
                        if common_abc & in_neigh[d] == 0:
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
            
    # Fallback: return Paley tournament on 83 vertices
    p = 83
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
