import math

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def construct():
    # We try Paley tournaments for primes p = 3 mod 4
    # Starting from small primes to find the smallest valid n
    # Known lower bound is 31.
    
    # Primes p = 3 mod 4 in range [31, 90]
    candidates = []
    for p in range(31, 91):
        if p % 4 == 3 and is_prime(p):
            candidates.append(p)
            
    # Fallback to 83 if no smaller one works or to be safe
    # But we will check validity for each candidate
    
    def check_validity(n, orientation):
        # n is number of vertices
        # orientation is list of bits for pairs (i,j) i<j
        # Reconstruct in-neighborhoods as bitmasks
        in_neigh = [0] * n
        
        # Helper to get index in orientation list
        # Index for (i,j) i<j is sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
        # Or simpler: just iterate and fill
        
        # It's easier to build the adjacency matrix or in_neigh directly from the Paley rule
        # But the function signature requires returning orientation.
        # We can check validity during construction or after.
        # Let's construct orientation first, then check.
        
        # To speed up check, we can compute in_neigh from the Paley rule directly without storing orientation list first?
        # No, we need to return orientation.
        # But we can check validity using the Paley rule directly.
        
        # For a set {a,b,c,d}, is there v such that v->a, v->b, v->c, v->d?
        # v->u iff (u-v) % p is a square.
        # So we need (a-v) % p in squares AND (b-v) % p in squares AND ...
        
        # Let's just build the in_neigh bitmasks from the generated orientation.
        
        # Build in_neigh
        # orientation[k] corresponds to pair (i,j)
        # We need a mapping from pair (i,j) to index k.
        # k = i * (2*n - i - 1) // 2 + (j - i - 1)
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                if bit == 0:
                    # i -> j
                    in_neigh[j] |= (1 << i)
                else:
                    # j -> i
                    in_neigh[i] |= (1 << j)
                idx += 1
        
        # Check all 4-sets
        # Using itertools.combinations
        from itertools import combinations
        for a, b, c, d in combinations(range(n), 4):
            # Dominators are in in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]
            # If the intersection is non-empty, it's valid for this set.
            # Note: The intersection might contain a,b,c,d?
            # in_neigh[a] does not contain a.
            # So if v is in the intersection, v is not a,b,c,d.
            # So we just check if the AND is non-zero.
            if (in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]) == 0:
                return False
        return True

    for p in candidates:
        squares = {x * x % p for x in range(1, p)}
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        
        if check_validity(p, orientation):
            return {'n': p, 'orientation': orientation}
            
    # If none found (should not happen for p=83), return p=83
    p = 83
    squares = {x * x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}
