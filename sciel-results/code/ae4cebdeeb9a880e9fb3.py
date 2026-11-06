import time
import math

def construct():
    start_time = time.time()
    deadline = 110  # seconds
    
    # Candidate primes p ≡ 3 (mod 4)
    # Smallest known valid n is 67. Lower bound 31.
    # We will try primes starting from 67 upwards.
    # If we find a valid one smaller than 83, we win.
    # If not, we return 83 (the seed's solution).
    
    primes = [67, 71, 79, 83, 87, 91] # 87, 91 not prime. 
    # Primes 3 mod 4:
    # 67 (3 mod 4)
    # 71 (3 mod 4)
    # 79 (3 mod 4)
    # 83 (3 mod 4)
    # 87 not prime
    # 91 not prime
    # 95 not prime
    # 103 (3 mod 4)
    
    candidates = [67, 71, 79, 83]
    
    best_n = 83
    best_orientation = None
    
    # We know 83 works (from seed). 
    # Let's verify 83 first to be safe, or just assume it.
    # Actually, let's check 67, 71, 79 first.
    
    for p in candidates:
        if time.time() - start_time > deadline:
            break
            
        # Construct Paley tournament on p vertices
        # i -> j iff (j - i) is a quadratic residue mod p
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
            
        # orientation list
        # Index k corresponds to pair (i, j) where i < j
        # Order: (0,1), (0,2), ..., (0,p-1), (1,2), ...
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    orientation.append(0) # i -> j
                else:
                    orientation.append(1) # j -> i
        
        # Check validity
        # Build adjacency matrix for faster lookup
        # adj[i][j] = 1 if i -> j else 0
        adj = [[0] * p for _ in range(p)]
        idx = 0
        for i in range(p):
            for j in range(i + 1, p):
                bit = orientation[idx]
                if bit == 0:
                    adj[i][j] = 1
                    adj[j][i] = 0
                else:
                    adj[i][j] = 0
                    adj[j][i] = 1
                idx += 1
        
        valid = True
        # Check all 4-subsets
        # To speed up, we can break early if invalid
        for i in range(p):
            if time.time() - start_time > deadline:
                valid = False
                break
            for j in range(i + 1, p):
                for k in range(j + 1, p):
                    for l in range(k + 1, p):
                        # Check vertices i, j, k, l
                        # Out-degree of i in {i,j,k,l}: adj[i][j] + adj[i][k] + adj[i][l]
                        # If any is 3, it's valid for this set.
                        
                        deg_i = adj[i][j] + adj[i][k] + adj[i][l]
                        if deg_i == 3:
                            continue
                        deg_j = adj[j][i] + adj[j][k] + adj[j][l]
                        if deg_j == 3:
                            continue
                        deg_k = adj[k][i] + adj[k][j] + adj[k][l]
                        if deg_k == 3:
                            continue
                        deg_l = adj[l][i] + adj[l][j] + adj[l][k]
                        if deg_l == 3:
                            continue
                        
                        # If none is 3, invalid
                        valid = False
                        break
                    if not valid: break
                if not valid: break
            if not valid: break
        
        if valid:
            best_n = p
            best_orientation = orientation
            # If we found a valid one smaller than 83, we can stop?
            # The task is to minimize n.
            # So we should keep checking smaller n?
            # We are iterating in increasing order: 67, 71, 79, 83.
            # So the first valid one is the smallest.
            break

    # If no valid found in candidates (or time ran out), fallback to 83
    if best_orientation is None:
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
        
    return {'n': best_n, 'orientation': best_orientation}
