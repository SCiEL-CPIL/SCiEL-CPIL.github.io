import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Identify candidates
    # Must contain 255.
    # Any x in S must have gcd(x, 255) > 1.
    # 255 = 3 * 5 * 17
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, 255) > 1:
            candidates.append(x)
    
    n = len(candidates)
    
    # 2. Build adjacency as bitmasks for fast intersection
    # adj_mask[i] is a bitmask where bit j is set if gcd(candidates[i], candidates[j]) > 1
    adj_mask = [0] * n
    for i in range(n):
        vi = candidates[i]
        for j in range(i + 1, n):
            vj = candidates[j]
            if math.gcd(vi, vj) > 1:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    # 3. Find Maximum Clique using Bron-Kerbosch with Pivoting
    # Using bitmasks for P, X, and R
    
    best_clique = []
    best_size = 0
    
    # Initial lower bound: multiples of 3 that are in candidates
    # All multiples of 3 have gcd >= 3 with each other and with 255
    initial_lb = [x for x in candidates if x % 3 == 0]
    best_size = len(initial_lb)
    best_clique = list(initial_lb)
    
    def bron_kerbosch(R_mask, P_mask, X_mask):
        nonlocal best_clique, best_size
        
        # Check time limit
        if time.time() > deadline:
            return
            
        # If P is empty but X is not, this branch cannot lead to a better solution
        if P_mask == 0:
            if X_mask == 0:
                # Found a maximal clique
                if bin(R_mask).count('1') > best_size:
                    best_size = bin(R_mask).count('1')
                    # Extract clique from R_mask
                    clique = [candidates[i] for i in range(n) if (R_mask >> i) & 1]
                    best_clique = clique
            return
        
        # Prune if we can't beat best_size
        if bin(R_mask).count('1') + bin(P_mask).count('1') <= best_size:
            return
        
        # Choose pivot u from P union X to minimize recursive calls
        # u should maximize |adj[u] & P|
        PX_mask = P_mask | X_mask
        if PX_mask:
            # Find pivot with max degree in P
            best_deg = -1
            pivot = -1
            temp = PX_mask
            while temp:
                v = (temp & (-temp)).bit_length() - 1
                temp &= temp - 1
                deg = bin(adj_mask[v] & P_mask).count('1')
                if deg > best_deg:
                    best_deg = deg
                    pivot = v
            
            # Iterate over P \ N(pivot)
            iter_mask = P_mask & ~(adj_mask[pivot] if pivot >= 0 else 0)
        else:
            iter_mask = P_mask
        
        # Iterate over vertices in iter_mask
        temp = iter_mask
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            temp &= temp - 1
            
            new_R = R_mask | (1 << v)
            new_P = P_mask & adj_mask[v]
            new_X = X_mask & adj_mask[v]
            
            bron_kerbosch(new_R, new_P, new_X)
            
            P_mask &= ~(1 << v)
            X_mask |= (1 << v)
            
            if time.time() > deadline:
                return
    
    # Initial call
    initial_P = (1 << n) - 1  # All candidates
    initial_X = 0
    bron_kerbosch(0, initial_P, initial_X)
    
    # Convert indices back to values
    if best_clique:
        result = sorted(best_clique)
    else:
        # Fallback to multiples of 3
        result = [i for i in range(1, 256) if i % 3 == 0]
        
    # Ensure 255 is in the result
    if 255 not in result:
        result.append(255)
        result.sort()
        
    return result
