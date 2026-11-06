import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin before 90s kill
    
    n = 512
    values = list(range(1, n + 1))
    
    # Precompute adjacency as bitsets (Python integers)
    # adj_bits[i] is a bitmask where bit j is set if lcm(i+1, j+1) <= 512
    adj_bits = [0] * n
    for i in range(n):
        a = i + 1
        mask = 0
        for j in range(n):
            if i == j:
                continue
            b = j + 1
            l = a * b // gcd(a, b)
            if l <= 512:
                mask |= (1 << j)
        adj_bits[i] = mask
    
    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')
    
    # Helper to get highest set bit index
    def highest_bit(x):
        if x == 0:
            return -1
        return x.bit_length() - 1
    
    # Helper to get lowest set bit index
    def lowest_bit(x):
        if x == 0:
            return -1
        return (x & -x).bit_length() - 1
    
    # Initial best clique: Divisors of 360 (size 24)
    divs_360 = [d for d in range(1, 361) if 360 % d == 0]
    best_clique = set(divs_360)
    best_size = len(divs_360)
    
    # Try multiple greedy heuristics to find a larger initial clique
    def greedy_clique(order=None):
        """Greedy clique construction with optional node ordering."""
        P = (1 << n) - 1  # All nodes
        R = 0
        while P:
            # Pick node with max degree in P
            if order is not None:
                # Try nodes in given order, pick first one in P
                for v in order:
                    if P & (1 << v):
                        break
                else:
                    v = lowest_bit(P)
            else:
                # Pick node with max degree in P
                best_v = -1
                best_deg = -1
                temp = P
                while temp:
                    v = lowest_bit(temp)
                    deg = popcount(P & adj_bits[v])
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    temp &= temp - 1
                v = best_v
            
            R |= (1 << v)
            P &= adj_bits[v]
        
        sz = popcount(R)
        if sz > best_size:
            return R
        return None
    
    # Standard greedy
    g1 = greedy_clique()
    if g1:
        best_clique = set(i+1 for i in range(n) if g1 & (1 << i))
        best_size = popcount(g1)
    
    # Greedy with descending order (larger numbers first)
    order_desc = list(range(n-1, -1, -1))
    g2 = greedy_clique(order_desc)
    if g2:
        best_clique = set(i+1 for i in range(n) if g2 & (1 << i))
        best_size = popcount(g2)
    
    # Greedy with ascending order
    order_asc = list(range(n))
    g3 = greedy_clique(order_asc)
    if g3:
        best_clique = set(i+1 for i in range(n) if g3 & (1 << i))
        best_size = popcount(g3)
    
    # Bron-Kerbosch with bitsets and pivot
    def bronk(R, P, X):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        if P == 0 and X == 0:
            sz = popcount(R)
            if sz > best_size:
                best_size = sz
                best_clique = set(i+1 for i in range(n) if R & (1 << i))
            return
        if P == 0:
            return
        
        # Upper bound: can't beat best_size
        if popcount(R) + popcount(P) <= best_size:
            return
        
        # Choose pivot u from P ∪ X to minimize |P \ N(u)|
        union = P | X
        if union == 0:
            return
        
        best_pivot = -1
        best_remaining = n + 1
        temp = union
        while temp:
            u = lowest_bit(temp)
            remaining = P & ~adj_bits[u]
            rem_count = popcount(remaining)
            if rem_count < best_remaining:
                best_remaining = rem_count
                best_pivot = u
                if rem_count == 0:
                    break
            temp &= temp - 1
        
        # If P is subset of N(pivot), no candidates
        candidates = P & ~adj_bits[best_pivot]
        
        # Iterate through candidates
        temp = candidates
        while temp:
            v = lowest_bit(temp)
            bit_v = 1 << v
            bronk(R | bit_v, P & adj_bits[v], X & adj_bits[v])
            P &= ~bit_v
            X |= bit_v
            temp &= temp - 1
    
    all_nodes = (1 << n) - 1
    bronk(0, all_nodes, 0)
    
    if not best_clique:
        return [1]
    
    return list(best_clique)
