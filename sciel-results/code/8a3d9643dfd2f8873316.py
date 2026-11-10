import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    N = 512
    n = N  # Number of vertices
    
    # Build adjacency bitsets: adj[i] is a bitset where bit j is set if i and j are "incompatible" (lcm > N)
    # We want maximum independent set = maximum set with no edges
    # Use Python integers as bitsets (arbitrary precision)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # adj[i] has bit j set if lcm(i+1, j+1) > N (incompatible)
    adj = [0] * n
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if lcm(a, b) > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Bron-Kerbosch with bitsets to find maximum independent set
    # We find maximum clique in the complement graph, which is maximum independent set in original
    # Complement: two vertices are connected in complement if they are NOT connected in original
    # i.e., lcm(a,b) <= N
    
    # For maximum independent set in G, find maximum clique in complement of G
    # comp_adj[i] has bit j set if lcm(i+1, j+1) <= N (compatible)
    comp_adj = [0] * n
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if lcm(a, b) <= N:
                comp_adj[i] |= (1 << j)
                comp_adj[j] |= (1 << i)
    
    best_clique = [0] * n  # Bitset of best clique found
    best_size = 0
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        
        if P == 0 and X == 0:
            # R is a maximal clique
            r_size = bin(R).count('1')
            if r_size > best_size:
                best_size = r_size
                best_clique = R
            return
        
        # Choose pivot to minimize branching
        # Pick vertex with max degree in P
        union = P
        # Find a good pivot: vertex in P ∪ X with max neighbors in P
        # For simplicity, just pick first vertex in P
        if P:
            # Get first set bit in P
            pivot = P & (-P)
            pivot_idx = pivot.bit_length() - 1
            # Candidates are P & ~comp_adj[pivot_idx] (vertices in P not adjacent to pivot)
            candidates = P & ~comp_adj[pivot_idx]
        else:
            candidates = P
        
        while candidates:
            if time.time() > deadline:
                return
            # Pick a vertex from candidates
            v_bit = candidates & (-candidates)
            v = v_bit.bit_length() - 1
            candidates ^= v_bit
            
            # Recurse
            bron_kerbosch(R | v_bit, P & comp_adj[v], X & comp_adj[v])
            P ^= v_bit
            X |= v_bit
    
    # Run Bron-Kerbosch
    bron_kerbosch(0, (1 << n) - 1, 0)
    
    # Convert best clique bitset to list
    result = []
    for i in range(n):
        if best_clique & (1 << i):
            result.append(i + 1)
    
    if not result:
        result = [1]
    
    return result
