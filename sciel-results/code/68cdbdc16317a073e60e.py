import time
from math import gcd, lcm
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0
    
    max_n = 512
    
    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return set(divs)
    
    # Precompute compatibility
    # compatible[i][j] = True if lcm(i, j) <= max_n
    # We will use bitsets for efficiency in B&B
    # Bit i is set in adj[j] if i and j are compatible
    
    adj = [0] * (max_n + 1)
    for i in range(1, max_n + 1):
        for j in range(i + 1, max_n + 1):
            if lcm(i, j) <= max_n:
                # Set bit j in adj[i] and bit i in adj[j]
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Vertex ordering: sort by degree (popcount of adj)
    # We want to process high-degree vertices first? Or low?
    # Standard B&B often benefits from ordering.
    # Let's create a list of vertices sorted by degree descending.
    vertices = list(range(1, max_n + 1))
    degrees = [adj[v].bit_count() for v in vertices]
    # Sort vertices by degree descending
    sorted_vertices = [v for v, _ in sorted(zip(vertices, degrees), key=lambda x: x[1], reverse=True)]
    
    # Remap vertices to 0..N-1 for bitset indexing?
    # Python integers handle arbitrary size, so we can just use 1..512 as bit positions.
    # However, iterating over set bits is faster if we process them.
    
    best_clique = []
    best_size = 0
    
    # Initial lower bound from greedy to prune
    def greedy_clique():
        S = set()
        # Start with the vertex with highest degree?
        # Or just try adding all compatible
        # Let's try a simple greedy: pick v, add all neighbors, recurse?
        # Actually, just use the divisors of 512 as a strong seed
        seed = get_divisors(512)
        S = set(seed)
        # Expand
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, max_n + 1):
                if x in S:
                    continue
                if all((adj[x] >> y) & 1 for y in S):
                    S.add(x)
                    changed = True
        return S
    
    initial_seed = greedy_clique()
    best_clique = list(initial_seed)
    best_size = len(best_clique)
    
    # Bron-Kerbosch with pivot
    # R: current clique
    # P: candidates that can be added to R
    # X: excluded candidates (already processed)
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        if not P:
            return
            
        # Pivot selection: choose u in P U X that maximizes |P \ N(u)|
        # This minimizes the number of recursive calls
        P_X = P | X
        if not P_X:
            return
            
        # Find pivot u
        # We want u that maximizes the number of candidates NOT connected to u
        # i.e., maximize |P & ~adj[u]|
        # Or equivalently, minimize |P & adj[u]|
        
        # Iterate through P_X to find best pivot
        best_pivot = None
        min_count = max_n + 1
        
        # Optimization: only check a subset of pivots if P_X is large?
        # With 512 nodes, checking all is fine.
        temp_P = P
        while temp_P:
            # Extract lowest set bit
            lsb = temp_P & -temp_P
            u_bit = lsb.bit_length() - 1
            u = u_bit
            temp_P ^= lsb
            
            # Count candidates in P that are NOT neighbors of u
            # ~adj[u] needs to be masked to valid range?
            # adj[u] has bits for 1..512.
            # P has bits for 1..512.
            # non_neighbors = P & ~adj[u]
            # count = non_neighbors.bit_count()
            
            # To avoid creating large intermediate ints, we can just do:
            # count = (P & ~adj[u]).bit_count()
            
            count = (P & ~adj[u]).bit_count()
            if count < min_count:
                min_count = count
                best_pivot = u
                if count == 0:
                    break
        
        if best_pivot is None:
            return
            
        # Recurse on candidates in P \ N(best_pivot)
        # Candidates to process: P & ~adj[best_pivot]
        candidates = P & ~adj[best_pivot]
        
        while candidates:
            # Extract one candidate v
            lsb = candidates & -candidates
            v_bit = lsb.bit_length() - 1
            v = v_bit
            candidates ^= lsb
            
            # Recursive call
            # New R = R + {v}
            # New P = P & adj[v]
            # New X = X & adj[v]
            
            bron_kerbosch(R | (1 << v), P & adj[v], X & adj[v])
            
            # Remove v from P, add to X
            P &= ~(1 << v)
            X |= (1 << v)
            
            if time.time() > deadline:
                return

    # Initial P: all vertices
    # Initial X: empty
    # Initial R: empty
    
    # We can use the greedy result to set a lower bound, but B&B starts from scratch.
    # To speed up, we can start with a specific vertex?
    # No, standard B&B is fine.
    
    # To improve pruning, we can order vertices.
    # The bitset representation uses the original indices 1..512.
    # The pivot selection iterates through bits.
    
    # Let's run the search
    # P = all bits 1..512 set
    initial_P = 0
    for i in range(1, max_n + 1):
        initial_P |= (1 << i)
    
    bron_kerbosch(0, initial_P, 0)
    
    # If B&B didn't find anything better or timed out, ensure we return a valid set
    # If best_clique is empty, fallback to greedy
    if not best_clique:
        best_clique = list(initial_seed)
        
    return best_clique
