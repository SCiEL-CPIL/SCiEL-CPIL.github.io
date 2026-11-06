import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin

    def get_edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    edges.append([i, j])
        return edges

    def verify_full(adj, n):
        """
        Verify that no subset of size >= 6 induces a regular subgraph.
        Optimized: Check sizes 6, 7, ... in order. Stop early if found.
        """
        vertices = list(range(n))
        for k in range(6, n + 1):
            for comb in itertools.combinations(vertices, k):
                mask = 0
                for u in comb:
                    mask |= (1 << u)
                
                first_deg = -1
                is_regular = True
                for u in comb:
                    deg = bin(adj[u] & mask).count('1')
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_regular = False
                        break
                
                if is_regular:
                    return False
        return True

    def generate_1_factorization(n):
        """
        Generate a 1-factorization of K_n. n must be even.
        Returns a list of n//2 matchings, each a list of (u, v) pairs.
        """
        if n % 2 != 0:
            raise ValueError("n must be even")
        
        # Standard construction for K_2k
        # Vertices 0..2k-2 and infinity (2k-1)
        # Round i: 
        #   Edge (inf, i)
        #   For j in 1..k-1: Edge ((i+j) % (2k-1), (i-j) % (2k-1))
        
        factors = []
        for i in range(n - 1):
            matching = []
            # Edge with the "infinity" vertex (n-1)
            matching.append((i, n - 1))
            
            for j in range(1, n // 2):
                u = (i + j) % (n - 1)
                v = (i - j) % (n - 1)
                matching.append((u, v))
            factors.append(matching)
        
        return factors

    def find_valid_graph_factorization(n, time_budget):
        """
        Find a valid graph on n vertices using 1-factorization of K_n (if n even) 
        or K_{n+1} (if n odd) by removing vertices.
        """
        deadline = start_time + time_budget
        
        if n % 2 == 0:
            factors = generate_1_factorization(n)
            num_factors = len(factors)
        else:
            # Use K_{n+1} and remove one vertex
            factors_full = generate_1_factorization(n + 1)
            # We will represent the graph on n vertices (0..n-1)
            # by ignoring edges incident to n.
            # Each factor in K_{n+1} becomes a matching on n vertices (size (n-1)/2)
            # if we remove vertex n.
            factors = []
            for m in factors_full:
                new_m = []
                for u, v in m:
                    if u != n and v != n:
                        new_m.append((u, v))
                factors.append(new_m)
            num_factors = len(factors)
            
        # We want to select a random subset of these factors.
        # There are 2^num_factors possibilities.
        # We will try random subsets.
        
        for attempt in range(10000):
            if time.time() > deadline:
                break
                
            # Randomly select a subset of factors
            # Bias towards smaller subsets? Or uniform?
            # Uniform is fine.
            selected_factors = []
            for i in range(num_factors):
                if random.random() < 0.5:
                    selected_factors.append(i)
            
            # Construct graph
            adj = [0] * n
            for idx in selected_factors:
                for u, v in factors[idx]:
                    if u < n and v < n: # Safety check for odd n case
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Verify
            if verify_full(adj, n):
                return adj
                
        return None

    # Try n=21
    # 21 is odd, so we use K_22 factorization.
    # Time budget: 100s
    adj21 = find_valid_graph_factorization(21, 100.0)
    if adj21 is not None:
        return {'n': 21, 'edges': get_edges_from_adj(adj21, 21)}
        
    # Try n=20
    # 20 is even, use K_20 factorization.
    # Time budget: 40s
    adj20 = find_valid_graph_factorization(20, 40.0)
    if adj20 is not None:
        return {'n': 20, 'edges': get_edges_from_adj(adj20, 20)}
        
    # Fallback
    n = 5
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return {'n': n, 'edges': get_edges_from_adj(adj, n)}
