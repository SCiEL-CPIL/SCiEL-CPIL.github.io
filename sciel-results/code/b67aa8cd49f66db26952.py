import numpy as np
import random

def construct():
    n = 31
    target_edges = 91
    
    def greedy_build(random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            
        A = np.zeros((n, n), dtype=int)
        edges = []
        
        for _ in range(target_edges):
            A2 = A @ A
            A3 = A2 @ A
            
            best_val = float('inf')
            candidates = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if A[i][j] == 0:
                        val = A3[i][j]
                        if val < best_val:
                            best_val = val
                            candidates = [(i, j)]
                        elif val == best_val:
                            candidates.append((i, j))
                            
            # Pick randomly from best candidates to allow different structures
            i, j = random.choice(candidates)
            A[i][j] = 1
            A[j][i] = 1
            edges.append([i, j])
            
        # Count 4-cycles
        # Number of 4-cycles is (1/8) * ( (A^2)^2 )_trace - ... ?
        # Easier: (1/8) * ( (A^2 \odot A^2) )? No.
        # Number of 4-cycles = 1/8 * ( (A^2)_{ij} )?
        # Let M = A^2. M[i][j] is number of paths of length 2.
        # A 4-cycle is a pair of vertices (u,v) with 2 common neighbors?
        # No, a 4-cycle u-v-w-x-u corresponds to 2 common neighbors of u and w?
        # Actually, number of 4-cycles = 1/8 * sum_{i,j} (A^2[i][j]) * (A^2[i][j]) - ...
        # Formula: C4 = 1/8 * ( (A^2 \circ A^2)_{sum} - 2 * sum_{i} (A^2[i][i])^2 ... )
        # Simpler:
        # For each pair (i, j), let c_ij be number of common neighbors.
        # Number of 4-cycles is sum_{i<j} (c_ij choose 2) / 2? No.
        # Each 4-cycle u-v-w-x-u is counted once for pair (u,w) and once for (v,x).
        # So total 4-cycles = 1/2 * sum_{i<j} C(c_ij, 2).
        
        A2 = A @ A
        total_c4 = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = A2[i][j]
                if c >= 2:
                    total_c4 += c * (c - 1) // 2
        total_c4 //= 2
        
        return edges, total_c4

    best_edges = None
    best_score = float('inf')
    
    # Run multiple restarts
    # 120 seconds limit.
    # Each run: 91 steps * (31^3 matmul + 31^2 loop).
    # 31^3 is ~30k. 91 * 30k is ~3M ops. Very fast.
    # We can run hundreds of restarts.
    
    import time
    start_time = time.time()
    max_time = 110 # Leave buffer
    
    run_id = 0
    while time.time() - start_time < max_time:
        edges, score = greedy_build(run_id)
        if score < best_score:
            best_score = score
            best_edges = edges
        run_id += 1
        
    return best_edges
