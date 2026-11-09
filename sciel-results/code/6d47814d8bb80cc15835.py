import math
import time
import random

def construct():
    N = 512
    deadline = time.time() + 55.0  # Safety margin
    
    # Precompute conflict masks
    # conflict_masks[a] is a bitmask of b such that lcm(a, b) > N
    conflict_masks = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            if a * b // math.gcd(a, b) > N:
                conflict_masks[a] |= (1 << b)
                if a != b:
                    conflict_masks[b] |= (1 << a)
    
    best_set = set(range(1, 23)) # Fallback to seed
    best_size = len(best_set)
    
    def greedy_mis(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        # Use bitmask for available nodes
        avail_mask = (1 << (N + 1)) - 2 # Bits 1..N set
        solution = set()
        
        while avail_mask:
            # Find node with min degree in available subgraph
            min_deg = float('inf')
            top_nodes = []
            
            # Iterate through available nodes
            # To speed up, we can iterate through set bits, but range(1, N+1) is fine for N=512
            for u in range(1, N + 1):
                if (avail_mask >> u) & 1:
                    deg = bin(conflict_masks[u] & avail_mask).count("1")
                    if deg < min_deg:
                        min_deg = deg
                        top_nodes = [u]
                    elif deg == min_deg:
                        top_nodes.append(u)
            
            if not top_nodes:
                break
                
            # Pick randomly from top_nodes
            chosen = random.choice(top_nodes)
            solution.add(chosen)
            
            # Remove chosen and its neighbors
            avail_mask &= ~(conflict_masks[chosen] | (1 << chosen))
        
        return solution

    def local_search(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            # Try 1-swap: replace u in S with v not in S if v < u and v conflicts only with u
            # This heuristic prefers smaller numbers which are "safer" (less likely to conflict)
            for v in range(1, N + 1):
                if v in S:
                    continue
                
                # Find conflicts of v in S
                conflicts_in_S = []
                # Iterate S
                for u in S:
                    if conflict_masks[v] & (1 << u):
                        conflicts_in_S.append(u)
                
                if len(conflicts_in_S) == 0:
                    S.add(v)
                    improved = True
                elif len(conflicts_in_S) == 1:
                    u = conflicts_in_S[0]
                    # Swap if v is smaller (safer)
                    if v < u:
                        S.remove(u)
                        S.add(v)
                        improved = True
        return S

    # Run multiple restarts
    start_time = time.time()
    restart_count = 0
    
    # Initial deterministic run
    best_set = local_search(greedy_mis(0))
    best_size = len(best_set)
    
    while time.time() < deadline:
        restart_count += 1
        # Use different seeds
        s = local_search(greedy_mis(restart_count))
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
        if time.time() > deadline:
            break
            
    return list(best_set)
