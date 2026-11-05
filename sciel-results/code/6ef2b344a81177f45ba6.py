"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    N = 512
    # Vertices are 1..511
    # Index i corresponds to value i+1
    
    weights = [1.0 / (N - (i + 1)) for i in range(N)]
    
    # Sieve primes
    is_prime = [True] * N
    is_prime[0] = is_prime[1] = False
    for i in range(2, N):
        if is_prime[i]:
            for j in range(i*2, N, i):
                is_prime[j] = False
    primes = [i for i in range(2, N) if is_prime[i]]
    
    # Conflict masks
    conflict_mask = [0] * N
    for p in primes:
        multiples = []
        for v in range(p, N, p):
            if v >= 2:
                multiples.append(v - 1)
        for i_idx in multiples:
            mask = 0
            for j_idx in multiples:
                if i_idx != j_idx:
                    mask |= (1 << j_idx)
            conflict_mask[i_idx] |= mask
            
    # Sort indices by weight descending
    indices = list(range(N))
    indices.sort(key=lambda i: -weights[i])
    
    # Greedy initial solution
    def greedy_init():
        selected = []
        used_mask = 0
        for i in indices:
            if (conflict_mask[i] & used_mask) == 0:
                selected.append(i)
                used_mask |= (1 << i)
        return selected

    init_set = greedy_init()
    best_score = sum(weights[i] for i in init_set)
    best_mask = 0
    for i in init_set:
        best_mask |= (1 << i)
        
    sys.setrecursionlimit(10000)
    
    def get_max_weight_idx(mask):
        for i in indices:
            if mask & (1 << i):
                return i
        return -1

    def color_bound(mask):
        if mask == 0:
            return 0.0
        verts = []
        for i in indices:
            if mask & (1 << i):
                verts.append(i)
                
        color_map = {}
        max_w = []
        
        for v in verts:
            forbidden = set()
            for u in verts:
                if u == v:
                    continue
                if conflict_mask[v] & (1 << u):
                    if u in color_map:
                        forbidden.add(color_map[u])
                        
            c = 0
            while c in forbidden:
                c += 1
            color_map[v] = c
            if c >= len(max_w):
                max_w.append(weights[v])
            else:
                if weights[v] > max_w[c]:
                    max_w[c] = weights[v]
                    
        return sum(max_w)

    def dfs(available_mask, current_score, current_mask):
        nonlocal best_score, best_mask
        
        if time.time() > deadline:
            return
        
        if available_mask == 0:
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
            return
        
        # Pruning
        ub = current_score + color_bound(available_mask)
        if ub <= best_score:
            return
        
        v = get_max_weight_idx(available_mask)
        if v == -1:
            return
            
        # Include v
        new_mask = available_mask & ~(1 << v) & ~conflict_mask[v]
        dfs(new_mask, current_score + weights[v], current_mask | (1 << v))
        
        # Exclude v
        dfs(available_mask & ~(1 << v), current_score, current_mask)

    all_mask = (1 << N) - 1
    dfs(all_mask, 0.0, 0)
    
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    # Validate
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return [i+1 for i in init_set]
                
    return result
# EVOLVE-BLOCK-END
