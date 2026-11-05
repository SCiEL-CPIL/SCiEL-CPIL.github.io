import sys
import time
from collections import defaultdict

def construct():
    N = 180
    numbers = list(range(1, N + 1))
    
    # 1. Precompute forbidden triples
    forbidden = set()
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    triple = tuple(sorted((a, b, c)))
                    forbidden.add(triple)
    
    # 2. Build conflict graph (edges between any pair in a forbidden triple)
    # neighbors[i] is a bitmask of vertices j such that {i, j, k} is a forbidden triple for some k
    neighbors = [0] * (N + 1)
    for (a, b, c) in forbidden:
        neighbors[a] |= (1 << (b - 1)) | (1 << (c - 1))
        neighbors[b] |= (1 << (a - 1)) | (1 << (c - 1))
        neighbors[c] |= (1 << (a - 1)) | (1 << (b - 1))
    
    # 3. Greedy Heuristic to find a good initial solution
    # We want a large independent set in the conflict graph.
    # Strategy: Greedy coloring / selection.
    # Let's try a few random orders and a degree-based order.
    
    def greedy_is(order):
        S = set()
        for v in order:
            # Check if v conflicts with any in S
            # Since it's a graph, v conflicts with u if u is in neighbors[v]
            # We need to check if (neighbors[v] & S_mask) == 0
            # But maintaining S_mask is easy.
            pass
        # Better implementation with bitmasks
        S_mask = 0
        for v in order:
            bit = 1 << (v - 1)
            if S_mask & neighbors[v] == 0:
                S_mask |= bit
        return S_mask

    best_mask = 0
    best_size = 0
    
    # Try descending order (large numbers first)
    order1 = list(range(N, 0, -1))
    m1 = greedy_is(order1)
    if m1.bit_count() > best_size:
        best_size = m1.bit_count()
        best_mask = m1
        
    # Try ascending order
    order2 = list(range(1, N + 1))
    m2 = greedy_is(order2)
    if m2.bit_count() > best_size:
        best_size = m2.bit_count()
        best_mask = m2
        
    # Try random orders
    import random
    random.seed(42)
    for _ in range(100):
        order = list(range(1, N + 1))
        random.shuffle(order)
        m = greedy_is(order)
        if m.bit_count() > best_size:
            best_size = m.bit_count()
            best_mask = m
            
    # 4. Branch and Bound to improve / prove optimality
    # We use the conflict graph. Any independent set is valid.
    # Upper bound: current_size + popcount(candidates)
    # This bound is weak if the graph is sparse.
    # A better bound: Greedy coloring of the subgraph induced by candidates.
    # But coloring is expensive.
    # Let's stick to the simple bound but use a good branching heuristic.
    
    start_time = time.time()
    time_limit = 50.0  # Leave 10s for overhead/verification
    
    # Convert best_mask to a list to return later
    # We need to store the best set found during B&B as well
    
    # Recursive B&B
    # To avoid recursion depth issues and overhead, we can use a stack or just careful recursion.
    # N=180, recursion depth 180 is fine.
    
    # We need to track the actual set to return, not just the size.
    # Storing the mask is sufficient.
    
    global_best_mask = best_mask
    global_best_size = best_size
    
    def solve(candidates, current_size, current_mask):
        nonlocal global_best_mask, global_best_size, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        # Pruning
        if current_size + candidates.bit_count() <= global_best_size:
            return
            
        if candidates == 0:
            if current_size > global_best_size:
                global_best_size = current_size
                global_best_mask = current_mask
            return
            
        # Heuristic: Pick a vertex with high degree in the candidate subgraph?
        # Or just pick the lowest set bit for simplicity?
        # Picking a vertex that "blocks" many others is good.
        # Let's pick the vertex v in candidates that maximizes popcount(candidates & neighbors[v])
        # This is O(N) per node, which might be slow.
        # Let's just pick the first available vertex (lowest index) for now, or highest.
        # Actually, picking a vertex with high degree in the *remaining* graph is better.
        
        # Fast approximation: Just pick the lowest set bit.
        # v_idx = (candidates & -candidates).bit_length() - 1
        # v = v_idx + 1
        
        # Let's try to be a bit smarter: Pick v such that neighbors[v] & candidates is max.
        # Iterate through set bits of candidates.
        best_v = -1
        max_conflicts = -1
        temp_cands = candidates
        while temp_cands:
            lsb = temp_cands & -temp_cands
            v_idx = lsb.bit_length() - 1
            v = v_idx + 1
            conf = (neighbors[v] & candidates).bit_count()
            if conf > max_conflicts:
                max_conflicts = conf
                best_v = v
            temp_cands ^= lsb
            
        if best_v == -1:
            return
            
        v_bit = 1 << (best_v - 1)
        
        # Branch 1: Include best_v
        # New candidates: remove best_v and its neighbors
        new_cands_include = candidates & ~(neighbors[best_v] | v_bit)
        solve(new_cands_include, current_size + 1, current_mask | v_bit)
        
        # Branch 2: Exclude best_v
        # New candidates: remove best_v
        new_cands_exclude = candidates & ~v_bit
        solve(new_cands_exclude, current_size, current_mask)
        
    # Initial call
    # Candidates: all vertices 1..N
    all_candidates = (1 << N) - 1
    # Start with empty set
    solve(all_candidates, 0, 0)
    
    # Convert best_mask to list
    result = []
    for i in range(1, N + 1):
        if global_best_mask & (1 << (i - 1)):
            result.append(i)
            
    return result

# EVOLVE-BLOCK-END
