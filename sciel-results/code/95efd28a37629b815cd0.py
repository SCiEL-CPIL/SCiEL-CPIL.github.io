"""Rewritten program: maximize pairwise-noncoprime subset of [1, 255] containing 255.

Uses Tomita's branch-and-bound maximum clique with greedy coloring bounds,
on the reduced candidate set (numbers divisible by 3, 5, or 17),
constrained to include 255.
"""
import math
import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    n = 255
    target = 255
    
    # Candidate set: numbers in [1,255] divisible by at least one of 3,5,17
    candidates = []
    for x in range(1, n+1):
        if x % 3 == 0 or x % 5 == 0 or x % 17 == 0:
            candidates.append(x)
    
    if target not in candidates:
        candidates.append(target)
    
    m = len(candidates)
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    target_idx = val_to_idx[target]
    
    # Build adjacency as bitmasks for speed
    # adj[i] is a bitmask of neighbors
    adj = [0] * m
    for i in range(m):
        for j in range(i+1, m):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Neighbors of target (excluding target itself)
    target_neigh_mask = adj[target_idx] & ~(1 << target_idx)
    
    # List of neighbor indices
    neighbors = []
    for i in range(m):
        if i != target_idx and (target_neigh_mask >> i) & 1:
            neighbors.append(i)
    
    n_sub = len(neighbors)
    neigh_set = set(neighbors)
    
    # Build subgraph adjacency as bitmasks (local indices 0..n_sub-1)
    local_to_global = neighbors
    global_to_local = {g: l for l, g in enumerate(neighbors)}
    
    sub_adj = [0] * n_sub
    for i in range(n_sub):
        gi = local_to_global[i]
        # Neighbors of gi in the subgraph
        mask = adj[gi] & target_neigh_mask  # neighbors that are also neighbors of target
        for j in range(n_sub):
            gj = local_to_global[j]
            if (mask >> gj) & 1:
                sub_adj[i] |= (1 << j)
    
    best_clique = []
    best_size = 0
    
    def color_sort(P):
        """Greedy coloring to get an upper bound. Returns (P_order, colors) where
        P_order is P sorted by color, and colors[i] is the color of P_order[i].
        The max color gives an upper bound on clique size in P."""
        if not P:
            return [], []
        
        P = list(P)
        # Greedy coloring: repeatedly pick uncolored vertex, assign smallest available color
        # Process in order of degree for better bounds
        # Actually, Tomita's algorithm processes in a specific way.
        # Simple approach: sort by degree in P, then greedy color
        P_sorted = sorted(P, key=lambda v: bin(sub_adj[v] & _mask(P)).count('1'), reverse=True)
        
        color_of = {}
        next_color = 0
        for v in P_sorted:
            # Find smallest color not used by neighbors of v in P
            used = set()
            for u in P_sorted:
                if u == v:
                    break
                if u in color_of and (sub_adj[v] >> u) & 1:
                    used.add(color_of[u])
            c = 0
            while c in used:
                c += 1
            color_of[v] = c
            next_color = max(next_color, c + 1)
        
        # Sort by color
        P_order = sorted(P_sorted, key=lambda v: color_of[v])
        colors = [color_of[v] + 1 for v in P_order]  # 1-indexed colors
        return P_order, colors
    
    def _mask(P):
        msk = 0
        for v in P:
            msk |= (1 << v)
        return msk
    
    def expand(P, C):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if not P:
            if len(C) > best_size:
                best_size = len(C)
                best_clique = list(C)
            return
        
        # Color sort for bound
        P_order, colors = color_sort(P)
        
        # Process in reverse order (highest color first)
        for idx in range(len(P_order) - 1, -1, -1):
            if time.time() > deadline:
                return
            
            v = P_order[idx]
            c = colors[idx]
            
            # Bound: if len(C) + c <= best_size, prune
            if len(C) + c <= best_size:
                return
            
            new_C = C + [v]
            # New P: neighbors of v in current P (excluding v)
            # P without v
            P_new = [u for u in P_order if u != v]
            # Filter to neighbors of v
            P_new = [u for u in P_new if (sub_adj[v] >> u) & 1]
            
            expand(P_new, new_C)
            
            # Remove v from P
            P_order.pop(idx)
            
            if not P_order:
                break
    
    # Initial P = all neighbors
    expand(list(range(n_sub)), [])
    
    # Convert best_clique (local indices) to values
    result = [local_to_global[v] for v in best_clique]
    result = [candidates[v] for v in result]
    result.append(target)
    
    # Verify
    result = list(set(result))
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if gcd(result[i], result[j]) == 1:
                return list(range(3, 256, 3))
    
    return sorted(result)
# EVOLVE-BLOCK-END
