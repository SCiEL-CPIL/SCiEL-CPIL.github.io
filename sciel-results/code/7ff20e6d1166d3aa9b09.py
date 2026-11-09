import math
import time
from math import gcd
from itertools import combinations

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, with gcd(a,b) > 1
    for all distinct pairs, maximizing cardinality.
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Build the candidate set: numbers in [1,255] that share a factor with 255
    # 255 = 3 * 5 * 17
    candidates = [x for x in range(1, 256) 
                  if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    cand_set = set(candidates)
    n = len(candidates)
    
    # Find index of 255
    idx_255 = candidates.index(255)
    
    # Build adjacency: two candidates are adjacent if gcd > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Maximum clique search with branch and bound
    # Must include 255, so start with neighbors of 255
    neighbors_of_255 = adj[idx_255]
    
    # Greedy coloring for upper bound
    # Order vertices by degree for better coloring
    def greedy_color_bound(vertex_list):
        """Compute greedy coloring upper bound for clique size in vertex_list."""
        if not vertex_list:
            return 0
        # Sort by degree within the subgraph
        vlist = sorted(vertex_list, key=lambda v: -len(adj[v] & set(vertex_list)))
        colors = {}
        max_color = 0
        for v in vlist:
            used_colors = set()
            for u in adj[v] & set(vlist):
                if u in colors:
                    used_colors.add(colors[u])
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            max_color = max(max_color, c + 1)
        return max_color
    
    best_clique = {idx_255}
    best_size = 1
    
    def branch_and_bound(clique, candidates_remaining):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if not candidates_remaining:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = set(clique)
            return
        
        # Upper bound using greedy coloring
        ub = len(clique) + greedy_color_bound(candidates_remaining)
        if ub <= best_size:
            return
        
        # Order candidates by degree in remaining subgraph
        vlist = sorted(candidates_remaining, key=lambda v: -len(adj[v] & candidates_remaining))
        
        for v in vlist:
            if time.time() > deadline:
                return
            
            # Check if v connects to all in clique
            if all(u in adj[v] for u in clique):
                new_clique = clique | {v}
                new_remaining = candidates_remaining & adj[v] - {v}
                branch_and_bound(new_clique, new_remaining)
            
            candidates_remaining.discard(v)
    
    # Initial candidates: neighbors of 255 (excluding 255 itself)
    initial_cands = neighbors_of_255.copy()
    branch_and_bound({idx_255}, initial_cands)
    
    # Convert back to values
    result = [candidates[i] for i in best_clique]
    result.sort()
    
    if 255 not in result:
        result.append(255)
    
    return result
