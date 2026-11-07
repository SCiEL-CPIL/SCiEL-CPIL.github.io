"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

n = P['n']
all_edges = set()
adj = [set() for _ in range(n)]
for u, v in P['edges']:
    all_edges.add((u, v))
    all_edges.add((v, u))
    adj[u].add(v)
    adj[v].add(u)

def expand_biclique(L, R, remaining_edges):
    """Expand a biclique greedily to cover more remaining edges."""
    L = set(L)
    R = set(R)
    changed = True
    while changed:
        changed = False
        # Try adding vertices to L
        for v in range(n):
            if v not in L:
                # v can join L if for all r in R, (v,r) is in remaining_edges
                if all((v, r) in remaining_edges for r in R):
                    # Also check it's not already in R
                    if v not in R:
                        L.add(v)
                        changed = True
        # Try adding vertices to R
        for v in range(n):
            if v not in R:
                if all((l, v) in remaining_edges for l in L):
                    if v not in L:
                        R.add(v)
                        changed = True
    return list(L), list(R)

def greedy_partition(remaining_edges, deadline):
    """Greedy: repeatedly find largest biclique covering most remaining edges."""
    bicliques = []
    rem = set(remaining_edges)
    
    while rem:
        if time.time() > deadline:
            break
        
        best_cover = 0
        best_L = None
        best_R = None
        
        # Try each remaining edge as a seed
        edges_list = list(rem)
        # Sample if too many
        if len(edges_list) > 100:
            random.shuffle(edges_list)
            edges_list = edges_list[:100]
        
        for (u, v) in edges_list:
            # Try expanding with u in L, v in R
            L, R = expand_biclique([u], [v], rem)
            # Count edges covered
            cover = 0
            for a in L:
                for b in R:
                    if (a, b) in rem:
                        cover += 1
            if cover > best_cover:
                best_cover = cover
                best_L = L
                best_R = R
        
        if best_L is None or best_cover == 0:
            # Fallback: single edge
            (u, v) = next(iter(rem))
            best_L = [u]
            best_R = [v]
        
        bicliques.append([sorted(best_L), sorted(best_R)])
        
        # Remove covered edges
        for a in best_L:
            for b in best_R:
                if (a, b) in rem:
                    rem.discard((a, b))
                if (b, a) in rem:
                    rem.discard((b, a))
    
    return bicliques, rem

def local_search(bicliques, deadline):
    """Try to merge or improve bicliques."""
    # Try merging pairs of bicliques
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(bicliques)):
            for j in range(i + 1, len(bicliques)):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                # Check if merging L1∪L2 and R1∪R2 is valid
                # i.e., all cross pairs are edges
                merged_L = set(L1) | set(L2)
                merged_R = set(R1) | set(R2)
                valid = True
                for a in merged_L:
                    for b in merged_R:
                        if (a, b) not in all_edges and a != b:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    bicliques = bicliques[:i] + bicliques[i+1:j] + bicliques[j+1:]
                    bicliques.append([sorted(merged_L), sorted(merged_R)])
                    improved = True
                    break
            if improved:
                break
    return bicliques

def construct():
    start = time.time()
    deadline = start + 80.0  # Safety margin
    
    # Strategy 1: Greedy with multiple random restarts
    best_result = None
    best_count = float('inf')
    
    # Initial greedy
    random.seed(42)
    bicliques, rem = greedy_partition(all_edges, deadline - 10)
    if not rem:
        bicliques = local_search(bicliques, deadline - 5)
        count = len(bicliques)
        if count < best_count:
            best_count = count
            best_result = bicliques
    
    # Try different random seeds for greedy
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        random.seed(seed)
        bicliques, rem = greedy_partition(all_edges, deadline - 5)
        if not rem:
            bicliques = local_search(bicliques, deadline - 2)
            count = len(bicliques)
            if count < best_count:
                best_count = count
                best_result = bicliques
    
    # Fallback: per-vertex bicliques
    if best_result is None:
        best_result = [[[u], [v for a, v in P['edges'] if a == u]] 
                       for u in range(n) if any(a == u for a, v in P['edges'])]
    
    # Convert to proper types
    result = []
    for b in best_result:
        L = list(b[0]) if isinstance(b[0], (list, tuple)) else [b[0]]
        R = list(b[1]) if isinstance(b[1], (list, tuple)) else [b[1]]
        # Ensure Python ints
        L = [int(x) for x in L]
        R = [int(x) for x in R]
        result.append([L, R])
    
    return result
# EVOLVE-BLOCK-END
