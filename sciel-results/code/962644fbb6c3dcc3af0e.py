import time
import sys
from math import gcd
from collections import defaultdict, Counter
import random

# Problem Definition
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

def build_adjacency():
    adj = [[] for _ in range(N)]
    for u, v in EDGES:
        adj[u].append(v)
        adj[v].append(u)
    return adj

ADJ = build_adjacency()

def is_valid(coloring):
    """
    Checks if the coloring is valid:
    1. Adjacent vertices have different colors.
    2. For any pair of colors (c1, c2), the subgraph induced by vertices 
       with color c1 or c2 is a forest (no cycles).
    """
    n = len(coloring)
    
    # 1. Check adjacent differ
    for u, v in EDGES:
        if coloring[u] == coloring[v]:
            return False
            
    # 2. Group edges by color pair
    # Only edges connecting different colors matter for the bicolored subgraph
    pair_edges = defaultdict(list)
    for u, v in EDGES:
        cu, cv = coloring[u], coloring[v]
        if cu == cv:
            continue
        key = (cu, cv) if cu < cv else (cv, cu)
        pair_edges[key].append((u, v))
        
    # 3. Check for cycles in each bicolored subgraph using Union-Find
    for (c1, c2), edges in pair_edges.items():
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
            
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True
            
        for u, v in edges:
            if not union(u, v):
                # Cycle found
                return False
                
    return True

def get_best_coloring():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for return
    
    best_coloring = None
    best_score = N + 1
    
    # Strategy 1: Try Linear Colorings c(v) = (a*v + b) % k
    # This exploits the Cayley structure.
    for k in range(3, 15):
        if time.time() > deadline:
            break
        for a in range(1, k):
            if gcd(k, a) != 1:
                continue
            for b in range(k):
                if time.time() > deadline:
                    break
                coloring = [int((a * v + b) % k) for v in range(N)]
                if is_valid(coloring):
                    used = len(set(coloring))
                    if used < best_score:
                        best_score = used
                        best_coloring = coloring[:]
                    
    # Strategy 2: Greedy Coloring + Merge Reduction
    # If linear didn't find something very small, try to construct one greedily and reduce.
    if time.time() < deadline - 10:
        # Try a few random initial colorings
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Initialize with a random valid-ish coloring (start with many colors)
            # Using k=N ensures validity of "adjacent differ" if we permute, 
            # but bicolored cycles are trivially absent if all colors are distinct?
            # No, if all colors are distinct, each color pair has at most 1 edge (or 0), so it's a forest.
            # So a permutation of 0..N-1 is always valid.
            # We want to reduce this.
            coloring = list(range(N))
            
            # Shuffle to avoid getting stuck in the same local minimum
            random.shuffle(coloring)
            
            # Aggressive Merging
            # We try to merge colors. To make merging efficient, we map colors to 0..k-1
            # First, compress colors to 0..k-1
            unique = sorted(set(coloring))
            mapping = {c: i for i, c in enumerate(unique)}
            current_coloring = [mapping[c] for c in coloring]
            current_k = len(unique)
            
            # Try to reduce current_k
            improved = True
            while improved and current_k > 2:
                improved = False
                # Try merging each pair of colors (i, j) with i < j
                # Prioritize merging colors with fewer edges between them?
                # Or just iterate all pairs.
                for i in range(current_k):
                    if improved:
                        break
                    for j in range(i + 1, current_k):
                        # Create a new coloring where j becomes i
                        new_coloring = [i if c == j else c for c in current_coloring]
                        
                        # Check validity
                        if is_valid(new_coloring):
                            current_coloring = new_coloring
                            current_k -= 1
                            improved = True
                            # Renumber to keep indices compact?
                            # Not strictly necessary for is_valid, but good for logic
                            # Actually, is_valid handles any int labels.
                            break
                            
            used = len(set(current_coloring))
            if used < best_score:
                best_score = used
                best_coloring = current_coloring[:]
                
            # Early exit if we hit theoretical minimums
            if best_score <= 4:
                break

    # Strategy 3: Local Search / Simulated Annealing on Color Assignment
    # If we still have room, try to perturb the best coloring found.
    if best_coloring is not None and time.time() < deadline - 5:
        # We'll try to "swap" colors of non-adjacent vertices to see if we can merge.
        # This is complex to implement robustly in a short time.
        # The merge strategy above is quite powerful for this size.
        pass

    # Fallback: Always have a valid solution
    if best_coloring is None:
        best_coloring = list(range(N))
        best_score = N

    # Final Verification
    if not is_valid(best_coloring):
        # Should not happen, but safety net
        best_coloring = list(range(N))
        
    # Ensure types are int
    return [int(c) for c in best_coloring]

def construct():
    return get_best_coloring()

if __name__ == "__main__":
    res = construct()
    print(len(set(res)))
    print(is_valid(res))
