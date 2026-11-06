import numpy as np
import itertools
import time
import random

# Precomputation
# 1. Patterns for 3-regular bipartite graphs on 4+4
# A 3-regular bipartite graph on 4+4 is defined by a permutation matrix P (4x4)
# such that the adjacency matrix A = J - P (where J is all-ones).
# Wait, J has 4 ones per row. We want 3. So A = J - P.
# The edges are (i, j) where A[i, j] = 1.
# So the pattern is the set of (i, j) where P[i, j] = 0.
# There are 4! = 24 permutations.

def get_patterns():
    patterns = []
    for perm in itertools.permutations(range(4)):
        # perm[i] is the column index of the 0 in row i
        # So A[i, j] = 1 if j != perm[i]
        mask = 0
        for i in range(4):
            for j in range(4):
                if j != perm[i]:
                    # Bit position: i * 4 + j
                    mask |= (1 << (i * 4 + j))
        patterns.append(mask)
    return patterns

PATTERNS = get_patterns()

# 2. LUT for 16-bit masks
# LUT[mask] = True if mask contains any pattern
LUT = [False] * (1 << 16)
for m in range(1 << 16):
    for p in PATTERNS:
        if (m & p) == p:
            LUT[m] = True
            break

# 3. Subsets and Bipartitions
# Subsets: combinations of 8 vertices from 16
SUBSETS = list(itertools.combinations(range(16), 8))

# For each subset, we need the bipartitions.
# A bipartition is a pair of sets (A, B) of 4 indices each, partitioning the subset.
# We can represent a bipartition by the 4 indices in A (sorted).
# B is the other 4.
# We need to map the local indices 0..7 to global indices.
# Let's precompute for each subset, a list of (A_global, B_global) tuples.
# A_global is a tuple of 4 global vertex indices.
# B_global is a tuple of 4 global vertex indices.

BIPARTITIONS = []
for sub in SUBSETS:
    # sub is a tuple of 8 global indices
    # We need to choose 4 of them for A.
    # The rest form B.
    parts = list(itertools.combinations(sub, 4))
    # To avoid duplicates (A, B) and (B, A), we can just take all combinations.
    # Since the graph is undirected, the bipartition (A, B) is the same as (B, A)
    # in terms of the edge set check?
    # The check is: does G[A, B] contain Q3?
    # G[A, B] is the set of edges between A and B.
    # The structure is symmetric.
    # However, the pattern check is directional (rows A, cols B).
    # But Q3 is symmetric.
    # So checking (A, B) is sufficient.
    # We have 70 combinations.
    BIPARTITIONS.append(parts)

def extract_mask(graph, A, B):
    # graph is 16x16 numpy array
    # A, B are tuples of 4 global indices
    mask = 0
    for i in range(4):
        for j in range(4):
            if graph[A[i], B[j]]:
                mask |= (1 << (i * 4 + j))
    return mask

def find_q3(graph):
    # Returns (edge1, edge2) if Q3 found, else None
    for sub in SUBSETS:
        for A, B in BIPARTITIONS[SUBSETS.index(sub)]: # This index lookup is slow
            # Better to iterate with index
            pass

# Optimization: Precompute list of (sub_idx, A, B)
# Or just iterate.
# Let's restructure.

ALL_BIPS = []
for idx, sub in enumerate(SUBSETS):
    parts = list(itertools.combinations(sub, 4))
    for A in parts:
        B = tuple(x for x in sub if x not in A)
        ALL_BIPS.append((sub, A, B))

def find_q3(graph):
    for sub, A, B in ALL_BIPS:
        mask = extract_mask(graph, A, B)
        if LUT[mask]:
            # Found a Q3
            # Return one edge from this Q3
            # We need to find which pattern matched to get the edges?
            # Or just return a random edge from the bipartite subgraph?
            # The subgraph G[A, B] contains Q3.
            # So any edge in G[A, B] is part of *some* Q3?
            # Not necessarily the one that matched.
            # But removing any edge in G[A, B] might break this specific embedding.
            # Let's return a random edge from G[A, B].
            edges = []
            for i in range(4):
                for j in range(4):
                    if graph[A[i], B[j]]:
                        edges.append((A[i], B[j]))
            if edges:
                return random.choice(edges)
    return None

def construct():
    # Start with complete graph
    graph = np.ones((16, 16), dtype=bool)
    np.fill_diagonal(graph, False)
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Greedy removal
    while True:
        if time.time() > deadline:
            break
        
        edge = find_q3(graph)
        if edge is None:
            break
        
        u, v = edge
        graph[u, v] = False
        graph[v, u] = False
    
    # Try to add edges back (Local Search)
    # Generate list of missing edges
    missing_edges = [(i, j) for i in range(16) for j in range(i+1, 16) if not graph[i, j]]
    random.shuffle(missing_edges)
    
    for u, v in missing_edges:
        if time.time() > deadline:
            break
        # Try adding
        graph[u, v] = True
        graph[v, u] = True
        
        # Check if valid
        if find_q3(graph) is not None:
            # Invalid, revert
            graph[u, v] = False
            graph[v, u] = False
        else:
            # Valid, keep
            pass
            
    # Return list of edges
    edges = []
    for i in range(16):
        for j in range(i+1, 16):
            if graph[i, j]:
                edges.append([i, j])
                
    return edges
