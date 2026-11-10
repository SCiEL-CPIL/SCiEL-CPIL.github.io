"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_DEADLINE = time.time() + 110  # 10s safety margin before 120s kill

def _count_popcount(x):
    """Count set bits in integer."""
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c

def _find_bad_subset(G, n):
    """
    Find a subset of size >= 6 that induces a regular subgraph.
    Returns (k, subset_list) or (0, None) if none found.
    Uses bitmask representation for speed.
    """
    # Precompute adjacency as bitmask list
    # G[i] has bit j set if edge (i,j) exists
    
    # Check subsets of size 6 first (most common bad case)
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            # Compute degree of each vertex within the subset
            degs = []
            for v in subset:
                # Count neighbors of v that are in subset
                mask = G[v]
                # Create mask for subset
                submask = 0
                for u in subset:
                    submask |= (1 << u)
                # Remove self
                submask &= ~(1 << v)
                # Count common neighbors
                d = _count_popcount(mask & submask)
                degs.append(d)
            
            if len(set(degs)) == 1:
                return k, list(subset)
    
    return 0, None

def _find_bad_subset_fast(G, n):
    """
    Optimized: precompute subset masks and check degrees more efficiently.
    Returns (k, subset_list) or (0, None).
    """
    all_vertices = list(range(n))
    
    # For efficiency, check size 6 first
    for k in range(6, n + 1):
        for subset in combinations(all_vertices, k):
            # Build mask for this subset
            submask = 0
            for u in subset:
                submask |= (1 << u)
            
            degs = []
            for v in subset:
                # Count neighbors of v within subset (excluding v itself)
                d = _count_popcount(G[v] & (submask ^ (1 << v)))
                degs.append(d)
            
            if len(set(degs)) == 1:
                return k, list(subset)
    
    return 0, None

def _is_valid(G, n):
    """Check if G has no regular induced subgraph on >= 6 vertices."""
    k, _ = _find_bad_subset_fast(G, n)
    return k == 0

def _try_local_search(n, seed, time_limit):
    """
    Local search: start with random graph, iteratively fix bad subsets.
    Strategy: 
    1. Start with random graph
    2. Find all bad subsets (or a sample)
    3. For each bad subset, flip edges to break regularity
    4. Repeat until no bad subsets or time limit
    """
    random.seed(seed)
    deadline = time.time() + time_limit
    
    # Start with random graph - bias toward sparser graphs
    # Regular subgraphs are harder to avoid in dense graphs
    p = 0.3  # Probability of edge
    G = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                G[i] |= (1 << j)
                G[j] |= (1 << i)
    
    # Check initial
    if _is_valid(G, n):
        return G
    
    # Local search iterations
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Find a bad subset
        k, subset = _find_bad_subset_fast(G, n)
        if k == 0:
            return G
        
        # Strategy: flip an edge within the bad subset to break regularity
        # Choose edge that will most likely break the regularity
        # If all degrees are equal to d, flipping an edge changes two degrees
        # We want to make at least one vertex have different degree
        
        # Compute current degrees in subset
        degs = {}
        for v in subset:
            d = _count_popcount(G[v] & ((1 << n) - 1))
            # Count within subset
            submask = 0
            for u in subset:
                submask |= (1 << u)
            d = _count_popcount(G[v] & (submask ^ (1 << v)))
            degs[v] = d
        
        # All degrees should be equal (it's a bad subset)
        common_deg = degs[subset[0]]
        
        # Try flipping edges to break regularity
        # An edge flip between u and v in subset changes:
        # - deg[u] by +/-1, deg[v] by +/-1
        # - Other vertices' degrees unchanged
        # So we can break regularity by making one vertex have different degree
        
        # Pick a random pair in subset
        pairs = list(combinations(subset, 2))
        random.shuffle(pairs)
        
        flipped = False
        for u, v in pairs[:5]:  # Try up to 5 random pairs
            # Check if flipping this edge breaks regularity
            if (G[u] >> v) & 1:
                # Edge exists, removing it
                new_degu = degs[u] - 1
                new_degv = degs[v] - 1
            else:
                # Edge doesn't exist, adding it
                new_degu = degs[u] + 1
                new_degv = degs[v] + 1
            
            # Check if this breaks regularity
            # After flip, u has new_degu, v has new_degv, others have common_deg
            if new_degu != common_deg or new_degv != common_deg:
                # This flip breaks regularity
                if (G[u] >> v) & 1:
                    G[u] &= ~(1 << v)
                    G[v] &= ~(1 << u)
                else:
                    G[u] |= (1 << v)
                    G[v] |= (1 << u)
                flipped = True
                break
        
        if not flipped:
            # Try a random flip anywhere in subset
            u, v = random.choice(pairs)
            if (G[u] >> v) & 1:
                G[u] &= ~(1 << v)
                G[v] &= ~(1 << u)
            else:
                G[u] |= (1 << v)
                G[v] |= (1 << u)
    
    # Final check
    if _is_valid(G, n):
        return G
    return None

def _to_edges(G, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (G[i] >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    start = time.time()
    best_n = 5
    best_G = [0] * 5  # empty graph on 5 vertices
    
    # Try to find larger graphs, starting from 20 and going down
    for n in range(20, 4, -1):
        if time.time() - start > 105:
            break
        
        # Try multiple seeds with different strategies
        found = False
        for seed in range(30):
            if time.time() - start > 108:
                break
            
            # Vary the initial density
            if seed < 10:
                p = 0.25
            elif seed < 20:
                p = 0.35
            else:
                p = 0.45
            
            # Temporarily modify _try_local_search behavior via seed
            G = _try_local_search(n, seed, 3.0)
            if G is not None:
                best_n = n
                best_G = G
                found = True
                break
        
        if found:
            break
    
    return {'n': best_n, 'edges': _to_edges(best_G, best_n)}
# EVOLVE-BLOCK-END
