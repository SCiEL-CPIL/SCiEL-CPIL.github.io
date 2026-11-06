"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

DEADLINE = 110.0  # safety margin before 150s kill

def _precompute_partitions():
    """
    Precompute all (8-subset, partition) pairs as flat list of (A_tuple, B_tuple).
    A and B are 4-tuples of vertex indices.
    Also precompute for each partition the bit positions for fast masking.
    """
    n = 16
    all_partitions = []
    for s in combinations(range(n), 8):
        s_set = set(s)
        for a in combinations(s, 4):
            a_set = set(a)
            b = tuple(v for v in s if v not in a_set)
            # Store (a, b) where a, b are 4-tuples
            all_partitions.append((a, b))
    return all_partitions

# Precompute globally: list of (A, B) where A, B are 4-tuples
ALL_PARTITIONS = _precompute_partitions()

def _popcount(x):
    """Count set bits in integer."""
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c

def _check_partition(masks, a, b):
    """
    Check if the missing edges between A and B form a matching.
    If so, return the list of missing edge pairs. Otherwise return None.
    """
    # For each a_i in A, compute 4-bit mask of which b_j are NOT adjacent
    m0 = 0
    m1 = 0
    m2 = 0
    m3 = 0
    
    # a[0]
    av = a[0]
    am = masks[av]
    if not (am >> b[0]) & 1: m0 |= 1
    if not (am >> b[1]) & 1: m0 |= 2
    if not (am >> b[2]) & 1: m0 |= 4
    if not (am >> b[3]) & 1: m0 |= 8
    
    # a[1]
    av = a[1]
    am = masks[av]
    if not (am >> b[0]) & 1: m1 |= 1
    if not (am >> b[1]) & 1: m1 |= 2
    if not (am >> b[2]) & 1: m1 |= 4
    if not (am >> b[3]) & 1: m1 |= 8
    
    # a[2]
    av = a[2]
    am = masks[av]
    if not (am >> b[0]) & 1: m2 |= 1
    if not (am >> b[1]) & 1: m2 |= 2
    if not (am >> b[2]) & 1: m2 |= 4
    if not (am >> b[3]) & 1: m2 |= 8
    
    # a[3]
    av = a[3]
    am = masks[av]
    if not (am >> b[0]) & 1: m3 |= 1
    if not (am >> b[1]) & 1: m3 |= 2
    if not (am >> b[2]) & 1: m3 |= 4
    if not (am >> b[3]) & 1: m3 |= 8
    
    # Check each m_i has popcount <= 1
    if _popcount(m0) > 1 or _popcount(m1) > 1 or _popcount(m2) > 1 or _popcount(m3) > 1:
        return None
    
    # Check no two m_i share a bit (i.e., no b_j is missing from two a_i's)
    combined = m0 | m1 | m2 | m3
    if _popcount(combined) != _popcount(m0) + _popcount(m1) + _popcount(m2) + _popcount(m3):
        return None
    
    # Build missing edges list
    missing = []
    for i, m in enumerate((m0, m1, m2, m3)):
        if m & 1: missing.append((a[i], b[0]))
        if m & 2: missing.append((a[i], b[1]))
        if m & 4: missing.append((a[i], b[2]))
        if m & 8: missing.append((a[i], b[3]))
    
    return missing

def _contains_q3_fast(masks):
    """Check if graph contains Q3. Returns True if found."""
    for a, b in ALL_PARTITIONS:
        if _check_partition(masks, a, b) is not None:
            return True
    return False

def _find_q3_edge(masks):
    """Find an edge that is part of a Q3, or None if Q3-free."""
    for a, b in ALL_PARTITIONS:
        missing = _check_partition(masks, a, b)
        if missing is not None:
            # Find a present edge in A x B
            missing_set = set(missing)
            for i in range(4):
                for j in range(4):
                    if (a[i], b[j]) not in missing_set:
                        return (a[i], b[j])
    return None

def _adj_to_masks(adj):
    masks = [0] * 16
    for i in range(16):
        for j in range(i + 1, 16):
            if adj[i][j]:
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)
    return masks

def construct():
    start_time = time.time()
    n = 16
    deadline = start_time + DEADLINE
    
    # Start with K16
    adj = [[True] * n for _ in range(n)]
    for i in range(n):
        adj[i][i] = False
        
    masks = _adj_to_masks(adj)
    
    # Greedy removal with fast Q3 detection
    while time.time() < deadline - 5:
        edge = _find_q3_edge(masks)
        if edge is None:
            break
        u, v = edge
        # Remove edge
        adj[u][v] = adj[v][u] = False
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)
        
    # Check if we succeeded
    if _contains_q3_fast(masks):
        # Force validity by removing random edges
        while _contains_q3_fast(masks) and time.time() < deadline:
            u, v = random.sample(range(n), 2)
            if adj[u][v]:
                adj[u][v] = adj[v][u] = False
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
    
    if _contains_q3_fast(masks):
        # Fallback to empty graph
        adj = [[False] * n for _ in range(n)]
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append([i, j])
                
    return edges
# EVOLVE-BLOCK-END
