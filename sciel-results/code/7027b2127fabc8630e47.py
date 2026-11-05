import time
import random
import math
import numpy as np
from itertools import combinations

def _gen_masks(n, k):
    """Generate all k-combinations as bitmasks."""
    masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        masks.append(mask)
    return masks

def _build_popcount(max_val):
    """Build popcount array up to max_val (exclusive)."""
    size = 1 << max_val
    popcount = np.zeros(size, dtype=np.int32)
    for i in range(1, size):
        popcount[i] = popcount[i >> 1] + (i & 1)
    return popcount

def _check_regular(adj_bits, mask, n, popcount):
    """Check if induced subgraph on mask is regular. adj_bits: list of n ints."""
    verts = [i for i in range(n) if mask & (1 << i)]
    if not verts:
        return True
    d0 = popcount[adj_bits[verts[0]] & mask]
    for v in verts[1:]:
        if popcount[adj_bits[v] & mask] != d0:
            return False
    return True

def _count_bad(adj_bits, masks, n, popcount):
    """Count number of regular induced subgraphs in the given masks."""
    count = 0
    for m in masks:
        if _check_regular(adj_bits, m, n, popcount):
            count += 1
    return count

def _full_verify(adj_bits, n, popcount):
    """Full verification: check all subsets of size >= 6."""
    for k in range(6, n + 1):
        masks = _gen_masks(n, k)
        for m in masks:
            if _check_regular(adj_bits, m, n, popcount):
                return False
    return True

def _adj_to_bits(adj, n):
    """Convert numpy adjacency matrix to list of bitmask ints."""
    bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
    return bits

def _bits_to_adj(bits, n):
    """Convert list of bitmask ints to numpy adjacency matrix."""
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i + 1, n):
            if bits[i] & (1 << j):
                adj[i, j] = 1
                adj[j, i] = 1
    return adj

def _get_edges(adj, n):
    """Get list of edges from adjacency matrix."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
    return edges

def _schreier_graph(n, seed=None):
    """
    Construct a graph using a Schreier-like approach:
    Generate random permutations and define edges based on orbit structure.
    This creates graphs with limited symmetry, reducing regular induced subgraphs.
    """
    rng = random.Random(seed)
    adj = np.zeros((n, n), dtype=np.uint8)
    
    # Generate a few random permutations
    perms = []
    for _ in range(3):
        perm = list(range(n))
        rng.shuffle(perm)
        perms.append(perm)
    
    # For each vertex, connect to images under permutations with some probability
    for i in range(n):
        for perm in perms:
            j = perm[i]
            if j > i:
                if rng.random() < 0.4:
                    adj[i, j] = 1
                    adj[j, i] = 1
            elif j < i and rng.random() < 0.2:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # Add some random extra edges to break symmetry
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.15:
                adj[i, j] = 1
                adj[j, i] = 1
    
    return adj

def _local_search(adj, n, popcount, masks_6, masks_7, deadline, rng):
    """
    Local search starting from initial adjacency matrix.
    Returns (adj, best_bad) where best_bad is the minimum bad count found.
    """
    bits = _adj_to_bits(adj, n)
    current_bad = _count_bad(bits, masks_6 + masks_7, n, popcount)
    best_adj = adj.copy()
    best_bad = current_bad
    
    T = 1.0
    cooling = 0.9995
    
    while time.time() < deadline:
        # Pick a random edge to flip
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        
        # Flip edge
        if adj[u, v] == 1:
            adj[u, v] = 0
            adj[v, u] = 0
            bits[u] &= ~(1 << v)
            bits[v] &= ~(1 << u)
        else:
            adj[u, v] = 1
            adj[v, u] = 1
            bits[u] |= (1 << v)
            bits[v] |= (1 << u)
        
        new_bad = _count_bad(bits, masks_6 + masks_7, n, popcount)
        
        if new_bad < current_bad or rng.random() < math.exp(-(new_bad - current_bad) / T):
            current_bad = new_bad
            if new_bad < best_bad:
                best_bad = new_bad
                best_adj = adj.copy()
                bits = _adj_to_bits(adj, n)
                if best_bad == 0:
                    if _full_verify(bits, n, popcount):
                        return adj, 0
        else:
            # Revert
            if adj[u, v] == 1:
                adj[u, v] = 0
                adj[v, u] = 0
                bits[u] &= ~(1 << v)
                bits[v] &= ~(1 << u)
            else:
                adj[u, v] = 1
                adj[v, u] = 1
                bits[u] |= (1 << v)
                bits[v] |= (1 << u)
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    return best_adj, best_bad

def _try_n(n, deadline, rng):
    """Try to find a valid graph on n vertices. Returns adj or None."""
    popcount = _build_popcount(n)
    masks_6 = _gen_masks(n, 6)
    masks_7 = _gen_masks(n, 7)
    
    # Try multiple random restarts
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Generate initial graph
        adj = _schreier_graph(n, seed=rng.randint(0, 2**31 - 1))
        
        # Run local search
        adj, best_bad = _local_search(adj, n, popcount, masks_6, masks_7, deadline, rng)
        
        if best_bad == 0:
            bits = _adj_to_bits(adj, n)
            if _full_verify(bits, n, popcount):
                return adj
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    rng = random.Random(42)
    
    # Try n=20 first
    for n in [20, 19, 18, 17, 16, 15]:
        if time.time() > deadline - 5:
            break
        
        adj = _try_n(n, deadline, rng)
        if adj is not None:
            edges = _get_edges(adj, n)
            return {'n': n, 'edges': edges}
    
    # Fallback: construct a small valid graph
    # n=6: Any non-regular graph on 6 vertices is valid (only the whole graph needs checking)
    n = 6
    adj = np.zeros((n, n), dtype=np.uint8)
    # Path graph P6: degrees 1,2,2,2,2,1 - not regular
    for i in range(n - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    
    # Verify
    popcount = _build_popcount(n)
    bits = _adj_to_bits(adj, n)
    if _full_verify(bits, n, popcount):
        edges = _get_edges(adj, n)
        return {'n': n, 'edges': edges}
    
    # Absolute fallback
    return {'n': 1, 'edges': []}

if __name__ == '__main__':
    res = construct()
    print(res)
