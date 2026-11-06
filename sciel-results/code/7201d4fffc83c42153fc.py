import time
import random
import numpy as np

def _has_k222_mask(adj, n):
    """
    Check if graph with bitmask adjacency 'adj' on n vertices contains K2,2,2.
    K2,2,2 is complete tripartite with parts of size 2.
    """
    if n < 6:
        return False
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                continue
            C = adj[u] & adj[v]
            verts_in_c = []
            mask = C
            while mask:
                lsb = mask & -mask
                v_idx = lsb.bit_length() - 1
                verts_in_c.append(v_idx)
                mask ^= lsb
            m = len(verts_in_c)
            if m < 4:
                continue
            for i in range(m):
                a = verts_in_c[i]
                for j in range(i + 1, m):
                    b = verts_in_c[j]
                    if adj[a] & (1 << b):
                        continue
                    neighbors_ab_in_c = (adj[a] & adj[b] & C) & ~(1 << a) & ~(1 << b)
                    verts_ab = []
                    mask2 = neighbors_ab_in_c
                    while mask2:
                        lsb = mask2 & -mask2
                        v_idx = lsb.bit_length() - 1
                        verts_ab.append(v_idx)
                        mask2 ^= lsb
                    k = len(verts_ab)
                    if k < 2:
                        continue
                    for x in range(k):
                        c = verts_ab[x]
                        for y in range(x + 1, k):
                            d = verts_ab[y]
                            if not (adj[c] & (1 << d)):
                                return True
    return False

def _build_adj_from_colors(colors, n):
    """Build adjacency bitmasks from color list."""
    adj = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            idx += 1
    return adj

def _count_k222_numpy(adj_matrix, n):
    """
    Vectorized count of K2,2,2 subgraphs using NumPy.
    adj_matrix: n x n boolean numpy array where adj_matrix[i,j]=True means edge exists.
    Returns count of K2,2,2 subgraphs.
    """
    if n < 6:
        return 0
    
    # Non-adjacency matrix (excluding diagonal)
    non_adj = ~adj_matrix
    np.fill_diagonal(non_adj, False)
    
    # For each pair (u,v) that is a non-edge, find common neighbors
    # Then for each pair (a,b) of non-edges among those common neighbors,
    # check if there exist (c,d) non-edges in common neighbors of all four.
    
    count = 0
    # Iterate over non-edge pairs (u,v)
    non_edge_pairs = np.argwhere(non_adj)
    non_edge_pairs = non_edge_pairs[non_edge_pairs[:, 0] < non_edge_pairs[:, 1]]
    
    for u, v in non_edge_pairs:
        # Common neighbors of u and v
        common = adj_matrix[u] & adj_matrix[v]
        common[u] = False
        common[v] = False
        common_verts = np.where(common)[0]
        m = len(common_verts)
        if m < 4:
            continue
        
        # Among common_verts, find non-edge pairs (a,b)
        sub_non_adj = non_adj[np.ix_(common_verts, common_verts)]
        ne_pairs_sub = np.argwhere(sub_non_adj)
        ne_pairs_sub = ne_pairs_sub[ne_pairs_sub[:, 0] < ne_pairs_sub[:, 1]]
        
        for ai, bi in ne_pairs_sub:
            a = common_verts[ai]
            b = common_verts[bi]
            # Candidates for (c,d): in common, non-adjacent to each other,
            # and adjacent to both a and b
            cand_mask = common & adj_matrix[a] & adj_matrix[b]
            cand_mask[a] = False
            cand_mask[b] = False
            cand_verts = np.where(cand_mask)[0]
            k = len(cand_verts)
            if k < 2:
                continue
            # Check for non-edge pair among candidates
            sub_cand_non_adj = non_adj[np.ix_(cand_verts, cand_verts)]
            if np.any(sub_cand_non_adj):
                # Count non-edge pairs in candidates
                ne_cand = np.argwhere(sub_cand_non_adj)
                ne_cand = ne_cand[ne_cand[:, 0] < ne_cand[:, 1]]
                count += len(ne_cand)
                if count > 0:
                    return count  # Early exit: we just need to know > 0
    
    return count

def _is_valid_fast(colors, n):
    """Fast validity check using bitmask method."""
    if len(colors) != n * (n - 1) // 2:
        return False
    
    adj1 = _build_adj_from_colors(colors, n)
    if _has_k222_mask(adj1, n):
        return False
    
    full_mask = (1 << n) - 1
    adj0 = [0] * n
    for i in range(n):
        adj0[i] = full_mask ^ adj1[i] ^ (1 << i)
    
    if _has_k222_mask(adj0, n):
        return False
    return True

def _paley(n, p):
    """Paley coloring using prime p >= n."""
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)
    return colors

def _local_search_vectorized(colors, n, max_iters=20000, time_limit=80):
    """
    Local search with vectorized violation counting.
    Uses NumPy to quickly evaluate the effect of flipping edges.
    """
    start = time.time()
    
    # Initial validity check
    if _is_valid_fast(colors, n):
        return True
    
    # Build adjacency matrices for both colors
    adj1 = np.zeros((n, n), dtype=bool)
    adj0 = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj1[i, j] = True
                adj1[j, i] = True
            else:
                adj0[i, j] = True
                adj0[j, i] = True
            idx += 1
    
    # Precompute edge index mapping
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[(i, j)] = idx
            idx += 1
    
    # Precompute non-adjacency for quick reference
    # Use the bitmask method for individual checks but NumPy for batch evaluation
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Pick a random edge to flip
        idx = random.randint(0, len(colors) - 1)
        i = 0
        j = 1
        # Find which edge this index corresponds to
        # Precompute list of (i,j) for each edge index
        edge_ij = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_ij.append((i, j))
        
        i, j = edge_ij[idx]
        
        # Flip
        colors[idx] = 1 - colors[idx]
        
        # Update adjacency matrices
        if colors[idx] == 1:
            adj1[i, j] = True
            adj1[j, i] = True
            adj0[i, j] = False
            adj0[j, i] = False
        else:
            adj0[i, j] = True
            adj0[j, i] = True
            adj1[i, j] = False
            adj1[j, i] = False
        
        # Quick check: count K2,2,2 in both colors
        # Use a fast approximate check first
        valid = True
        # Check color 1
        if _count_k222_numpy(adj1, n) > 0:
            valid = False
        if valid and _count_k222_numpy(adj0, n) > 0:
            valid = False
        
        if valid:
            # Confirm with exact bitmask check
            if _is_valid_fast(colors, n):
                return True
            # Revert if false positive
            colors[idx] = 1 - colors[idx]
            if colors[idx] == 1:
                adj1[i, j] = True
                adj1[j, i] = True
                adj0[i, j] = False
                adj0[j, i] = False
            else:
                adj0[i, j] = True
                adj0[j, i] = True
                adj1[i, j] = False
                adj1[j, i] = False
        else:
            # Revert
            colors[idx] = 1 - colors[idx]
            if colors[idx] == 1:
                adj1[i, j] = True
                adj1[j, i] = True
                adj0[i, j] = False
                adj0[j, i] = False
            else:
                adj0[i, j] = True
                adj0[j, i] = True
                adj1[i, j] = False
                adj1[j, i] = False
    
    return _is_valid_fast(colors, n)

def _local_search_smart(colors, n, max_iters=50000, time_limit=90):
    """
    Smarter local search: prioritize flipping edges that are in many K2,2,2 violations.
    """
    start = time.time()
    
    if _is_valid_fast(colors, n):
        return True
    
    edge_ij = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_ij.append((i, j))
    
    # Build adjacency bitmasks
    adj1 = _build_adj_from_colors(colors, n)
    full_mask = (1 << n) - 1
    adj0 = [0] * n
    for i in range(n):
        adj0[i] = full_mask ^ adj1[i] ^ (1 << i)
    
    # Track which edges are in violations
    violation_count = np.zeros(len(colors), dtype=int)
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Check current validity
        if _has_k222_mask(adj1, n) or _has_k222_mask(adj0, n):
            # Find a violation and flip an edge in it
            # For simplicity, just pick random edge weighted by violation count
            if np.sum(violation_count) > 0:
                # Weighted random selection
                probs = violation_count / np.sum(violation_count)
                idx = np.random.choice(len(colors), p=probs)
            else:
                idx = random.randint(0, len(colors) - 1)
        else:
            # Already valid?
            if _is_valid_fast(colors, n):
                return True
            idx = random.randint(0, len(colors) - 1)
        
        i, j = edge_ij[idx]
        
        # Flip
        colors[idx] = 1 - colors[idx]
        if colors[idx] == 1:
            adj1[i] |= (1 << j)
            adj1[j] |= (1 << i)
            adj0[i] &= ~(1 << j)
            adj0[j] &= ~(1 << i)
        else:
            adj0[i] |= (1 << j)
            adj0[j] |= (1 << i)
            adj1[i] &= ~(1 << j)
            adj1[j] &= ~(1 << i)
        
        # Check validity
        if not _has_k222_mask(adj1, n) and not _has_k222_mask(adj0, n):
            if _is_valid_fast(colors, n):
                return True
    
    return _is_valid_fast(colors, n)

def construct():
    deadline = time.time() + 135  # Safety margin (150s limit)
    
    # Try Paley constructions for n=29
    n_target = 29
    colors = _paley(n_target, n_target)
    if colors:
        if _is_valid_fast(colors, n_target):
            return {'n': n_target, 'colors': colors}
        # Local search from Paley start
        remaining = deadline - time.time()
        if remaining > 20:
            if _local_search_smart(colors, n_target, max_iters=30000, time_limit=min(80, remaining - 10)):
                return {'n': n_target, 'colors': colors}
    
    # Try other primes for n=29
    primes = [31, 37, 41, 43, 47, 53, 59, 61]
    for p in primes:
        if time.time() > deadline - 30:
            break
        colors = _paley(29, p)
        if colors is None:
            continue
        if _is_valid_fast(colors, 29):
            return {'n': 29, 'colors': colors}
        remaining = deadline - time.time()
        if remaining > 15:
            if _local_search_smart(colors, 29, max_iters=15000, time_limit=min(40, remaining - 10)):
                return {'n': 29, 'colors': colors}
    
    # Random restarts for n=29
    attempts = 0
    while time.time() < deadline - 40 and attempts < 15:
        attempts += 1
        colors = [random.randint(0, 1) for _ in range(29 * 28 // 2)]
        remaining = deadline - time.time()
        if remaining > 20:
            if _local_search_smart(colors, 29, max_iters=20000, time_limit=min(60, remaining - 15)):
                return {'n': 29, 'colors': colors}
    
    # Try n=28
    if time.time() < deadline - 30:
        colors = _paley(28, 29)
        if colors:
            if _is_valid_fast(colors, 28):
                return {'n': 28, 'colors': colors}
            remaining = deadline - time.time()
            if remaining > 15:
                if _local_search_smart(colors, 28, max_iters=15000, time_limit=min(40, remaining - 10)):
                    return {'n': 28, 'colors': colors}
        
        for _ in range(5):
            if time.time() > deadline - 20:
                break
            colors = [random.randint(0, 1) for _ in range(28 * 27 // 2)]
            remaining = deadline - time.time()
            if remaining > 15:
                if _local_search_smart(colors, 28, max_iters=10000, time_limit=min(30, remaining - 10)):
                    return {'n': 28, 'colors': colors}
    
    # Try n=27
    if time.time() < deadline - 25:
        colors = _paley(27, 29)
        if colors:
            if _is_valid_fast(colors, 27):
                return {'n': 27, 'colors': colors}
            remaining = deadline - time.time()
            if remaining > 15:
                if _local_search_smart(colors, 27, max_iters=10000, time_limit=min(30, remaining - 10)):
                    return {'n': 27, 'colors': colors}
        
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            colors = [random.randint(0, 1) for _ in range(27 * 26 // 2)]
            remaining = deadline - time.time()
            if remaining > 10:
                if _local_search_smart(colors, 27, max_iters=8000, time_limit=min(20, remaining - 8)):
                    return {'n': 27, 'colors': colors}
    
    # Fallback: try smaller n with Paley
    for n in range(26, 15, -1):
        if time.time() > deadline - 10:
            break
        # Find suitable prime
        p = n
        while p < 50:
            is_prime = True
            if p < 2:
                is_prime = False
            else:
                for i in range(2, int(p ** 0.5) + 1):
                    if p % i == 0:
                        is_prime = False
                        break
            if is_prime and p % 4 == 1:
                break
            p += 1
        
        colors = _paley(n, p)
        if colors:
            if _is_valid_fast(colors, n):
                return {'n': n, 'colors': colors}
            remaining = deadline - time.time()
            if remaining > 10:
                if _local_search_smart(colors, n, max_iters=5000, time_limit=min(15, remaining - 8)):
                    return {'n': n, 'colors': colors}
        
        # Random
        colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
        remaining = deadline - time.time()
        if remaining > 10:
            if _local_search_smart(colors, n, max_iters=5000, time_limit=min(15, remaining - 8)):
                return {'n': n, 'colors': colors}
    
    # Absolute fallback
    return {'n': 1, 'colors': []}
