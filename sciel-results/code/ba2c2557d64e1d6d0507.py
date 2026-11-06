"""K4-free, no independent 6-set graph maximizer.
Strategy: Constraint-programming-style backtracking search that constructs vertices sequentially,
maintaining the set of possible adjacency patterns for the new vertex that do not complete a K4
or an independent 6-set.
"""
import time
import random
import sys

DEADLINE = 140  # safety margin before 150s kill


def _has_k4(adj, n):
    """Check if graph has a K4. adj is list of int bitmasks."""
    for i in range(n):
        for j in range(i + 1, n):
            if not ((adj[i] >> j) & 1):
                continue
            for k in range(j + 1, n):
                if not ((adj[i] >> k) & 1 and (adj[j] >> k) & 1):
                    continue
                common = adj[i] & adj[j] & adj[k]
                mask = common & ~(1 << i) & ~(1 << j) & ~(1 << k)
                if mask:
                    return True
    return False


def _has_independent_6(adj, n):
    """Check if graph has an independent set of 6."""
    comp = [((1 << n) - 1) ^ adj[i] ^ (1 << i) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not ((comp[i] >> j) & 1):
                continue
            for k in range(j + 1, n):
                if not ((comp[i] >> k) & 1 and (comp[j] >> k) & 1):
                    continue
                for l in range(k + 1, n):
                    if not ((comp[i] >> l) & 1 and (comp[j] >> l) & 1 and (comp[k] >> l) & 1):
                        continue
                    common_ijk = comp[i] & comp[j] & comp[k] & comp[l]
                    mask = common_ijk & ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
                    bits = []
                    m = mask
                    while m:
                        b = m & (-m)
                        bits.append(b.bit_length() - 1)
                        m ^= b
                    for a in range(len(bits)):
                        for b in range(a + 1, len(bits)):
                            va, vb = bits[a], bits[b]
                            if (comp[va] >> vb) & 1:
                                return True
    return False


def _validate(adj, n):
    if n < 1 or n > 40:
        return False
    if _has_k4(adj, n):
        return False
    if _has_independent_6(adj, n):
        return False
    return True


def _count_k4s(adj, n, max_count=10000):
    """Count K4s, stop at max_count."""
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not ((adj[i] >> j) & 1):
                continue
            for k in range(j + 1, n):
                if not ((adj[i] >> k) & 1 and (adj[j] >> k) & 1):
                    continue
                common = adj[i] & adj[j] & adj[k]
                mask = common & ~(1 << i) & ~(1 << j) & ~(1 << k)
                count += mask.bit_count()
                if count >= max_count:
                    return count
    return count


def _count_independent_6(adj, n, max_count=10000):
    """Count independent 6-sets, stop at max_count."""
    comp = [((1 << n) - 1) ^ adj[i] ^ (1 << i) for i in range(n)]
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not ((comp[i] >> j) & 1):
                continue
            for k in range(j + 1, n):
                if not ((comp[i] >> k) & 1 and (comp[j] >> k) & 1):
                    continue
                for l in range(k + 1, n):
                    if not ((comp[i] >> l) & 1 and (comp[j] >> l) & 1 and (comp[k] >> l) & 1):
                        continue
                    common_ijk = comp[i] & comp[j] & comp[k] & comp[l]
                    mask = common_ijk & ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
                    bits = []
                    m = mask
                    while m:
                        b = m & (-m)
                        bits.append(b.bit_length() - 1)
                        m ^= b
                    for a in range(len(bits)):
                        for b in range(a + 1, len(bits)):
                            va, vb = bits[a], bits[b]
                            if (comp[va] >> vb) & 1:
                                count += 1
                                if count >= max_count:
                                    return count
    return count


def _score(adj, n):
    """Score: negative count of K4s and independent 6-sets. Higher is better."""
    k4s = _count_k4s(adj, n, max_count=10000)
    ind6 = _count_independent_6(adj, n, max_count=10000)
    return -(k4s * 10 + ind6 * 10)


def _local_search(adj, n, time_limit, seed=42):
    """Local search via edge flips."""
    t0 = time.time()
    random.seed(seed)
    
    current_score = _score(adj, n)
    best_adj = adj[:]
    best_score = current_score
    
    # Edge list for random selection
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for iteration in range(5000):
        if time.time() - t0 > time_limit:
            break
        
        # Pick a random edge to flip
        u, v = random.choice(all_edges)
        
        # Flip the edge
        if (adj[u] >> v) & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        new_score = _score(adj, n)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_adj = adj[:]
        else:
            # Revert
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Early stop if valid
        if current_score == 0:
            break
    
    return best_adj, best_score


def _construct_random(n, density, seed):
    """Construct a random graph with given density."""
    random.seed(seed)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < density:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _spectral_initial(n, seed):
    """Generate initial graph using spectral method."""
    import numpy as np
    np.random.seed(seed)
    # Random symmetric matrix
    A = np.random.randn(n, n)
    A = (A + A.T) / 2
    # Threshold at median
    median_val = np.median(A)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > median_val:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _greedy_fill(adj, n, time_limit, seed):
    """Greedy fill: add edges where safe."""
    t0 = time.time()
    random.seed(seed)
    
    for iteration in range(1000):
        if time.time() - t0 > time_limit:
            break
        
        # Find an edge that can be added without creating K4
        added = False
        for _ in range(20):
            u, v = random.sample(range(n), 2)
            if (adj[u] >> v) & 1:
                continue
            
            # Check if adding edge creates K4
            # A K4 is created if there exist i, j such that:
            # u-v, u-i, u-j, v-i, v-j, i-j all exist
            # So check common neighbors of u and v that are adjacent to each other
            common = adj[u] & adj[v]
            mask = common & ~(1 << u) & ~(1 << v)
            # Check if any two in mask are adjacent
            creates_k4 = False
            m = mask
            while m:
                b1 = m & (-m)
                i = b1.bit_length() - 1
                m ^= b1
                rest = m
                while rest:
                    b2 = rest & (-rest)
                    j = b2.bit_length() - 1
                    rest ^= b2
                    if (adj[i] >> j) & 1:
                        creates_k4 = True
                        break
                if creates_k4:
                    break
            
            if not creates_k4:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added = True
                break
    
    return adj


def _backtracking_construct(n, time_limit, seed=42):
    """
    Constraint-programming-style backtracking search that constructs vertices sequentially.
    For each new vertex, we try all possible adjacency patterns to previous vertices,
    pruning patterns that would create a K4 or independent 6-set.
    """
    t0 = time.time()
    random.seed(seed)
    
    # adj[i] is a bitmask of neighbors for vertex i
    adj = [0] * n
    
    # We'll build vertex by vertex. For vertex v, we decide which previous vertices it connects to.
    # A pattern is a bitmask of which previous vertices (0..v-1) are neighbors.
    
    # To check if adding vertex v with pattern P creates a K4:
    # A K4 is formed if there exist three vertices a, b, c < v such that:
    # - v is connected to a, b, c (i.e., a, b, c are in P)
    # - a, b, c form a triangle in the existing graph
    # So we need to check: among the vertices in P, is there a triangle?
    
    # To check if adding vertex v with pattern P creates an independent 6-set:
    # An independent 6-set is formed if there exist five vertices a, b, c, d, e < v such that:
    # - v is NOT connected to a, b, c, d, e (i.e., a, b, c, d, e are NOT in P)
    # - a, b, c, d, e form an independent set in the existing graph
    # So we need to check: among the vertices NOT in P, is there an independent 5-set?
    
    # For efficiency, we'll precompute:
    # - For each set of previous vertices, whether they form a triangle (for K4 check)
    # - For each set of previous vertices, whether they form an independent set (for ind6 check)
    
    # But this is exponential. Instead, we'll do the check on-the-fly.
    
    def _creates_k4(v, pattern):
        """Check if connecting vertex v to pattern creates a K4."""
        # Get the vertices in pattern
        verts = []
        m = pattern
        while m:
            b = m & (-m)
            verts.append(b.bit_length() - 1)
            m ^= b
        
        # Check if any three of these form a triangle
        for i in range(len(verts)):
            a = verts[i]
            for j in range(i + 1, len(verts)):
                b = verts[j]
                if not ((adj[a] >> b) & 1):
                    continue
                for k in range(j + 1, len(verts)):
                    c = verts[k]
                    if ((adj[a] >> c) & 1) and ((adj[b] >> c) & 1):
                        return True
        return False
    
    def _creates_ind6(v, pattern):
        """Check if connecting vertex v to pattern creates an independent 6-set."""
        # Get the vertices NOT in pattern
        all_prev = (1 << v) - 1
        not_in_pattern = all_prev ^ pattern
        
        verts = []
        m = not_in_pattern
        while m:
            b = m & (-m)
            verts.append(b.bit_length() - 1)
            m ^= b
        
        # Check if any five of these form an independent set
        # An independent set means no edges between them
        for i in range(len(verts)):
            a = verts[i]
            for j in range(i + 1, len(verts)):
                b = verts[j]
                if (adj[a] >> b) & 1:
                    continue
                for k in range(j + 1, len(verts)):
                    c = verts[k]
                    if (adj[a] >> c) & 1 or (adj[b] >> c) & 1:
                        continue
                    for l in range(k + 1, len(verts)):
                        d = verts[l]
                        if (adj[a] >> d) & 1 or (adj[b] >> d) & 1 or (adj[c] >> d) & 1:
                            continue
                        for m_idx in range(l + 1, len(verts)):
                            e = verts[m_idx]
                            if (adj[a] >> e) & 1 or (adj[b] >> e) & 1 or (adj[c] >> e) & 1 or (adj[d] >> e) & 1:
                                continue
                            return True
        return False
    
    # Backtracking: for each vertex, try all patterns
    # We'll use a recursive approach with pruning
    
    def backtrack(v):
        if time.time() - t0 > time_limit:
            return False
        
        if v == n:
            return True
        
        # Try all possible patterns for vertex v
        # Pattern is a bitmask of which previous vertices are neighbors
        # There are 2^v possible patterns
        
        # For efficiency, we'll randomize the order of patterns
        all_patterns = list(range(1 << v))
        random.shuffle(all_patterns)
        
        for pattern in all_patterns:
            if time.time() - t0 > time_limit:
                return False
            
            # Check if this pattern creates a K4
            if _creates_k4(v, pattern):
                continue
            
            # Check if this pattern creates an independent 6-set
            if _creates_ind6(v, pattern):
                continue
            
            # Apply the pattern
            for i in range(v):
                if (pattern >> i) & 1:
                    adj[v] |= (1 << i)
                    adj[i] |= (1 << v)
                else:
                    adj[v] &= ~(1 << i)
                    adj[i] &= ~(1 << v)
            
            # Recurse
            if backtrack(v + 1):
                return True
            
            # Undo the pattern
            for i in range(v):
                adj[v] &= ~(1 << i)
                adj[i] &= ~(1 << v)
        
        return False
    
    # Start backtracking
    success = backtrack(0)
    
    if success:
        return adj
    else:
        return None


def construct():
    t0 = time.time()
    best_graph = None
    best_n = 0
    best_score = -10**9
    
    # Strategy 1: Backtracking construction
    for n in [36, 35, 34, 33, 32]:
        if time.time() - t0 > DEADLINE - 30:
            break
        
        time_budget = min(30.0, DEADLINE - 30 - (time.time() - t0))
        if time_budget < 5.0:
            break
        
        # Try backtracking with different seeds
        for seed in range(10):
            if time.time() - t0 > DEADLINE - 30:
                break
            
            time_per_seed = min(5.0, DEADLINE - 30 - (time.time() - t0))
            if time_per_seed < 1.0:
                break
            
            adj = _backtracking_construct(n, time_per_seed, seed)
            
            if adj is not None:
                # Validate
                if _validate(adj, n):
                    if n > best_n:
                        best_n = n
                        best_score = _score(adj, n)
                        edge_set = set()
                        for i in range(n):
                            for j in range(i + 1, n):
                                if (adj[i] >> j) & 1:
                                    edge_set.add((i, j))
                        best_graph = {'n': n, 'edges': sorted(edge_set)}
                        # Found a valid graph, try larger n next
                        break
        
        # If we found a valid graph for this n, try larger n
        if best_graph is not None and best_n >= n:
            continue
    
    # If backtracking didn't find anything, fall back to random + local search
    if best_graph is None or best_n < 35:
        # Strategy 2: Random + local search
        for density in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for seed in range(5):
                if time.time() - t0 > DEADLINE - 20:
                    break
                
                time_budget = min(5.0, DEADLINE - 20 - (time.time() - t0))
                if time_budget < 1.0:
                    break
                
                n = 36
                
                adj = _construct_random(n, density, seed)
                adj, score = _local_search(adj, n, time_budget, seed)
                
                if score == 0:
                    if n > best_n:
                        best_n = n
                        best_score = score
                        edge_set = set()
                        for i in range(n):
                            for j in range(i + 1, n):
                                if (adj[i] >> j) & 1:
                                    edge_set.add((i, j))
                        best_graph = {'n': n, 'edges': sorted(edge_set)}
                
                # Also try smaller sizes
                if best_n < 35:
                    for smaller_n in [35, 34, 33]:
                        if smaller_n <= best_n:
                            continue
                        if time.time() - t0 > DEADLINE - 15:
                            break
                        
                        adj_small = _construct_random(smaller_n, density, seed)
                        adj_small, score_small = _local_search(adj_small, smaller_n, time_budget * 0.5, seed)
                        
                        if score_small == 0:
                            if smaller_n > best_n:
                                best_n = smaller_n
                                best_score = score_small
                                edge_set = set()
                                for i in range(smaller_n):
                                    for j in range(i + 1, smaller_n):
                                        if (adj_small[i] >> j) & 1:
                                            edge_set.add((i, j))
                                best_graph = {'n': smaller_n, 'edges': sorted(edge_set)}
    
    # Fallback: try known small valid graphs
    if best_graph is None:
        # 5-vertex cycle
        return {'n': 5, 'edges': [[0,1],[1,2],[2,3],[3,4],[4,0]]}
    
    return best_graph
