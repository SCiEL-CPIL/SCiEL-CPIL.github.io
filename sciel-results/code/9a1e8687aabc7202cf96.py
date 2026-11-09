import time
import random
import sys

DEADLINE = 330.0
SAFETY_MARGIN = 5.0

def _paley_graph(q):
    """Construct Paley graph on q vertices. Returns list of bitmasks."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return q, adj

def _double_cover_paley(q):
    """
    Construct a double cover of the Paley graph on q vertices.
    Vertices are (i, parity) for i in 0..q-1, parity in {0,1}.
    Vertex index = i * 2 + parity.
    Edge between (i, p) and (j, p') if:
      - i != j
      - (j - i) mod q is a quadratic residue mod q (Paley adjacency)
      - p XOR p' = 1 (parity must differ)
    """
    n = 2 * q
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * n
    
    for i in range(q):
        for j in range(i + 1, q):
            diff = (j - i) % q
            if diff in residues:
                v_i0 = i * 2
                v_i1 = i * 2 + 1
                v_j0 = j * 2
                v_j1 = j * 2 + 1
                
                # (i,0) connected to (j,1)
                adj[v_i0] |= (1 << v_j1)
                adj[v_j1] |= (1 << v_i0)
                
                # (i,1) connected to (j,0)
                adj[v_i1] |= (1 << v_j0)
                adj[v_j0] |= (1 << v_i1)
    
    return n, adj

def _find_violations(adj, n, limit=500):
    """Find up to 'limit' K5 or I5 violations. Returns list of sets of 5 vertices."""
    violations = []
    
    # Check K5
    for i in range(n):
        for j in range(i+1, n):
            if (adj[i] >> j) & 1:
                common = adj[i] & adj[j]
                mask = common
                while mask and len(violations) < limit:
                    lsb = mask & -mask
                    k = lsb.bit_length() - 1
                    mask ^= lsb
                    if k <= j: continue
                    C = adj[k] & common
                    if C & (C - 1):
                        mask_c = C
                        while mask_c and len(violations) < limit:
                            lsb_c = mask_c & -mask_c
                            u = lsb_c.bit_length() - 1
                            mask_c ^= lsb_c
                            if u <= k: continue
                            if (adj[u] & C):
                                candidates = adj[u] & C
                                if candidates:
                                     lsb_v = candidates & -candidates
                                     v = lsb_v.bit_length() - 1
                                     violations.append(frozenset([i, j, k, u, v]))
    
    # Check I5 (K5 in complement)
    comp_adj = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = full_mask ^ (1 << i) ^ adj[i]
        
    for i in range(n):
        for j in range(i+1, n):
            if (comp_adj[i] >> j) & 1:
                common = comp_adj[i] & comp_adj[j]
                mask = common
                while mask and len(violations) < limit:
                    lsb = mask & -mask
                    k = lsb.bit_length() - 1
                    mask ^= lsb
                    if k <= j: continue
                    C = comp_adj[k] & common
                    if C & (C - 1):
                        mask_c = C
                        while mask_c and len(violations) < limit:
                            lsb_c = mask_c & -mask_c
                            u = lsb_c.bit_length() - 1
                            mask_c ^= lsb_c
                            if u <= k: continue
                            if (comp_adj[u] & C):
                                candidates = comp_adj[u] & C
                                if candidates:
                                     lsb_v = candidates & -candidates
                                     v = lsb_v.bit_length() - 1
                                     violations.append(frozenset([i, j, k, u, v]))
    
    return violations

def _remove_vertices(adj, n, to_remove):
    """Remove a set of vertices from the graph."""
    if not to_remove:
        return n, adj
        
    to_remove_set = set(to_remove)
    new_n = n - len(to_remove_set)
    
    index_map = {}
    new_idx = 0
    for i in range(n):
        if i not in to_remove_set:
            index_map[i] = new_idx
            new_idx += 1
            
    new_adj = [0] * new_n
    for i in range(n):
        if i in to_remove_set:
            continue
        ni = index_map[i]
        mask = adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            mask ^= lsb
            if j in to_remove_set or i == j:
                continue
            nj = index_map[j]
            new_adj[ni] |= (1 << nj)
            
    return new_n, new_adj

def _is_55_free(adj, n):
    """Check if graph is (5,5)-free using bitsets."""
    if n < 5:
        return True
    
    # Check K5
    for i in range(n):
        for j in range(i+1, n):
            if (adj[i] >> j) & 1:
                common = adj[i] & adj[j]
                mask = common
                while mask:
                    lsb = mask & -mask
                    k = lsb.bit_length() - 1
                    mask ^= lsb
                    if k <= j: continue
                    C = adj[k] & common
                    if C & (C - 1):
                        mask_c = C
                        while mask_c:
                            lsb_c = mask_c & -mask_c
                            u = lsb_c.bit_length() - 1
                            mask_c ^= lsb_c
                            if (adj[u] & C):
                                return False
    
    # Check I5
    comp_adj = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = full_mask ^ (1 << i) ^ adj[i]
        
    for i in range(n):
        for j in range(i+1, n):
            if (comp_adj[i] >> j) & 1:
                common = comp_adj[i] & comp_adj[j]
                mask = common
                while mask:
                    lsb = mask & -mask
                    k = lsb.bit_length() - 1
                    mask ^= lsb
                    if k <= j: continue
                    C = comp_adj[k] & common
                    if C & (C - 1):
                        mask_c = C
                        while mask_c:
                            lsb_c = mask_c & -mask_c
                            u = lsb_c.bit_length() - 1
                            mask_c ^= lsb_c
                            if (comp_adj[u] & C):
                                return False
    return True

def _to_edges(adj, n):
    """Convert bitset adjacency to list of (i, j) edges."""
    edges = []
    for i in range(n):
        mask = adj[i] & ~((1 << (i+1)) - 1)
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            edges.append((i, j))
            mask ^= lsb
    return edges

def _greedy_prune_double_cover(adj, n, time_start, seed=0):
    """
    Prune the double cover by finding violations and removing a hitting set of vertices.
    Uses randomized greedy selection to escape local minima.
    """
    deadline = time_start + (DEADLINE - SAFETY_MARGIN)
    current_adj = adj[:]
    current_n = n
    random.seed(seed)
    
    while time.time() < deadline:
        if _is_55_free(current_adj, current_n):
            break
            
        violations = _find_violations(current_adj, current_n, limit=200)
        if not violations:
            break
            
        freq = {}
        for v_set in violations:
            for v in v_set:
                freq[v] = freq.get(v, 0) + 1
                
        if not freq:
             break
             
        # Score: prefer vertices with high frequency in violations.
        # Add some randomness to the selection.
        # Sort by frequency descending.
        candidates = sorted(freq.keys(), key=lambda v: -freq[v])
        
        # Pick from the top 10% or top 5 candidates with probability
        # to allow exploration.
        top_k = max(1, len(candidates) // 10)
        top_candidates = candidates[:top_k]
        
        # Weighted random choice from top candidates
        weights = [freq[v] for v in top_candidates]
        total_weight = sum(weights)
        r = random.random() * total_weight
        cumulative = 0
        v_to_remove = top_candidates[0]
        for v, w in zip(top_candidates, weights):
            cumulative += w
            if r <= cumulative:
                v_to_remove = v
                break
        
        new_n, new_adj = _remove_vertices(current_adj, current_n, [v_to_remove])
        current_n = new_n
        current_adj = new_adj
        
        if current_n < 5:
            break
            
    return current_n, current_adj

def construct():
    """Construct a (5,5)-free graph. Returns (n, edges)."""
    time_start = time.time()
    
    # Strategy 1: Start with Paley graph on 37 vertices
    q, adj_paley = _paley_graph(37)
    best_n = q
    best_adj = adj_paley[:]
    
    # Verify seed
    if not _is_55_free(adj_paley, q):
        raise ValueError("Seed graph is not (5,5)-free")
        
    # Strategy 2: Try double cover approach with multiple randomized restarts
    n_dc, adj_dc = _double_cover_paley(37)
    
    # We have ~300s. 
    # Each restart might take a few seconds if it prunes a lot.
    # Let's do as many restarts as possible.
    
    for seed in range(1000):
        if time.time() - time_start > DEADLINE - SAFETY_MARGIN:
            break
            
        n_pruned, adj_pruned = _greedy_prune_double_cover(adj_dc, n_dc, time_start, seed=seed)
        
        if n_pruned > best_n:
            if _is_55_free(adj_pruned, n_pruned):
                best_n = n_pruned
                best_adj = adj_pruned[:]
                print(f"Restart {seed}: Improved to {best_n} vertices", file=sys.stderr)
                
        # If we hit 42 (known best), we can stop trying to beat it? 
        # No, we want to maximize. 43, 44, 45, 46 are possible.
        
    edges = _to_edges(best_adj, best_n)
    return best_n, edges

if __name__ == "__main__":
    n, edges = construct()
    print(f"Constructed graph with {n} vertices and {len(edges)} edges")
