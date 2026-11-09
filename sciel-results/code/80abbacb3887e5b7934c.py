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

def _double_cover_paley(q, time_start=None):
    """
    Construct a double cover of the Paley graph on q vertices.
    Vertices are (i, parity) for i in 0..q-1, parity in {0,1}.
    Vertex index = i * 2 + parity.
    Edge between (i, p) and (j, p') if:
      - i != j
      - (j - i) mod q is a quadratic residue mod q (Paley adjacency)
      - p XOR p' = 1 (parity must differ)
    This creates a bipartite-like structure that may help avoid K5 and I5.
    """
    n = 2 * q
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * n
    
    # For each pair i < j, check if (j-i) mod q is a quadratic residue.
    # If so, connect (i,0)-(j,1) and (i,1)-(j,0).
    # This handles both directions since we check all i<j pairs and the
    # residue check (j-i)%q covers the cyclic structure.
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

def _find_violations(adj, n, limit=200):
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

def _greedy_prune_double_cover(adj, n, time_start):
    """
    Prune the double cover by finding violations and removing a hitting set of vertices.
    """
    deadline = time_start + (DEADLINE - SAFETY_MARGIN)
    current_adj = adj[:]
    current_n = n
    
    while time.time() < deadline:
        if _is_55_free(current_adj, current_n):
            break
            
        violations = _find_violations(current_adj, current_n, limit=100)
        if not violations:
            break
            
        freq = {}
        for v_set in violations:
            for v in v_set:
                freq[v] = freq.get(v, 0) + 1
                
        if not freq:
             break
             
        # Score: prefer vertices with high frequency in violations,
        # but also consider degree (remove high-degree vertices first to break more cliques)
        def sort_key(v):
            deg = bin(current_adj[v]).count('1')
            return (-freq[v], deg)
            
        candidates = sorted(freq.keys(), key=sort_key)
        
        # Remove the top candidate
        v_to_remove = candidates[0]
        
        new_n, new_adj = _remove_vertices(current_adj, current_n, [v_to_remove])
        current_n = new_n
        current_adj = new_adj
        
        if current_n < 5:
            break
            
    return current_n, current_adj

def _has_k4_in_mask(mask, adj, n):
    """Check if the subgraph induced by vertices in 'mask' contains a K4."""
    # Extract vertices in mask
    verts = []
    m = mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        verts.append(v)
        m ^= lsb
    
    if len(verts) < 4:
        return False
        
    # Check for K4 among these vertices
    # For efficiency, only check if the induced subgraph is dense enough?
    # Simple O(k^4) check where k is popcount(mask)
    # If k is large, this is slow. But in the search, k might be around n/2.
    # For n=37, k~18, 18^4 = 100k ops. This is fast enough for a heuristic step.
    
    for i in range(len(verts)):
        vi = verts[i]
        for j in range(i+1, len(verts)):
            vj = verts[j]
            if not (adj[vi] & (1 << vj)):
                continue
            for k in range(j+1, len(verts)):
                vk = verts[k]
                if not (adj[vi] & (1 << vk)) or not (adj[vj] & (1 << vk)):
                    continue
                for l in range(k+1, len(verts)):
                    vl = verts[l]
                    if (adj[vi] & (1 << vl)) and (adj[vj] & (1 << vl)) and \
                       (adj[vk] & (1 << vl)):
                        return True
    return False

def _has_i4_in_mask(mask, adj, n):
    """Check if the subgraph induced by vertices in 'mask' contains an independent set of size 4 (I4)."""
    # This is equivalent to checking if the complement subgraph has a K4.
    # Construct complement adjacency for the vertices in mask?
    # Or just check directly: 4 vertices with no edges between them.
    
    verts = []
    m = mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        verts.append(v)
        m ^= lsb
        
    if len(verts) < 4:
        return False
        
    for i in range(len(verts)):
        vi = verts[i]
        for j in range(i+1, len(verts)):
            vj = verts[j]
            if (adj[vi] & (1 << vj)):
                continue
            for k in range(j+1, len(verts)):
                vk = verts[k]
                if (adj[vi] & (1 << vk)) or (adj[vj] & (1 << vk)):
                    continue
                for l in range(k+1, len(verts)):
                    vl = verts[l]
                    if (adj[vi] & (1 << vl)) or (adj[vj] & (1 << vl)) or (adj[vk] & (1 << vl)):
                        continue
                    return True
    return False

def _count_violations(S_mask, adj, n):
    """Count K4s in S and I4s in complement of S for a smoother objective."""
    count = 0
    full_mask = (1 << n) - 1
    
    # Count K4s in S
    verts_in_S = []
    m = S_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        verts_in_S.append(v)
        m ^= lsb
    
    if len(verts_in_S) >= 4:
        for i in range(len(verts_in_S)):
            vi = verts_in_S[i]
            for j in range(i+1, len(verts_in_S)):
                vj = verts_in_S[j]
                if not (adj[vi] & (1 << vj)):
                    continue
                for k in range(j+1, len(verts_in_S)):
                    vk = verts_in_S[k]
                    if not (adj[vi] & (1 << vk)) or not (adj[vj] & (1 << vk)):
                        continue
                    for l in range(k+1, len(verts_in_S)):
                        vl = verts_in_S[l]
                        if (adj[vi] & (1 << vl)) and (adj[vj] & (1 << vl)) and \
                           (adj[vk] & (1 << vl)):
                            count += 1
                            if count > 100:
                                return count
    
    # Count I4s in complement of S
    comp_mask = full_mask ^ S_mask
    verts_in_comp = []
    m = comp_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        verts_in_comp.append(v)
        m ^= lsb
    
    if len(verts_in_comp) >= 4:
        for i in range(len(verts_in_comp)):
            vi = verts_in_comp[i]
            for j in range(i+1, len(verts_in_comp)):
                vj = verts_in_comp[j]
                if (adj[vi] & (1 << vj)):
                    continue
                for k in range(j+1, len(verts_in_comp)):
                    vk = verts_in_comp[k]
                    if (adj[vi] & (1 << vk)) or (adj[vj] & (1 << vk)):
                        continue
                    for l in range(k+1, len(verts_in_comp)):
                        vl = verts_in_comp[l]
                        if (adj[vi] & (1 << vl)) or (adj[vj] & (1 << vl)) or \
                           (adj[vk] & (1 << vl)):
                            count += 1
                            if count > 100:
                                return count
    
    return count

def _try_add_vertex(adj, n, time_start):
    """Try to find a valid neighborhood S for a new vertex."""
    full_mask = (1 << n) - 1
    
    # Heuristic initial S: 
    # Maybe start with a random subset of size n/2?
    # Or start with a small set and grow?
    # Let's try a few random initializations.
    
    best_S = None
    
    for init_attempt in range(30):
        if time.time() - time_start > DEADLINE - SAFETY_MARGIN:
            return None
            
        # Random initialization
        # Bias towards balanced sets?
        S_mask = 0
        for i in range(n):
            if random.random() < 0.5:
                S_mask |= (1 << i)
                
        # Local search / Simulated Annealing
        T = 1.0
        current_violations = _count_violations(S_mask, adj, n)
        
        if current_violations == 0:
            return S_mask
            
        for step in range(1500):
            if time.time() - time_start > DEADLINE - SAFETY_MARGIN:
                return None
                
            # Pick a random vertex to flip
            v = random.randint(0, n - 1)
            bit = 1 << v
            new_S_mask = S_mask ^ bit
            
            new_violations = _count_violations(new_S_mask, adj, n)
            
            # Accept if better, or with prob exp(-delta/T)
            delta = new_violations - current_violations
            if delta < 0 or (delta == 0 and random.random() < 0.3) or \
               (delta > 0 and random.random() < pow(2.71828, -delta / T)):
                S_mask = new_S_mask
                current_violations = new_violations
                
            if current_violations == 0:
                return S_mask
                
            T *= 0.998 # Cool down more slowly
    
    return None

def _is_55_free(adj, n):
    """Check if graph is (5,5)-free using bitsets. Optimized for larger n."""
    if n < 5:
        return True
    
    # Check K5
    for i in range(n):
        for j in range(i+1, n):
            if (adj[i] >> j) & 1:
                common = adj[i] & adj[j]
                # We need a K3 in common
                mask = common
                while mask:
                    lsb = mask & -mask
                    k = lsb.bit_length() - 1
                    mask ^= lsb
                    if k <= j: continue # Ensure i < j < k
                    C = adj[k] & common
                    if C & (C - 1):
                        mask_c = C
                        while mask_c:
                            lsb_c = mask_c & -mask_c
                            u = lsb_c.bit_length() - 1
                            mask_c ^= lsb_c
                            if (adj[u] & C):
                                return False # Found K5
    
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
                                return False # Found I5
    return True

def _to_edges(adj, n):
    """Convert bitset adjacency to list of (i, j) edges."""
    edges = []
    for i in range(n):
        # Get bits set in adj[i] greater than i
        mask = adj[i] & ~((1 << (i+1)) - 1)
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            edges.append((i, j))
            mask ^= lsb
    return edges

def _try_prune_double_cover(adj, n, time_start):
    """
    Try to prune the double cover to get a (5,5)-free graph with as many vertices as possible.
    Uses greedy hitting set on violations.
    """
    return _greedy_prune_double_cover(adj, n, time_start)

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
        
    random.seed(42)
    
    # Strategy 2: Try double cover approach
    # The double cover has 74 vertices. Prune to (5,5)-free.
    n_dc, adj_dc = _double_cover_paley(37, time_start)
    
    if n_dc > best_n:
        # Try to prune the double cover
        n_pruned, adj_pruned = _try_prune_double_cover(adj_dc, n_dc, time_start)
        
        if n_pruned > best_n and _is_55_free(adj_pruned, n_pruned):
            best_n = n_pruned
            best_adj = adj_pruned[:]
            print(f"Double cover pruned to {best_n} vertices", file=sys.stderr)
    
    # Strategy 3: Try to extend the best graph found so far
    current_adj = best_adj[:]
    current_n = best_n
    
    stuck_count = 0
    
    while time.time() - time_start < DEADLINE - SAFETY_MARGIN:
        # Try to add a vertex
        S_mask = _try_add_vertex(current_adj, current_n, time_start)
        
        if S_mask is not None:
            # Add new vertex
            new_n = current_n + 1
            new_adj = current_adj[:] + [0]
            
            for i in range(current_n):
                if (S_mask >> i) & 1:
                    new_adj[i] |= (1 << new_n)
                    new_adj[new_n] |= (1 << i)
            
            current_adj = new_adj
            current_n = new_n
            stuck_count = 0
            
            if current_n > best_n:
                best_n = current_n
                best_adj = current_adj[:]
                print(f"Extended to {current_n} vertices", file=sys.stderr)
        else:
            # Stuck. Try random edge flips to unblock
            improved = False
            for _ in range(80):
                if time.time() - time_start > DEADLINE - SAFETY_MARGIN:
                    break
                    
                i = random.randint(0, current_n - 1)
                j = random.randint(0, current_n - 1)
                if i == j: continue
                
                bit_i = 1 << j
                bit_j = 1 << i
                current_adj[i] ^= bit_i
                current_adj[j] ^= bit_j
                
                if _is_55_free(current_adj, current_n):
                    improved = True
                    break
                else:
                    current_adj[i] ^= bit_i
                    current_adj[j] ^= bit_j
            
            if not improved:
                stuck_count += 1
                if stuck_count > 3:
                    break
            else:
                stuck_count = 0
                
    edges = _to_edges(best_adj, best_n)
    return best_n, edges

if __name__ == "__main__":
    n, edges = construct()
    print(f"Constructed graph with {n} vertices and {len(edges)} edges")
