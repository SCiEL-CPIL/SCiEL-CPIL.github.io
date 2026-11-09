import time
import numpy as np
import itertools

def paley_graph(q):
    """Construct the Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Fallback: Paley graph on 37 vertices
    n_seed, edges_seed = paley_graph(37)
    best_n = n_seed
    best_edges = edges_seed
    
    # Double cover construction on Z_37 x {0,1}
    # Vertex (i, b) where i in [0, 36], b in {0, 1}
    # Total 74 vertices, but we'll try to prune to ~42
    # Adjacency rule: (i, b) ~ (j, c) if:
    #   1. (j - i) % 37 is a quadratic residue mod 37, AND
    #   2. b == c (same parity layer)
    # OR
    #   1. (j - i) % 37 is a quadratic non-residue mod 37, AND
    #   2. b != c (opposite parity layer)
    
    def double_cover_adjacency(n):
        """Build adjacency for double cover on n vertices (up to 74)."""
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        adj = np.zeros((n, n), dtype=bool)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Map to base and parity
                base_i = i % q
                parity_i = i // q
                base_j = j % q
                parity_j = j // q
                
                diff = (base_j - base_i) % q
                
                if diff in residues:
                    # Quadratic residue: connect if same parity
                    if parity_i == parity_j:
                        adj[i, j] = True
                        adj[j, i] = True
                else:
                    # Quadratic non-residue: connect if different parity
                    if parity_i != parity_j:
                        adj[i, j] = True
                        adj[j, i] = True
        
        return adj
    
    # Helper to get indices for all 5-sets
    def get_5set_indices(n):
        combos = list(itertools.combinations(range(n), 5))
        N = len(combos)
        if N == 0:
            return np.empty((0, 10), dtype=np.int32), np.empty((0, 10), dtype=np.int32)
        rows = np.zeros((N, 10), dtype=np.int32)
        cols = np.zeros((N, 10), dtype=np.int32)
        for k, comb in enumerate(combos):
            pairs = list(itertools.combinations(comb, 2))
            for j, (u, v) in enumerate(pairs):
                rows[k, j] = u
                cols[k, j] = v
        return rows, cols

    def count_violations(adj, rows, cols):
        if len(rows) == 0:
            return 0
        vals = adj[rows, cols]
        k5 = np.all(vals, axis=1)
        ind = np.all(~vals, axis=1)
        return int(np.sum(k5)) + int(np.sum(ind))
    
    def find_violating_5sets(adj, rows, cols):
        """Return list of 5-sets that form K5 or independent set."""
        if len(rows) == 0:
            return []
        vals = adj[rows, cols]
        k5_mask = np.all(vals, axis=1)
        ind_mask = np.all(~vals, axis=1)
        
        violating_indices = np.where(k5_mask | ind_mask)[0]
        
        # Reconstruct 5-sets from indices
        # This is slow for large N, but we only call when needed
        combos = list(itertools.combinations(range(len(rows)), 5))
        # Actually, we need the original vertex combinations
        # Let's rebuild from the rows/cols mapping
        # For now, just return count and let search handle it
        return len(violating_indices)
    
    def local_search_prune(adj, n, time_limit):
        """Prune edges that participate in K5 or independent 5-sets."""
        rows, cols = get_5set_indices(n)
        current_viol = count_violations(adj, rows, cols)
        
        best_adj = adj.copy()
        best_viol = current_viol
        
        if best_viol == 0:
            return best_adj, best_viol
        
        # Greedy pruning: remove edges that are in many violating 5-sets
        # First, identify which edges are in violating 5-sets
        edge_viol_count = np.zeros((n, n), dtype=np.int32)
        
        # For efficiency, sample a subset of 5-sets
        num_samples = min(10000, len(rows))
        sample_indices = np.random.choice(len(rows), size=num_samples, replace=False)
        
        for idx in sample_indices:
            vals = adj[rows[idx], cols[idx]]
            is_k5 = np.all(vals)
            is_ind = np.all(~vals)
            if is_k5 or is_ind:
                # Count this edge's participation
                for j in range(10):
                    u = rows[idx, j]
                    v = cols[idx, j]
                    edge_viol_count[u, v] += 1
                    edge_viol_count[v, u] += 1
        
        # Remove edges with highest violation count
        for _ in range(50):  # Max 50 pruning iterations
            if time.time() > time_limit:
                break
            
            # Find edge with max violation count
            max_count = np.max(edge_viol_count)
            if max_count == 0:
                break
            
            # Find all edges with this count
            mask = edge_viol_count == max_count
            # Pick one randomly
            candidates = np.argwhere(mask)
            if len(candidates) == 0:
                break
            u, v = candidates[np.random.randint(len(candidates))]
            
            if u == v:
                continue
            
            # Try removing this edge
            old_val = adj[u, v]
            if old_val:
                adj[u, v] = False
                adj[v, u] = False
                new_viol = count_violations(adj, rows, cols)
                
                if new_viol <= current_viol:
                    current_viol = new_viol
                    if new_viol < best_viol:
                        best_viol = new_viol
                        best_adj = adj.copy()
                    # Update edge violation counts
                    # Recompute for efficiency (approximate)
                    edge_viol_count = np.zeros((n, n), dtype=np.int32)
                    num_samples2 = min(5000, len(rows))
                    sample_indices2 = np.random.choice(len(rows), size=num_samples2, replace=False)
                    for idx in sample_indices2:
                        vals2 = adj[rows[idx], cols[idx]]
                        is_k5 = np.all(vals2)
                        is_ind = np.all(~vals2)
                        if is_k5 or is_ind:
                            for j in range(10):
                                uu = rows[idx, j]
                                vv = cols[idx, j]
                                edge_viol_count[uu, vv] += 1
                                edge_viol_count[vv, uu] += 1
                else:
                    # Revert
                    adj[u, v] = True
                    adj[v, u] = True
        
        # Now do simulated annealing on remaining violations
        T = 1.0
        min_T = 0.01
        cooling = 0.999
        stagnation = 0
        max_stagnation = 10000
        
        while time.time() < time_limit:
            if best_viol == 0:
                break
            
            # Pick random edge to flip
            u = np.random.randint(0, n)
            v = np.random.randint(0, n)
            if u == v:
                continue
            
            old_val = adj[u, v]
            adj[u, v] = not old_val
            adj[v, u] = adj[u, v]
            
            new_viol = count_violations(adj, rows, cols)
            delta = new_viol - current_viol
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_viol = new_viol
                if new_viol < best_viol:
                    best_viol = new_viol
                    best_adj = adj.copy()
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                adj[u, v] = old_val
                adj[v, u] = old_val
                stagnation += 1
            
            T = max(min_T, T * cooling)
            
            if stagnation > max_stagnation:
                # Restart: re-initialize with double cover
                adj = double_cover_adjacency(n)
                current_viol = count_violations(adj, rows, cols)
                T = 1.0
                stagnation = 0
        
        return best_adj, best_viol

    # Strategy: Start with double cover on 74 vertices, then prune to ~42
    # Try different target sizes
    
    # First, try to get a valid 42-vertex graph from the double cover
    for target_n in [42, 41, 40, 39, 38]:
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 20:
            break
        
        # Initialize with double cover, then take first target_n vertices
        # Actually, let's use all 74 vertices initially and prune
        n_full = 74
        adj_full = double_cover_adjacency(n_full)
        
        # Prune to target_n by removing vertices with highest degree first
        # This is a heuristic to get a smaller subgraph
        degrees = adj_full.sum(axis=1)
        
        # Sort vertices by degree, remove highest degree vertices first
        order = np.argsort(degrees)[::-1]  # Highest degree first
        
        # Keep target_n vertices with lowest degrees
        keep_set = set(order[n_full - target_n:].tolist())
        
        # Build subgraph
        adj_sub = np.zeros((target_n, target_n), dtype=bool)
        vertex_map = {}
        for new_idx, old_idx in enumerate(sorted(keep_set)):
            vertex_map[old_idx] = new_idx
        
        for old_i in sorted(keep_set):
            for old_j in sorted(keep_set):
                if old_i != old_j and adj_full[old_i, old_j]:
                    adj_sub[vertex_map[old_i], vertex_map[old_j]] = True
        
        # Local search on subgraph
        time_for_this = min(remaining * 0.8, 120)
        adj_result, viol = local_search_prune(adj_sub, target_n, time.time() + time_for_this)
        
        if viol == 0:
            best_n = target_n
            edges = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if adj_result[i, j]:
                        edges.append((i, j))
            best_edges = edges
            break  # Found valid graph, stop
    
    # If we didn't find anything better, try direct construction for n=38-42
    if best_n == 37:
        for n in range(38, 43):
            if time.time() > deadline:
                break
            
            remaining = deadline - time.time()
            if remaining < 15:
                break
            
            # Initialize with Paley(37) + random edges for new vertices
            adj = np.zeros((n, n), dtype=bool)
            _, seed_edges = paley_graph(37)
            for u, v in seed_edges:
                adj[u, v] = True
                adj[v, u] = True
            
            for i in range(37, n):
                for j in range(i):
                    adj[i, j] = np.random.rand() < 0.5
                    adj[j, i] = adj[i, j]
            
            time_for_n = min(remaining * 0.7, 60)
            adj_result, viol = local_search_prune(adj, n, time.time() + time_for_n)
            
            if viol == 0:
                best_n = n
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj_result[i, j]:
                            edges.append((i, j))
                best_edges = edges
    
    return best_n, best_edges
