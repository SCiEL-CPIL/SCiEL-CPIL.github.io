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
    deadline = start_time + 285  # Safety margin
    
    # Fallback: Paley graph on 37 vertices
    n_seed, edges_seed = paley_graph(37)
    best_n = n_seed
    best_edges = edges_seed
    
    def build_5set_structure(n):
        """Pre-compute all 5-sets and their edge pairs for fast violation counting."""
        combos = list(itertools.combinations(range(n), 5))
        N = len(combos)
        if N == 0:
            return np.empty((0, 10), dtype=np.int32), np.empty((0, 10), dtype=np.int32), np.empty((0,), dtype=np.int32)
        
        rows = np.zeros((N, 10), dtype=np.int32)
        cols = np.zeros((N, 10), dtype=np.int32)
        for k, comb in enumerate(combos):
            pairs = list(itertools.combinations(comb, 2))
            for j, (u, v) in enumerate(pairs):
                rows[k, j] = u
                cols[k, j] = v
        return rows, cols, combos

    def count_violations(adj, rows, cols):
        if len(rows) == 0:
            return 0
        vals = adj[rows, cols]
        k5 = np.all(vals, axis=1)
        ind = np.all(~vals, axis=1)
        return int(np.sum(k5) + np.sum(ind))

    def search(n, time_limit):
        # Initialize adjacency matrix
        adj = np.zeros((n, n), dtype=bool)
        
        # Seed with Paley(37) if n >= 37
        if n >= 37:
            _, seed_edges = paley_graph(37)
            for u, v in seed_edges:
                adj[u, v] = True
                adj[v, u] = True
            # Randomize connections for new vertices
            for i in range(37, n):
                for j in range(i):
                    adj[i, j] = np.random.rand() < 0.5
                    adj[j, i] = adj[i, j]
        else:
            for i in range(n):
                for j in range(i+1, n):
                    adj[i, j] = np.random.rand() < 0.5
                    adj[j, i] = adj[i, j]
        
        rows, cols, combos = build_5set_structure(n)
        
        current_viol = count_violations(adj, rows, cols)
        best_adj = adj.copy()
        best_viol = current_viol
        
        if best_viol == 0:
            return best_adj, best_viol
            
        T = 1.0
        min_T = 0.01
        cooling = 0.999  # Very slow cooling
        
        stagnation = 0
        max_stagnation = 10000
        
        # Pre-compute for each edge, which 5-sets contain it
        # This allows incremental violation updates
        edge_to_5sets = {}
        for k, comb in enumerate(combos):
            for u, v in itertools.combinations(comb, 2):
                key = (min(u,v), max(u,v))
                if key not in edge_to_5sets:
                    edge_to_5sets[key] = []
                edge_to_5sets[key].append(k)
        
        # Current state of each 5-set: how many edges are present
        # For a 5-set with vertices v0..v4, track count of present edges
        # If count == 10: K5 violation
        # If count == 0: independent set violation
        edge_counts = np.zeros(len(combos), dtype=np.int32)
        for k, comb in enumerate(combos):
            for u, v in itertools.combinations(comb, 2):
                if adj[u, v]:
                    edge_counts[k] += 1
        
        # Recompute violations from edge_counts
        current_viol = int(np.sum((edge_counts == 10) | (edge_counts == 0)))
        best_viol = current_viol
        best_adj = adj.copy()
        
        if best_viol == 0:
            return best_adj, best_viol
            
        # Build list of all edges for random selection
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        while time.time() < time_limit:
            # Pick random edge to flip
            idx = np.random.randint(0, len(all_edges))
            u, v = all_edges[idx]
            
            # Flip
            old_val = adj[u, v]
            adj[u, v] = not old_val
            adj[v, u] = adj[u, v]
            
            # Update edge_counts for all 5-sets containing this edge
            affected = edge_to_5sets.get((u, v), [])
            if old_val:
                # Edge removed: decrement counts
                for k in affected:
                    edge_counts[k] -= 1
            else:
                # Edge added: increment counts
                for k in affected:
                    edge_counts[k] += 1
            
            # Recompute violations efficiently
            new_viol = int(np.sum((edge_counts == 10) | (edge_counts == 0)))
            
            delta = new_viol - current_viol
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_viol = new_viol
                if new_viol < best_viol:
                    best_viol = new_viol
                    best_adj = adj.copy()
                    stagnation = 0
                    if best_viol == 0:
                        return best_adj, best_viol
                else:
                    stagnation += 1
            else:
                # Revert
                adj[u, v] = old_val
                adj[v, u] = adj[u, v]
                # Revert edge_counts
                for k in affected:
                    if old_val:
                        edge_counts[k] += 1
                    else:
                        edge_counts[k] -= 1
                stagnation += 1
                
            T = max(min_T, T * cooling)
            
            # Restart logic
            if stagnation > max_stagnation:
                if n >= 37:
                    for i in range(37, n):
                        for j in range(i):
                            adj[i, j] = np.random.rand() < 0.5
                            adj[j, i] = adj[i, j]
                    # Rebuild edge_counts
                    for k in range(len(combos)):
                        count = 0
                        for u, v in itertools.combinations(combos[k], 2):
                            if adj[u, v]:
                                count += 1
                        edge_counts[k] = count
                    current_viol = int(np.sum((edge_counts == 10) | (edge_counts == 0)))
                    T = 1.0
                    stagnation = 0
                else:
                    for i in range(n):
                        for j in range(i+1, n):
                            adj[i, j] = np.random.rand() < 0.5
                            adj[j, i] = adj[i, j]
                    for k in range(len(combos)):
                        count = 0
                        for u, v in itertools.combinations(combos[k], 2):
                            if adj[u, v]:
                                count += 1
                        edge_counts[k] = count
                    current_viol = int(np.sum((edge_counts == 10) | (edge_counts == 0)))
                    T = 1.0
                    stagnation = 0

        return best_adj, best_viol

    # Try to find a valid graph for n > 37
    for n in range(38, 43):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 10:
            break
            
        time_for_n = min(50, remaining)
        
        adj, viol = search(n, time.time() + time_for_n)
        
        if viol == 0:
            best_n = n
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j]:
                        edges.append((i, j))
            best_edges = edges

    return best_n, best_edges
