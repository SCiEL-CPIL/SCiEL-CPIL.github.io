import time
import numpy as np
import itertools
import random

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
    deadline = start_time + 280  # Safety margin for 330s limit
    
    # Fallback: Paley graph on 37 vertices
    n_seed, edges_seed = paley_graph(37)
    best_n = n_seed
    best_edges = edges_seed
    
    def check_graph_5_5_free(n, adj):
        """
        Check if graph with adjacency matrix adj is (5,5)-free.
        Returns True if no K5 and no independent 5-set.
        """
        # Pre-compute all 5-sets
        # For n <= 42, number of 5-sets is manageable (e.g., C(42,5) ~ 850k)
        # But we can optimize by checking cliques and independent sets separately or together.
        
        # Vectorized check for all 5-sets
        # Generate all combinations of 5 indices
        # This might be slow in Python loops, so we use a trick or accept the cost if n is small.
        # For n=42, C(42,5) = 850668.
        # Checking each involves 10 pair lookups.
        # 8.5 million lookups. This is fast in NumPy if we can vectorize.
        
        # Let's try to vectorize the check for a single graph.
        # We can iterate over all 5-sets using itertools, but that's slow in Python.
        # Alternative: Use the fact that if a K5 exists, it's a subgraph.
        # Actually, for n=42, a simple loop over combinations might take ~1-2 seconds in Python.
        # We need to be faster.
        
        # Let's use a NumPy-based approach for the count.
        # But generating all 5-sets as a matrix is memory intensive and slow to generate.
        
        # Instead, let's use a recursive or backtracking check? No, that's for finding one.
        # We need to check if ANY exists.
        
        # Let's stick to the pre-computed 5-set structure from the current program, 
        # but only build it once per n.
        
        pass # Placeholder, implemented in helper below

    def build_5set_structure(n):
        """Pre-compute all 5-sets and their edge pairs for fast violation counting."""
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
        return int(np.sum(k5) + np.sum(ind))

    def search_circulant(n, time_limit):
        """
        Search for a valid circulant graph on n vertices.
        """
        if time.time() > time_limit:
            return None
        
        rows, cols = build_5set_structure(n)
        
        # Number of distinct offsets for circulant graph
        # Offsets are 1 to n//2
        # For even n, n//2 is a self-inverse offset (edge i <-> i+n/2)
        # For odd n, offsets are 1 to (n-1)//2, and n-i is the inverse
        m = n // 2
        # The set of possible offsets to consider for symmetry
        # If we pick offset d, we implicitly pick n-d.
        # So we just pick a subset of {1, ..., m}.
        
        # Random search over subsets of {1, ..., m}
        # We want density ~ 0.5, so expected number of edges ~ n/2.
        # Each offset d (d < n/2) contributes 2 edges per vertex (d and n-d).
        # Offset n/2 (if even) contributes 1 edge per vertex.
        # Total edges = n * (k * 2 + (1 if even and n/2 in S else 0)) / 2
        # We want this to be ~ n * (n-1) / 4 ? No, density 0.5 means ~ n(n-1)/4 edges.
        # Average degree ~ n/2.
        
        # Let's just randomize the inclusion of each offset with probability p.
        # To get degree ~ n/2, we need roughly n/4 offsets.
        
        best_viol = float('inf')
        best_S = None
        best_adj = None
        
        trials = 0
        while time.time() < time_limit:
            trials += 1
            # Random subset of offsets
            # We can use a random mask
            mask = np.random.rand(m) < 0.5
            # Adjust probability to target density 0.5
            # Expected degree = sum(mask[i] * 2 for i in 1..m-1) + (mask[m-1] if even)
            # Let's just use p=0.5 for now
            
            # Build adjacency matrix
            adj = np.zeros((n, n), dtype=bool)
            for d in range(1, m + 1):
                if mask[d-1]:
                    # Add edges for offset d
                    # i connected to (i+d)%n
                    # This is a circulant shift
                    # adj[i, (i+d)%n] = True
                    # We can do this with slicing or np.roll
                    # But we need to fill the matrix.
                    pass
            
            # Efficient way to build circulant adj:
            # Create a vector for the first row
            row0 = np.zeros(n, dtype=bool)
            for d in range(1, m + 1):
                if mask[d-1]:
                    row0[d] = True
                    row0[n-d] = True
            # For even n, if d == n/2, row0[n/2] is set once.
            # The loop above sets row0[d] and row0[n-d]. If d == n/2, it sets row0[n/2] twice (same value).
            
            # Construct full matrix by rolling row0
            # adj[i, j] = row0[(j-i)%n]
            # This is a Toeplitz matrix.
            # We can use np.roll or just index.
            # For speed, we can pre-compute the indices?
            # Or just use the fact that it's circulant.
            
            # Let's build it explicitly for speed in the loop, as n is small.
            adj = np.zeros((n, n), dtype=bool)
            for i in range(n):
                adj[i] = np.roll(row0, -i) # Wait, roll shifts.
                # If row0[j] is True, then edge (0, j) exists.
                # Edge (i, j) exists if (j-i)%n is in S.
                # So adj[i, j] = row0[(j-i)%n].
                # np.roll(row0, -i) gives row0 shifted left by i.
                # Element k in rolled array is row0[(k+i)%n].
                # We want adj[i, k] = row0[(k-i)%n].
                # So we need to shift right by i?
                # Let's just use a simple loop for n<=42, it's fast enough.
                for j in range(n):
                    adj[i, j] = row0[(j - i) % n]
            
            viol = count_violations(adj, rows, cols)
            
            if viol == 0:
                # Found a valid graph
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i, j]:
                            edges.append((i, j))
                return n, edges
            
            if viol < best_viol:
                best_viol = viol
                best_adj = adj.copy()
                best_S = mask.copy()

        # If no valid graph found, return None (or best invalid for seeding?)
        # For now, return None to trigger general search
        return None

    def search_general(n, time_limit, seed_adj=None):
        """
        Simulated Annealing search for general graph.
        """
        if time.time() > time_limit:
            return None, float('inf')
            
        rows, cols = build_5set_structure(n)
        
        # Initialize adjacency
        if seed_adj is not None:
            adj = seed_adj.copy()
        elif n >= 37:
            adj = np.zeros((n, n), dtype=bool)
            _, seed_edges = paley_graph(37)
            for u, v in seed_edges:
                adj[u, v] = True
                adj[v, u] = True
            # Randomize new vertices
            for i in range(37, n):
                for j in range(i):
                    adj[i, j] = np.random.rand() < 0.5
                    adj[j, i] = adj[i, j]
        else:
            adj = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i+1, n):
                    adj[i, j] = np.random.rand() < 0.5
                    adj[j, i] = adj[i, j]
        
        # Pre-compute edge to 5-sets mapping for incremental updates
        # This is expensive to build, but n is small.
        # We can reuse the logic from the current program.
        
        # For simplicity and speed of implementation, we will use the full count_violations
        # but optimize the SA loop.
        # Actually, the incremental update is crucial for speed.
        
        # Let's build the edge_to_5sets map
        combos = list(itertools.combinations(range(n), 5))
        edge_to_5sets = {}
        for k, comb in enumerate(combos):
            for u, v in itertools.combinations(comb, 2):
                key = (min(u,v), max(u,v))
                if key not in edge_to_5sets:
                    edge_to_5sets[key] = []
                edge_to_5sets[key].append(k)
        
        # Current state of each 5-set: count of edges
        edge_counts = np.zeros(len(combos), dtype=np.int32)
        for k, comb in enumerate(combos):
            for u, v in itertools.combinations(comb, 2):
                if adj[u, v]:
                    edge_counts[k] += 1
        
        current_viol = int(np.sum((edge_counts == 10) | (edge_counts == 0)))
        best_viol = current_viol
        best_adj = adj.copy()
        
        if best_viol == 0:
            return best_adj, 0
            
        T = 1.0
        min_T = 0.01
        cooling = 0.999
        stagnation = 0
        max_stagnation = 5000
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        while time.time() < time_limit:
            idx = np.random.randint(0, len(all_edges))
            u, v = all_edges[idx]
            
            old_val = adj[u, v]
            adj[u, v] = not old_val
            adj[v, u] = adj[u, v]
            
            affected = edge_to_5sets.get((u, v), [])
            if old_val:
                for k in affected:
                    edge_counts[k] -= 1
            else:
                for k in affected:
                    edge_counts[k] += 1
            
            new_viol = int(np.sum((edge_counts == 10) | (edge_counts == 0)))
            delta = new_viol - current_viol
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_viol = new_viol
                if new_viol < best_viol:
                    best_viol = new_viol
                    best_adj = adj.copy()
                    stagnation = 0
                    if best_viol == 0:
                        return best_adj, 0
                else:
                    stagnation += 1
            else:
                adj[u, v] = old_val
                adj[v, u] = adj[u, v]
                for k in affected:
                    if old_val:
                        edge_counts[k] += 1
                    else:
                        edge_counts[k] -= 1
                stagnation += 1
                
            T = max(min_T, T * cooling)
            
            if stagnation > max_stagnation:
                # Restart with random perturbation
                # Keep the best_adj? Or randomize?
                # Let's randomize the non-seed part if n > 37
                if n >= 37:
                    for i in range(37, n):
                        for j in range(i):
                            adj[i, j] = np.random.rand() < 0.5
                            adj[j, i] = adj[i, j]
                else:
                    for i in range(n):
                        for j in range(i+1, n):
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

        return best_adj, best_viol

    # Try n from 42 down to 38
    # First, try circulant search (fast)
    for n in range(42, 37, -1):
        if time.time() > deadline - 20:
            break
        time_for_circ = min(5, deadline - time.time())
        result = search_circulant(n, time.time() + time_for_circ)
        if result:
            best_n, best_edges = result
            break # Found a valid graph for this n, since we go down, this is the best n?
            # Wait, we go 42, 41... If we find 42, we are done.
            # If we don't find 42, we try 41.
            # So if we find one, we should return it immediately?
            # Yes, because we are iterating downwards from max.
            # But maybe we should check if we can find a larger one? No, 42 is the max we try.
            # So if we find n=42, return.
            # If we find n=41, return? No, we should check if 42 was possible.
            # Since we already checked 42 and failed, 41 is the best.
            # So we can return immediately upon finding a valid graph for the current n.
            
    # If no circulant graph found, use general search
    # We should prioritize larger n.
    for n in range(42, 37, -1):
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        # Allocate time based on n
        # Give more time to larger n? Or equal time?
        # Let's give equal time slices.
        time_for_n = min(40, remaining)
        
        adj, viol = search_general(n, time.time() + time_for_n)
        
        if viol == 0:
            best_n = n
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j]:
                        edges.append((i, j))
            best_edges = edges
            # Since we are going down, if we find 42, we can stop.
            # If we find 41, we can stop? 
            # We want the MAXIMUM n.
            # So if we find 42, return.
            # If we find 41, we should check if 42 was possible? 
            # We already tried 42 in the loop above.
            # So if we found 41, it means 42 failed.
            # So we can return 41.
            break

    return best_n, best_edges
