import itertools
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    def get_edges_from_adj(adj):
        """adj is n x n numpy bool array"""
        n = adj.shape[0]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return edges
    
    def has_regular_6(adj):
        """
        Check if there exists a 6-vertex subset inducing a regular graph.
        Returns (found, subset_indices) or (False, None).
        Uses NumPy vectorization for speed.
        """
        n = adj.shape[0]
        # Precompute all 6-subsets as index arrays
        # For n up to 22, C(22,6) = 74613 which is manageable
        subsets = np.array(list(itertools.combinations(range(n), 6)))
        if subsets.size == 0:
            return False, None
        
        # subsets shape: (num_subsets, 6)
        # For each subset, compute degrees of each vertex within the subset
        # adj_sub = adj[np.ix_(subsets, subsets)] -> (num_subsets, 6, 6)
        # This might be too large for n=22: 74613 * 6 * 6 = ~2.7M bools = 2.7MB, OK
        
        # Compute degree matrix: deg[i,j] = number of neighbors of subsets[i][j] within subset i
        # Use fancy indexing
        rows = subsets  # (S, 6)
        # For each subset s and position p, we want adj[subsets[s,p], subsets[s,:]]
        # This is a 3D lookup
        # adj[rows[:, None, :], rows] doesn't work directly
        # Let's do: for each subset index s, for each pos p, deg = sum(adj[rows[s,p], rows[s,:]])
        
        # Vectorized approach:
        # Create indices for the 3D array access
        # adj shape: (n, n)
        # We want result shape: (S, 6, 6) where result[s,p,q] = adj[rows[s,p], rows[s,q]]
        
        # Method: use np.take or advanced indexing
        # Let r = rows  # (S, 6)
        # We want adj[r[:, :, None], r[:, None, :]]
        # r[:, :, None] has shape (S, 6, 1), r[:, None, :] has shape (S, 1, 6)
        # Broadcasting gives (S, 6, 6)
        
        r = rows
        adj_sub = adj[r[:, :, None], r[:, None, :]]  # (S, 6, 6) bool array
        # deg[s, p] = sum of adj_sub[s, p, :]
        degs = adj_sub.sum(axis=2)  # (S, 6)
        
        # A subset is regular if all 6 degrees are equal
        # Check: degs[s, 0] == degs[s, 1] == ... == degs[s, 5]
        # Compute range of degrees in each subset
        min_deg = degs.min(axis=1)  # (S,)
        max_deg = degs.max(axis=1)  # (S,)
        
        regular_mask = (min_deg == max_deg)
        if regular_mask.any():
            idx = np.where(regular_mask)[0][0]
            return True, subsets[idx].tolist()
        
        return False, None
    
    def is_fully_valid(adj):
        """
        Full validation: check all subset sizes from 6 to n.
        Returns True if no regular induced subgraph of size >= 6 exists.
        """
        n = adj.shape[0]
        for k in range(6, n + 1):
            subsets = np.array(list(itertools.combinations(range(n), k)))
            if subsets.size == 0:
                continue
            r = subsets
            adj_sub = adj[r[:, :, None], r[:, None, :]]
            degs = adj_sub.sum(axis=2)
            min_deg = degs.min(axis=1)
            max_deg = degs.max(axis=1)
            if (min_deg == max_deg).any():
                return False
        return True
    
    def local_search(n, time_budget):
        """
        Local search: start from random graph, iteratively fix violations.
        Returns valid adj matrix or None.
        """
        t0 = time.time()
        max_steps = 50000
        
        for restart in range(100):
            if time.time() - t0 > time_budget:
                break
            
            # Random initial graph
            adj = (np.random.random((n, n)) < 0.5).astype(bool)
            np.fill_diagonal(adj, False)
            # Symmetrize
            adj = ((adj | adj.T) & ~np.eye(n, dtype=bool))
            
            # Local search
            for step in range(max_steps):
                if time.time() - t0 > time_budget:
                    break
                
                found, subset = has_regular_6(adj)
                if not found:
                    # Check full validity
                    if is_fully_valid(adj):
                        return adj
                    # If size-6 is clean but larger subsets are regular,
                    # we need to handle that. For now, try a few more steps.
                    # Actually, if no size-6 regular, larger ones can still be regular.
                    # Let's just try flipping random edges to break larger regularities.
                    # But this is rare. Let's do a few random flips and recheck.
                    for _ in range(20):
                        if time.time() - t0 > time_budget:
                            break
                        i, j = random.randint(0, n-1), random.randint(0, n-1)
                        if i != j:
                            adj[i, j] = not adj[i, j]
                            adj[j, i] = adj[i, j]
                        if is_fully_valid(adj):
                            return adj
                    continue
                
                # Found a regular 6-set. Flip a random edge within it.
                s = subset
                # Pick a random pair within the subset to flip
                i, j = random.sample(s, 2)
                adj[i, j] = not adj[i, j]
                adj[j, i] = adj[i, j]
        
        return None
    
    # Try n=22, 21, 20 with decreasing time budgets
    for n in [22, 21, 20]:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        budget = min(remaining * 0.4, 30)  # Spend up to 30s per n, or 40% of remaining
        if budget < 2:
            budget = max(1, remaining * 0.3)
        
        result = local_search(n, budget)
        if result is not None:
            return {'n': n, 'edges': get_edges_from_adj(result)}
    
    # Fallback: try n=19 with Paley graph
    p = 19
    adj = np.zeros((p, p), dtype=bool)
    qr = set()
    for i in range(1, p):
        qr.add((i * i) % p)
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in qr:
                adj[i, j] = True
                adj[j, i] = True
    
    if is_fully_valid(adj):
        return {'n': p, 'edges': get_edges_from_adj(adj)}
    
    # Fallback: small valid graph (n=5, no edges)
    return {'n': 5, 'edges': []}
