import time
import random
import numpy as np

def construct():
    """
    Construct a K2,3-free graph on 20 vertices maximizing edges.
    Uses a local search strategy with edge addition and edge swapping.
    Maintains a common-neighbor count matrix for O(1) validity checks (with O(n) updates).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 20
    
    def is_k23_free(adj_mat, common_mat):
        """
        Verify that the graph is K2,3-free.
        A graph has K2,3 if there exist two vertices u, v with at least 3 common neighbors.
        So we just need to check that for all u != v, common_mat[u, v] <= 2.
        """
        # Exclude diagonal
        mask = ~np.eye(n, dtype=bool)
        if np.max(common_mat[mask]) > 2:
            return False
        return True

    def count_common_neighbors(adj_mat):
        """Compute the common neighbor matrix."""
        return (adj_mat @ adj_mat).astype(np.int8)

    def can_add_edge(u, v, adj_mat, common_mat):
        """Check if edge (u,v) can be added to current graph."""
        if adj_mat[u, v]:
            return False
        # Condition 1: common[u, v] <= 2
        if common_mat[u, v] > 2:
            return False
        
        # Condition 2: For all w in N(v) \ {u}, common[u, w] <= 1
        # Note: u is not in N(v) since adj[u,v]=0
        nv = np.nonzero(adj_mat[v])[0]
        if len(nv) > 0:
            if np.max(common_mat[u, nv]) > 1:
                return False
        
        # Condition 3: For all w in N(u) \ {v}, common[v, w] <= 1
        nu = np.nonzero(adj_mat[u])[0]
        if len(nu) > 0:
            if np.max(common_mat[v, nu]) > 1:
                return False
                
        return True

    def perform_add(u, v, adj_mat, common_mat):
        """Add edge (u,v) to adj and common."""
        # Get neighbors BEFORE modification
        nv = np.nonzero(adj_mat[v])[0]
        nu = np.nonzero(adj_mat[u])[0]
        
        adj_mat[u, v] = 1
        adj_mat[v, u] = 1
        
        # Update common
        # common[u, w] += 1 for w in nv
        # common[v, w] += 1 for w in nu
        
        # nv contains no u (since adj[u,v] was 0)
        # nu contains no v
        if len(nv) > 0:
            common_mat[u, nv] += 1
            common_mat[nv, u] += 1
        if len(nu) > 0:
            common_mat[v, nu] += 1
            common_mat[nu, v] += 1

    def perform_remove(u, v, adj_mat, common_mat):
        """Remove edge (u,v) from adj and common."""
        # Get neighbors BEFORE modification
        # Note: u is in N(v) and v is in N(u) currently
        nv = np.nonzero(adj_mat[v])[0]
        nu = np.nonzero(adj_mat[u])[0]
        
        adj_mat[u, v] = 0
        adj_mat[v, u] = 0
        
        # Update common
        # For w in nv: common[u, w] -= 1
        # For w in nu: common[v, w] -= 1
        
        if len(nv) > 0:
            common_mat[u, nv] -= 1
            common_mat[nv, u] -= 1
        if len(nu) > 0:
            common_mat[v, nu] -= 1
            common_mat[nu, v] -= 1

    def greedy_add(adj_mat, common_mat, deadline):
        """
        Greedily add edges until no more can be added.
        Returns the number of edges added.
        """
        added = 0
        while time.time() < deadline:
            added_any = False
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj_mat[u, v]]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if time.time() > deadline:
                    return added
                if can_add_edge(u, v, adj_mat, common_mat):
                    perform_add(u, v, adj_mat, common_mat)
                    added_any = True
                    added += 1
                    
            if not added_any:
                break
        return added

    def repair(adj_mat, common_mat, deadline):
        """
        Remove edges to make the graph K2,3-free if it's not.
        This is a fallback in case of bugs, but we aim to never need it.
        """
        if is_k23_free(adj_mat, common_mat):
            return adj_mat, common_mat
        
        # Find violating pairs
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if common_mat[u, v] > 2:
                        # Need to remove edges to reduce common neighbors
                        # Find common neighbors
                        common_neighbors = np.nonzero(adj_mat[u] & adj_mat[v])[0]
                        # Remove edges from u or v to some common neighbors
                        # To minimize edge loss, remove edges from the vertex with higher degree?
                        # Or just remove one edge to break the K2,3
                        # We need to reduce common[u, v] to 2.
                        # So we need to remove at least 1 edge from u or v to a common neighbor.
                        
                        # Let's remove edge (u, w) where w is a common neighbor.
                        # Which w? Maybe the one with lowest degree to minimize impact?
                        # Or random.
                        w = random.choice(common_neighbors.tolist())
                        
                        # Remove (u, w)
                        perform_remove(u, w, adj_mat, common_mat)
                        changed = True
                        break
                if changed:
                    break
        return adj_mat, common_mat

    best_count = -1
    best_adj = None
    
    # Try multiple restarts with different random seeds
    for seed in range(50):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with random sparse graph or empty
        # Let's start with 5 disjoint K4s as a strong base
        adj = np.zeros((n, n), dtype=np.int8)
        for k in range(5):
            for i in range(4):
                for j in range(i + 1, 4):
                    u = 4 * k + i
                    v = 4 * k + j
                    adj[u, v] = 1
                    adj[v, u] = 1
                    
        common = count_common_neighbors(adj)
        
        # Greedy addition
        greedy_add(adj, common, deadline)
        
        # Local search with swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try swaps
            # Pick a random edge to remove
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
            if not edges:
                break
            random.shuffle(edges)
            
            for rem_u, rem_v in edges:
                if time.time() > deadline:
                    break
                
                # Try removing (rem_u, rem_v) and adding some other edge
                perform_remove(rem_u, rem_v, adj, common)
                
                # Try to add a random non-edge
                non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
                random.shuffle(non_edges)
                
                added_swap = False
                for add_u, add_v in non_edges:
                    if time.time() > deadline:
                        break
                    if can_add_edge(add_u, add_v, adj, common):
                        perform_add(add_u, add_v, adj, common)
                        added_swap = True
                        improved = True
                        break # Only add one edge per swap attempt
                
                if not added_swap:
                    # Restore the removed edge
                    perform_add(rem_u, rem_v, adj, common)
        
        # Check if valid
        if not is_k23_free(adj, common):
            adj, common = repair(adj, common, deadline)
            if not is_k23_free(adj, common):
                continue # Skip if repair failed
        
        cnt = int(np.sum(adj) // 2)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj.copy()
            
    # If we didn't find any valid graph (shouldn't happen), return empty
    if best_adj is None:
        return []
        
    # Convert best_adj to list of edges
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                best_edges.append([u, v])
                
    return best_edges
