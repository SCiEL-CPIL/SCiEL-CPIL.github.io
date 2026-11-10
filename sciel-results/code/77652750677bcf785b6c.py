import random
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    num_edges = n * (n - 1) // 2  # 190 possible edges
    
    # Precompute edge index mapping
    edge_to_idx = {}
    edge_list = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_idx[(i, j)] = idx
            edge_list.append((i, j))
            idx += 1
    
    # Precompute for each triple (i,j,k), which edges are "common neighbor" edges
    # For a triple (i,j,k), an edge (v, w) creates a K3,3 if v,w are in {i,j,k} and the other endpoint is common neighbor
    # Actually: K3,3 means 3 vertices on one side, 3 on other, all 9 edges present
    # So for triple A={i,j,k} and triple B={p,q,r}, we need all 9 edges between A and B
    # The penalty should be for each such pair of triples where all 9 edges exist
    
    # For QUBO: x_e in {0,1} for each edge e
    # Objective: maximize sum(x_e) - P * sum over all K3,3 subgraphs of [product of 9 edge vars]
    # This is degree 9, too high for standard QUBO. Let's use a different formulation.
    
    # Alternative: For each triple of vertices (i,j,k), count common neighbors.
    # Common neighbors of {i,j,k} = vertices v such that (i,v), (j,v), (k,v) all exist
    # We want common_count(i,j,k) <= 2 for all triples
    # common_count(i,j,k) = sum_v x_{i,v} * x_{j,v} * x_{k,v} (where x_{i,v}=1 if edge exists)
    # Penalty: sum over all triples of max(0, common_count - 2)^2
    # This is a cubic polynomial in x variables.
    
    # For simulated annealing, we can compute the energy change when flipping an edge.
    # Let's precompute the structure needed for fast energy evaluation.
    
    # For each edge (u,v), precompute which triples are affected when we flip this edge.
    # Flipping edge (u,v) affects:
    # 1. Triples containing u and v: (u,v,w) for all w != u,v
    #    For triple (u,v,w): common neighbors include vertices z where (u,z),(v,z),(w,z) all exist
    #    When we add/remove (u,v), it doesn't directly change common neighbors of (u,v,w)
    #    Wait, I'm confusing myself. Let me reconsider.
    
    # The K3,3-free condition: no 3 vertices have 3 common neighbors.
    # Equivalently: for every triple (i,j,k), |N(i) ∩ N(j) ∩ N(k)| <= 2
    # where N(v) is the neighbor set of v.
    
    # When we flip edge (u,v):
    # - If we add (u,v): u becomes a neighbor of v and vice versa
    # - This affects common neighbor counts for triples that include u or v
    # - Specifically, for any triple (a,b,c) where at least one of a,b,c is u or v:
    #   The common neighbor count may change if u or v was/is a common neighbor
    
    # Let's think about it differently. For a fixed triple (i,j,k):
    # common_neighbors(i,j,k) = {v : v adjacent to i, j, and k}
    # When we flip edge (u,v):
    # - If u is in {i,j,k} and v is a common neighbor candidate: 
    #   - If we add (u,v) and v is adjacent to the other two in the triple, then v becomes a common neighbor
    #   - If we remove (u,v) and v was adjacent to the other two, v stops being a common neighbor
    # - Similarly for v in {i,j,k}
    
    # This is getting complex. Let's use a direct approach with NumPy for the SA.
    
    def common_count_from_mask(mask, i, j, k):
        return (mask[i] & mask[j] & mask[k]).bit_count()
    
    def count_violations_from_mask(mask):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                mij = mask[i] & mask[j]
                for k in range(j + 1, n):
                    if (mij & mask[k]).bit_count() >= 3:
                        count += 1
        return count
    
    def edge_count_from_mask(mask):
        return sum(mask[i].bit_count() for i in range(n)) // 2
    
    def mask_to_edges(mask):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    # Precompute all triples
    all_triples = list(combinations(range(n), 3))
    
    # For SA with NumPy, represent the graph as a 20x20 adjacency matrix
    # and use vectorized operations for energy computation
    
    def compute_energy_vec(adj):
        """Compute total violation count using vectorized operations.
        adj is a 20x20 numpy array with 0/1 entries.
        Returns the number of triples with >= 3 common neighbors.
        """
        # For each triple (i,j,k), common neighbors = row_i & row_j & row_k
        # We can compute this as: for each k, compute adj[:,k] & adj[:,i] & adj[:,j]
        # But that's O(n^3) which is fine for n=20
        
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                ij_common = adj[i] & adj[j]
                for k in range(j + 1, n):
                    if np.sum(ij_common & adj[k]) >= 3:
                        violations += 1
        return violations
    
    def compute_energy_vec_fast(adj):
        """Faster vectorized energy computation."""
        # adj is n x n boolean/integer array
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                ij_common = adj[i] & adj[j]
                # For all k > j, check if common neighbors >= 3
                for k in range(j + 1, n):
                    if np.count_nonzero(ij_common & adj[k]) >= 3:
                        violations += 1
        return violations
    
    # Simulated Annealing with NumPy
    def simulated_annealing(adj, time_budget=30.0):
        """Run SA on the adjacency matrix. Returns improved adj matrix."""
        sa_start = time.time()
        sa_deadline = sa_start + time_budget
        
        n_local = n
        current_adj = adj.copy()
        current_violations = compute_energy_vec_fast(current_adj)
        current_edges = int(np.sum(current_adj)) // 2
        current_energy = -current_edges + 100 * current_violations
        
        best_adj = current_adj.copy()
        best_energy = current_energy
        best_edges = current_edges
        best_violations = current_violations
        
        T = 50.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        rng = np.random.default_rng(42)
        
        while T > T_min and time.time() < sa_deadline:
            # Pick a random edge to flip
            i = rng.integers(0, n_local)
            j = rng.integers(0, n_local)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            # Compute delta energy
            old_val = current_adj[i, j]
            
            # Temporarily flip
            current_adj[i, j] = 1 - old_val
            current_adj[j, i] = 1 - old_val
            
            new_violations = compute_energy_vec_fast(current_adj)
            new_edges = int(np.sum(current_adj)) // 2
            new_energy = -new_edges + 100 * new_violations
            
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                # Accept
                current_energy = new_energy
                current_edges = new_edges
                current_violations = new_violations
                
                if new_violations == 0 and new_edges > best_edges:
                    best_adj = current_adj.copy()
                    best_energy = current_energy
                    best_edges = new_edges
                    best_violations = new_violations
            else:
                # Reject - flip back
                current_adj[i, j] = old_val
                current_adj[j, i] = old_val
            
            T *= cooling_rate
            
            # Periodic check for valid solutions
            if current_violations == 0 and current_edges > best_edges:
                best_adj = current_adj.copy()
                best_energy = current_energy
                best_edges = current_edges
                best_violations = current_violations
        
        return best_adj, best_edges, best_violations
    
    # Repair function using mask-based approach
    def repair_mask(mask):
        """Remove edges to eliminate all violations."""
        max_iter = 3000
        for _ in range(max_iter):
            if time.time() > deadline:
                return False
            
            # Find violations
            viols = []
            for i in range(n):
                mi = mask[i]
                for j in range(i + 1, n):
                    mij = mi & mask[j]
                    for k in range(j + 1, n):
                        if (mij & mask[k]).bit_count() >= 3:
                            viols.append((i, j, k))
            
            if not viols:
                return True
            
            # Find edges to remove
            edge_viol_count = {}
            for i, j, k in viols:
                for v in range(n):
                    if v not in (i, j, k):
                        if mask[i] & (1 << v) and mask[j] & (1 << v) and mask[k] & (1 << v):
                            for a, b in [(i, v), (j, v), (k, v)]:
                                key = (min(a, b), max(a, b))
                                edge_viol_count[key] = edge_viol_count.get(key, 0) + 1
            
            if edge_viol_count:
                worst_edge = max(edge_viol_count, key=edge_viol_count.get)
                u, v = worst_edge
                mask[u] &= ~(1 << v)
                mask[v] &= ~(1 << u)
            else:
                # Fallback
                i, j, k = viols[0]
                candidates = []
                for v in range(n):
                    if v != i and mask[i] & (1 << v):
                        candidates.append((i, v))
                    if v != j and mask[j] & (1 << v):
                        candidates.append((j, v))
                    if v != k and mask[k] & (1 << v):
                        candidates.append((k, v))
                if not candidates:
                    return False
                u, v = random.choice(candidates)
                mask[u] &= ~(1 << v)
                mask[v] &= ~(1 << u)
        
        # Final check
        for i in range(n):
            for j in range(i + 1, n):
                mij = mask[i] & mask[j]
                for k in range(j + 1, n):
                    if (mij & mask[k]).bit_count() >= 3:
                        return False
        return True
    
    def add_edges_mask(mask):
        """Try to add edges greedily."""
        improved = True
        while improved:
            if time.time() > deadline:
                return
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return
                    if not (mask[i] & (1 << j)):
                        # Check if adding (i,j) creates violation
                        mask[i] |= (1 << j)
                        mask[j] |= (1 << i)
                        valid = True
                        for w in range(n):
                            if w == i or w == j:
                                continue
                            if (mask[i] & mask[j] & mask[w]).bit_count() >= 3:
                                valid = False
                                break
                            for z in range(w + 1, n):
                                if z == i or z == j:
                                    continue
                                if (mask[i] & mask[w] & mask[z]).bit_count() >= 3:
                                    valid = False
                                    break
                                if (mask[j] & mask[w] & mask[z]).bit_count() >= 3:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            improved = True
                        else:
                            mask[i] &= ~(1 << j)
                            mask[j] &= ~(1 << i)
    
    def local_search_mask(mask):
        """Edge swap local search."""
        if time.time() > deadline:
            return False
        current_count = edge_count_from_mask(mask)
        
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i] & (1 << j):
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
        
        if not edges or not non_edges:
            return False
        
        random.shuffle(edges)
        random.shuffle(non_edges)
        
        for _ in range(50):
            if time.time() > deadline:
                return False
            a, b = random.choice(edges)
            c, d = random.choice(non_edges)
            
            mask[a] &= ~(1 << b)
            mask[b] &= ~(1 << a)
            
            # Check if we can add (c,d)
            mask[c] |= (1 << d)
            mask[d] |= (1 << c)
            
            valid = True
            for i in range(n):
                mi = mask[i]
                for j in range(i + 1, n):
                    mij = mi & mask[j]
                    for k in range(j + 1, n):
                        if (mij & mask[k]).bit_count() >= 3:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                mask[c] &= ~(1 << d)
                mask[d] &= ~(1 << c)
                mask[a] |= (1 << b)
                mask[b] |= (1 << a)
            else:
                return True
        
        return False
    
    # Convert between mask and numpy adj
    def mask_to_adj(mask):
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(n):
                if mask[i] & (1 << j):
                    adj[i, j] = 1
        return adj
    
    def adj_to_mask(adj):
        mask = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    mask[i] |= (1 << j)
        return mask
    
    # Main search
    random.seed(42)
    best_mask = None
    best_count = -1
    
    # Strategy 1: SA from random dense graph
    for trial in range(8):
        if time.time() > deadline - 20:
            break
        
        p = 0.6 + 0.05 * trial
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        # Repair first
        mask = adj_to_mask(adj)
        if repair_mask(mask):
            add_edges_mask(mask)
            swap_improved = True
            while swap_improved and time.time() < deadline - 15:
                swap_improved = local_search_mask(mask)
            
            ec = edge_count_from_mask(mask)
            if ec > best_count:
                best_count = ec
                best_mask = mask[:]
        
        # Now run SA on the repaired graph
        if time.time() < deadline - 10:
            adj = mask_to_adj(mask)
            sa_adj, sa_edges, sa_viols = simulated_annealing(adj, time_budget=min(20, deadline - time.time() - 5))
            
            if sa_viols == 0 and sa_edges > best_count:
                sa_mask = adj_to_mask(sa_adj)
                # Try to add more edges
                add_edges_mask(sa_mask)
                ec = edge_count_from_mask(sa_mask)
                if ec > best_count:
                    best_count = ec
                    best_mask = sa_mask[:]
    
    # Strategy 2: SA from complete graph
    if time.time() < deadline - 15:
        adj = np.ones((n, n), dtype=np.int8) - np.eye(n, dtype=np.int8)
        mask = adj_to_mask(adj)
        if repair_mask(mask):
            add_edges_mask(mask)
            ec = edge_count_from_mask(mask)
            if ec > best_count:
                best_count = ec
                best_mask = mask[:]
        
        if time.time() < deadline - 10:
            adj = mask_to_adj(mask)
            sa_adj, sa_edges, sa_viols = simulated_annealing(adj, time_budget=min(15, deadline - time.time() - 5))
            if sa_viols == 0 and sa_edges > best_count:
                sa_mask = adj_to_mask(sa_adj)
                add_edges_mask(sa_mask)
                ec = edge_count_from_mask(sa_mask)
                if ec > best_count:
                    best_count = ec
                    best_mask = sa_mask[:]
    
    # Strategy 3: More SA restarts with different parameters
    if best_mask is not None and time.time() < deadline - 10:
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            
            mask = best_mask[:]
            # Perturb
            for i in range(n):
                for j in range(i + 1, n):
                    if mask[i] & (1 << j) and random.random() < 0.15:
                        mask[i] &= ~(1 << j)
                        mask[j] &= ~(1 << i)
            
            if repair_mask(mask):
                add_edges_mask(mask)
                ec = edge_count_from_mask(mask)
                if ec > best_count:
                    best_count = ec
                    best_mask = mask[:]
                
                if time.time() < deadline - 5:
                    adj = mask_to_adj(mask)
                    sa_adj, sa_edges, sa_viols = simulated_annealing(adj, time_budget=min(10, deadline - time.time() - 3))
                    if sa_viols == 0 and sa_edges > best_count:
                        sa_mask = adj_to_mask(sa_adj)
                        add_edges_mask(sa_mask)
                        ec = edge_count_from_mask(sa_mask)
                        if ec > best_count:
                            best_count = ec
                            best_mask = sa_mask[:]
    
    if best_mask is None:
        # Fallback: empty graph
        return []
    
    edges = mask_to_edges(best_mask)
    return edges
