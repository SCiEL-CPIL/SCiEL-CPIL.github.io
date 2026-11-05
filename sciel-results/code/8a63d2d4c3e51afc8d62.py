import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    n = 31
    target_edges = 91
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_edges_set = set(all_edges)
    
    def count_c4(adj):
        """Count 4-cycles using adjacency matrix."""
        # A 4-cycle is a subgraph isomorphic to C4.
        # Number of C4s = (1/8) * sum_{i,j} (A^2_{ij} choose 2) for i != j? No.
        # Let M = A^2. M_{ij} is the number of common neighbors of i and j.
        # A 4-cycle is defined by 4 distinct vertices i, j, k, l such that i-j-k-l-i is a cycle.
        # This corresponds to pairs (i, k) and (j, l) being the diagonals? No.
        # Standard formula:
        # Count pairs of vertices (u, v) and their common neighbors.
        # For each pair (u, v), let c_{uv} be the number of common neighbors.
        # Each C4 is counted 4 times? No.
        # A C4 has 4 vertices. It has 2 pairs of opposite vertices.
        # Let the vertices be 1, 2, 3, 4 in order.
        # Pairs of opposite vertices: (1,3) and (2,4).
        # Common neighbors of 1 and 3 are {2, 4}. So c_{13} >= 2.
        # Common neighbors of 2 and 4 are {1, 3}. So c_{24} >= 2.
        # The number of C4s is (1/8) * sum_{u < v} c_{uv} (c_{uv} - 1)?
        # Let's verify.
        # If we sum c_{uv}(c_{uv}-1)/2 over all u < v, we count the number of ways to choose 2 common neighbors for each pair (u,v).
        # For a specific C4 (1,2,3,4), the pairs of opposite vertices are (1,3) and (2,4).
        # For pair (1,3), the common neighbors are 2 and 4. This contributes 1 to the sum for pair (1,3).
        # For pair (2,4), the common neighbors are 1 and 3. This contributes 1 to the sum for pair (2,4).
        # So each C4 contributes 2 to the sum over u < v of c_{uv}(c_{uv}-1)/2.
        # Wait, is that all?
        # What about the sides? (1,2) has common neighbors? In a simple C4, 1 and 2 are adjacent, so they don't have 2 common neighbors in the C4 itself, but they might in the graph.
        # The formula counts "diamonds" or "wedges" that close into C4s.
        # Specifically, sum_{u < v} C(c_{uv}, 2) counts the number of pairs of paths of length 2 between u and v.
        # Each such pair of paths forms a C4 if the internal vertices are distinct.
        # Since the paths are u-x-v and u-y-v with x != y, the vertices are u, v, x, y.
        # Are they distinct? u != v. x != y. x != u, x != v (simple path). y != u, y != v.
        # Could x == u? No, x is a neighbor of u and v. If x==u, then u~u which is false.
        # So u, v, x, y are distinct.
        # Thus each term C(c_{uv}, 2) counts the number of C4s where u and v are opposite vertices.
        # Each C4 has exactly 2 pairs of opposite vertices.
        # So the total number of C4s is (1/2) * sum_{u < v} C(c_{uv}, 2).
        
        # Using numpy:
        # A is n x n. A^2 is n x n.
        # c_{uv} is (A^2)_{uv}.
        # We need sum over u < v of C(A^2_{uv}, 2).
        # C(k, 2) = k(k-1)/2.
        
        A2 = adj @ adj
        # Upper triangle of A2
        iu = np.triu_indices(n, 1)
        vals = A2[iu]
        # Count only pairs with common neighbors >= 2
        mask = vals >= 2
        total = np.sum(vals[mask] * (vals[mask] - 1)) // 2
        return total // 2

    def delta_c4_swap(adj, u_rm, v_rm, u_add, v_add):
        """
        Compute change in C4 count when removing (u_rm, v_rm) and adding (u_add, v_add).
        """
        # Affected vertices in the "diagonal" sum are those pairs (x, y) where
        # c_{xy} changes.
        # c_{xy} = |N(x) ∩ N(y)|.
        # Changing edges (u_rm, v_rm) and (u_add, v_add) changes N(u_rm), N(v_rm), N(u_add), N(v_add).
        # So c_{xy} changes if x or y is in {u_rm, v_rm, u_add, v_add}.
        
        affected = set()
        for v in (u_rm, v_rm, u_add, v_add):
            for u in range(n):
                if u != v:
                    a, b = min(u, v), max(u, v)
                    affected.add((a, b))
        
        old_contrib = 0
        new_contrib = 0
        
        # We need to calculate the contribution of these pairs to the sum S = sum C(c_{xy}, 2)
        # Note: The final C4 count is S // 2.
        # So delta_C4 = (new_S - old_S) // 2.
        
        for (x, y) in affected:
            # Old common neighbors
            # We can use the current adj matrix
            # Common neighbors are z such that adj[x,z] == 1 and adj[y,z] == 1
            # This is (adj @ adj)[x, y]
            old_common = adj[x, y] # Wait, adj[x,y] is 0/1. We need A2[x,y].
            # Let's just compute intersection of rows.
            # Rows are bitsets? No, numpy array.
            # adj[x] is a vector.
            # common = np.sum(adj[x] & adj[y])
            
            # Optimization: Pre-compute A2?
            # But A2 changes.
            # Let's just compute the intersection for the affected pairs.
            # n=31, so adj[x] is small.
            
            old_common = np.sum(adj[x] & adj[y])
            if old_common >= 2:
                old_contrib += old_common * (old_common - 1) // 2
            
            # New common neighbors
            # We need to simulate the swap on the adjacency of x and y?
            # No, we need the new adjacency matrix to compute common neighbors.
            # But we only changed 4 entries in adj.
            # Let's create a temporary copy of the relevant parts?
            # Or just calculate the change in common neighbors directly.
            
            # New common neighbors of x and y:
            # z is a new common neighbor if:
            # 1. z was a common neighbor, and remains one.
            # 2. z was NOT a common neighbor, but becomes one due to the swap.
            # 3. z WAS a common neighbor, but ceases to be one due to the swap.
            
            # Let's just do the temporary modification on a copy of adj?
            # adj is 31x31. Copying is fast.
            new_adj = adj.copy()
            new_adj[u_rm, v_rm] = 0
            new_adj[v_rm, u_rm] = 0
            new_adj[u_add, v_add] = 1
            new_adj[v_add, u_add] = 1
            
            new_common = np.sum(new_adj[x] & new_adj[y])
            if new_common >= 2:
                new_contrib += new_common * (new_common - 1) // 2
        
        return (new_contrib - old_contrib) // 2

    def generate_initial_graph():
        """Greedy C4-free construction to get close to 91 edges."""
        adj = np.zeros((n, n), dtype=np.int8)
        edges = set()
        
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for e in shuffled_edges:
            u, v = e
            # Check if adding (u,v) creates a C4
            # This happens if u and v have >= 2 common neighbors
            common = np.sum(adj[u] & adj[v])
            if common < 2:
                adj[u, v] = 1
                adj[v, u] = 1
                edges.add(e)
            if len(edges) >= target_edges:
                break
        
        # If we didn't reach 91, add random edges (will create C4s)
        if len(edges) < target_edges:
            remaining = [e for e in all_edges if e not in edges]
            random.shuffle(remaining)
            for e in remaining:
                if len(edges) >= target_edges:
                    break
                u, v = e
                adj[u, v] = 1
                adj[v, u] = 1
                edges.add(e)
        
        return adj, edges

    def local_search(adj_init, edges_init, time_limit):
        """Simulated annealing with edge swaps."""
        adj = adj_init.copy()
        edges = set(edges_init)
        
        current_score = count_c4(adj)
        best_score = current_score
        best_adj = adj.copy()
        best_edges = set(edges)
        
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        while time.time() - start_time < time_limit and T > min_T:
            # Pick random edge to remove
            if not edges:
                break
            e_rm = random.choice(list(edges))
            u_rm, v_rm = e_rm
            
            # Pick random non-edge to add
            # Optimization: Pre-compute list of non-edges?
            # Or just pick random pair and check.
            while True:
                u_add = random.randint(0, n - 1)
                v_add = random.randint(0, n - 1)
                if u_add == v_add:
                    continue
                if u_add > v_add:
                    u_add, v_add = v_add, u_add
                e_add = (u_add, v_add)
                if e_add not in edges:
                    break
            
            # Compute delta
            delta = delta_c4_swap(adj, u_rm, v_rm, u_add, v_add)
            
            # Accept or reject
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                adj[u_rm, v_rm] = 0
                adj[v_rm, u_rm] = 0
                adj[u_add, v_add] = 1
                adj[v_add, u_add] = 1
                edges.discard(e_rm)
                edges.add(e_add)
                
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj.copy()
                    best_edges = set(edges)
            # Else: reject, do nothing
            
            T *= cooling
        
        return best_score, best_adj, best_edges

    best_overall_score = float('inf')
    best_overall_edges = None
    
    # Run multiple restarts
    while time.time() - start_time < deadline - 2:
        init_adj, init_edges = generate_initial_graph()
        score, adj, edges = local_search(init_adj, init_edges, deadline - 2)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_edges = edges
            if best_overall_score <= 2:
                break
    
    if best_overall_edges is None:
        init_adj, init_edges = generate_initial_graph()
        best_overall_edges = init_edges
    
    return [[u, v] for u, v in best_overall_edges]

