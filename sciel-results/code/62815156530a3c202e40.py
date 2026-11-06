import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def time_left():
        return deadline - time.time()

    def get_edges_from_adj(adj, n):
        edges = []
        for u in range(n):
            au = adj[u]
            # Iterate over set bits in adj[u] where v > u
            remaining = au & ~((1 << (u + 1)) - 1)
            while remaining:
                lsb = remaining & -remaining
                v = lsb.bit_length() - 1
                edges.append([u, v])
                remaining ^= lsb
        return edges

    def check_triangle_free(adj, n):
        """
        Returns True if graph is triangle-free.
        Uses bitset intersection.
        """
        for u in range(n):
            au = adj[u]
            # Check all pairs of neighbors
            # If u is connected to v and w, and v is connected to w, it's a triangle.
            # We can iterate v in N(u) and check if N(v) intersects N(u) \ {v}.
            # More efficiently: for each v in N(u), if (adj[u] & adj[v]) != 0, 
            # we need to ensure no common neighbor.
            # Actually, if adj[u] & adj[v] is non-zero, and v is in adj[u], 
            # then any w in (adj[u] & adj[v]) forms triangle u-v-w.
            # So we just check if (adj[u] & adj[v]) is non-zero for any v in adj[u].
            # Wait, if u-v is an edge, v is in adj[u].
            # If there is a w such that u-w and v-w are edges, then w is in adj[u] and adj[v].
            # So yes, if (adj[u] & adj[v]) != 0 for any v in adj[u], there is a triangle.
            # However, we must be careful not to count self-loops (none here) or double counting.
            # Actually, if u~v, then v is in adj[u].
            # If w is a common neighbor, w is in adj[u] and adj[v].
            # So (adj[u] & adj[v]) contains w.
            # So the condition is: for all v in adj[u], (adj[u] & adj[v]) == 0.
            
            # Optimization: iterate through neighbors of u
            v = 0
            # We can just loop v from 0 to n-1
            for v in range(n):
                if (au >> v) & 1:
                    if au & adj[v]:
                        return False
        return True

    def exact_alpha(adj, n):
        """
        Exact maximum independent set size using Bron-Kerbosch on complement.
        Returns the size.
        """
        full_mask = (1 << n) - 1
        comp_adj = [0] * n
        for u in range(n):
            comp_adj[u] = (full_mask ^ adj[u]) & ~(1 << u)
        
        max_clique_size = 0
        
        # Use a recursive search with pruning
        # To speed up, we can use a greedy coloring bound on the complement graph (which is G)
        # But for n <= 40, a well-implemented BK is often fast enough if the graph is structured.
        
        def bronk(R, P, X):
            nonlocal max_clique_size
            if not P and not X:
                if len(R) > max_clique_size:
                    max_clique_size = len(R)
                return
            
            # Pruning
            if len(R) + P.bit_count() <= max_clique_size:
                return
            
            # Pivot selection: pick u from P U X that maximizes |P & N(u)| in complement
            # N(u) in complement is comp_adj[u]
            # We want to minimize the number of branches, so pick u with max degree in P
            # Actually, standard BK picks u to maximize |P \ N(u)|? No, to minimize |P \ N(u)|?
            # The loop is over P \ N(u). So we want N(u) to be as large as possible in P.
            # So pick u that maximizes (P & comp_adj[u]).bit_count()
            
            all_px = P | X
            if not all_px:
                return
                
            best_u = -1
            max_deg = -1
            # Iterate through candidates for pivot
            # For small n, we can just check all
            temp_px = all_px
            while temp_px:
                lsb = temp_px & -temp_px
                u = lsb.bit_length() - 1
                deg = (P & comp_adj[u]).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_u = u
                temp_px ^= lsb
            
            if best_u == -1:
                return

            # Iterate over P \ N(best_u)
            P_iter = P & ~comp_adj[best_u]
            
            while P_iter:
                v_lsb = P_iter & -P_iter
                v = v_lsb.bit_length() - 1
                
                bronk(R | (1 << v), P & comp_adj[v], X & comp_adj[v])
                
                P ^= v_lsb
                X |= v_lsb
                P_iter ^= v_lsb

        bronk(0, (1 << n) - 1, 0)
        return max_clique_size

    def is_valid(adj, n):
        if not check_triangle_free(adj, n):
            return False
        if exact_alpha(adj, n) >= 10:
            return False
        return True

    def build_andrasfai(k):
        """
        Construct Andrasfai graph A_k.
        Vertices 0..3k-1.
        Edge (i, j) if (i-j) mod 3k is in {1, 3k-1}.
        This graph is triangle-free and has independence number k.
        For k=9, n=27, alpha=9.
        """
        n = 3 * k
        adj = [0] * n
        for i in range(n):
            # Neighbors are i+1 and i-1 mod n
            adj[i] |= 1 << ((i + 1) % n)
            adj[i] |= 1 << ((i - 1) % n)
        return adj, n

    def try_local_search(adj, n, max_steps=1000):
        """
        Local search to improve the graph.
        Goal: Maintain triangle-free, minimize alpha.
        If we start with a valid graph (alpha < 10), we might try to increase n?
        No, n is fixed in this function.
        We use this to repair random graphs or refine seeds.
        """
        current_adj = adj[:]
        
        # Check initial state
        if not check_triangle_free(current_adj, n):
            # Try to remove edges to break triangles
            # Find a triangle and remove an edge
            changed = True
            while changed and time_left() > 1:
                changed = False
                # Find triangle
                tri_found = False
                for u in range(n):
                    au = current_adj[u]
                    for v in range(u + 1, n):
                        if (au >> v) & 1:
                            if au & current_adj[v]:
                                # Triangle u-v-w exists
                                # Find w
                                common = au & current_adj[v]
                                if common:
                                    w_lsb = common & -common
                                    w = w_lsb.bit_length() - 1
                                    # Remove edge (v, w)
                                    current_adj[v] ^= 1 << w
                                    current_adj[w] ^= 1 << v
                                    tri_found = True
                                    changed = True
                                    break
                    if tri_found:
                        break
                if not tri_found:
                    break
            
            if not check_triangle_free(current_adj, n):
                return None, 999

        # Now triangle-free. Minimize alpha.
        alpha = exact_alpha(current_adj, n)
        best_alpha = alpha
        best_adj = current_adj[:]
        
        for step in range(max_steps):
            if time_left() < 1:
                break
                
            # Randomly flip an edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip
            current_adj[u] ^= 1 << v
            current_adj[v] ^= 1 << u
            
            if not check_triangle_free(current_adj, n):
                # Revert
                current_adj[u] ^= 1 << v
                current_adj[v] ^= 1 << u
                continue
            
            new_alpha = exact_alpha(current_adj, n)
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_adj = current_adj[:]
            else:
                # Revert with some probability (simulated annealing style, but simple hill climbing for now)
                # If we want to explore, we might keep it if new_alpha == best_alpha
                # But for now, revert if not improved
                current_adj[u] ^= 1 << v
                current_adj[v] ^= 1 << u
        
        return best_adj, best_alpha

    # Strategy:
    # 1. Start with Andrasfai A_9 (n=27, alpha=9). This is a valid base.
    # 2. Try to extend it to n=28, 29, 30... by adding vertices.
    #    When adding a vertex, we must ensure no triangles and alpha doesn't increase.
    #    Adding a vertex v:
    #    - Choose neighbors S subset of existing vertices.
    #    - Condition 1: No triangles. This means S must be an independent set in the existing graph?
    #      No, if u, w in S, and u~w in existing graph, then u-v-w is a triangle.
    #      So S must be an independent set in the existing graph.
    #    - Condition 2: Alpha does not increase.
    #      If we add v, new alpha = max(old_alpha, 1 + max_independent_set_in_non_neighbors).
    #      Non-neighbors of v are V \ (S U {v}).
    #      We need 1 + alpha(G[Non-neighbors]) <= 9.
    #      So alpha(G[Non-neighbors]) <= 8.
    
    # Let's try to build up from A_9.
    
    base_adj, base_n = build_andrasfai(9)
    
    # Verify base
    if is_valid(base_adj, base_n):
        best_n = base_n
        best_edges = get_edges_from_adj(base_adj, base_n)
    else:
        # Should not happen, but fallback
        best_n = 26
        best_edges = []
        
    # Try to extend
    current_adj = base_adj[:]
    current_n = base_n
    
    for target_n in range(current_n + 1, 36):
        if time_left() < 5:
            break
            
        found_extension = False
        
        # Try different placements for the new vertex
        # We need to choose an independent set S in current graph such that
        # alpha(current_graph[Non-S]) <= 8.
        
        # Heuristic: Try small S first? Or S that maximizes the "separation"?
        # Actually, if S is small, Non-S is large, so alpha(Non-S) is likely large.
        # We want S to be large enough to "cover" the independence.
        # But S must be independent.
        
        # Let's iterate over possible independent sets S? Too many.
        # Instead, let's try adding the vertex and see if we can fix it.
        # Or, use a specific construction:
        # The Andrasfai graph A_9 is a circulant.
        # Maybe we can just try random independent sets S of size 1, 2, 3...
        
        # Get all independent sets of size up to 3?
        # Or just try random subsets S that are independent.
        
        for attempt in range(200):
            if time_left() < 2:
                break
            
            # Pick a random independent set S
            # Start with empty, add vertices that are independent of current S
            S = set()
            candidates = list(range(current_n))
            random.shuffle(candidates)
            
            # Try to build a random independent set
            for v in candidates:
                # Check if v is independent of all in S
                is_ind = True
                for s in S:
                    if (current_adj[v] >> s) & 1:
                        is_ind = False
                        break
                if is_ind:
                    S.add(v)
                    if len(S) > 4: # Keep S small to keep Non-S manageable?
                        break
            
            if not S:
                continue
            
            # Check if S is independent (should be by construction)
            # Check alpha of Non-S
            non_S = [v for v in range(current_n) if v not in S]
            if not non_S:
                # S is all vertices, alpha(Non-S) = 0. New alpha = 1.
                # This is valid if S is independent.
                pass
            
            # Build subgraph induced by non_S
            # We need to check if alpha of this subgraph <= 8.
            # For efficiency, we can just check if there is an independent set of size 9 in non_S.
            # If yes, invalid.
            
            # Fast check: Does non_S contain an independent set of size 9?
            # This is equivalent to: Does the complement of G[non_S] contain a clique of size 9?
            # Or just run a limited search for independent set of size 9.
            
            # Let's just compute exact alpha of the subgraph if it's small enough,
            # or use a heuristic.
            # For now, let's just try to add the vertex and run local search to fix alpha.
            
            # Add vertex current_n
            new_adj = current_adj[:] + [0]
            for s in S:
                new_adj[current_n] |= 1 << s
                new_adj[s] |= 1 << current_n
            
            # Check triangle-free (S is independent, so no triangles involving new vertex)
            # Check alpha
            if is_valid(new_adj, current_n + 1):
                current_adj = new_adj
                current_n += 1
                found_extension = True
                if current_n > best_n:
                    best_n = current_n
                    best_edges = get_edges_from_adj(current_adj, current_n)
                break
        
        if not found_extension:
            # Try local search on the current graph to see if we can improve alpha?
            # No, we are stuck at this n.
            # Maybe try a different random seed for S?
            # If we fail to extend, we might try to "restructure" the whole graph?
            # For now, stop extending.
            break
    
    # If we didn't find a large graph, fallback to a known good one or best found
    if best_n < 27:
        # Fallback to A_9
        base_adj, base_n = build_andrasfai(9)
        best_n = base_n
        best_edges = get_edges_from_adj(base_adj, base_n)
        
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
