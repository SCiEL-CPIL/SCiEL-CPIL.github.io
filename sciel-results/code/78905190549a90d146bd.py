"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time
    import sys

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def max_independent_set(nodes, conflict_edges):
        """
        Find max independent set in a graph defined by nodes and conflict_edges.
        nodes: list of vertices (subset of 0..k-1)
        conflict_edges: set of tuples (u, v) indicating u and v cannot both be in IS.
        Returns a list of vertices in the max IS.
        """
        if not nodes:
            return []
        
        # Optimization: If no conflicts, all nodes are in IS
        if not conflict_edges:
            return list(nodes)
            
        # Convert nodes to 0-indexed for bitmask or list indices?
        # Since k <= 30, we can use the actual vertex IDs for a dict-based approach or bitmask if small.
        # Given k=30, a simple backtracking with pruning is safer for Python speed unless N is very small.
        # Let's use a standard backtracking solver for Max IS.
        
        # Build adjacency sets for the conflict graph restricted to 'nodes'
        # Only consider edges where both endpoints are in 'nodes'
        relevant_edges = set()
        for e in conflict_edges:
            if e[0] in nodes_set and e[1] in nodes_set:
                relevant_edges.add(e)
        
        if not relevant_edges:
            return list(nodes)
            
        adj = {v: set() for v in nodes}
        for u, v in relevant_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Heuristic: Order nodes by degree desc to find conflicts early?
        # Or just use a greedy initialization and branch?
        
        best_set = set()
        
        def backtrack(candidates, current_set):
            nonlocal best_set
            
            if not candidates:
                if len(current_set) > len(best_set):
                    best_set = set(current_set)
                return
                
            # Pruning: if len(current_set) + len(candidates) <= len(best_set), return
            if len(current_set) + len(candidates) <= len(best_set):
                return
                
            # Pick a vertex with max degree in the induced subgraph?
            # For simplicity, pick the first one, but sorting by degree helps.
            # Let's pick v with max degree in 'candidates' restricted to adj
            max_deg = -1
            chosen = None
            # To avoid O(N^2) in this loop every time, we might just pick the first one.
            # But for Max IS, branch and bound works well with good ordering.
            for v in candidates:
                deg = len([n for n in candidates if n in adj[v]])
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
                    
            if chosen is None:
                return

            # Branch 1: Include 'chosen'
            new_current = list(current_set) + [chosen]
            # New candidates must not be adjacent to 'chosen'
            new_candidates = [n for n in candidates if n not in adj[chosen] and n != chosen]
            # Optimization: We can remove 'chosen' from new_candidates implicitly
            
            backtrack(new_candidates, new_current)
            
            # Branch 2: Exclude 'chosen'
            backtrack([n for n in candidates if n != chosen], current_set)

        nodes_set = set(nodes)
        backtrack(list(nodes), [])
        return list(best_set)

    def build_partition(n, k, p, seed):
        start_time = time.time()
        deadline = start_time + 115  # Safety margin
        
        # Generate graph
        rng = Random(seed)
        cross_adj = [set() for _ in range(n-k)]  # For independent vertices k..n-1
        
        for i, v in enumerate(range(k, n)):
            for u in range(k):
                if rng.random() < p:
                    cross_adj[i].add(u)
        
        # Independent vertices indices: 0..n-k-1 correspond to k..n-1
        # We want to process them.
        
        # 1. Prepare data for Greedy Selection
        # Sort independent vertices by size of neighborhood (descending)
        indeps = []
        for i in range(n - k):
            neighbors = cross_adj[i]
            indeps.append((len(neighbors), i, neighbors))
        
        indeps.sort(key=lambda x: x[0], reverse=True)
        
        covered_internal_edges = set()
        cliques = []
        
        # 2. Process each independent vertex
        for _, idx, neighbors in indeps:
            if time.time() > deadline - 5:
                # Fallback for remaining: just add 2-cliques
                for u in neighbors:
                    if (u, k + idx) not in [(a, b) for c in cliques for a in c for b in c if a < b]: # This check is slow, just assume valid for fallback?
                         # Fallback logic is tricky if we've already added some.
                         # Better: if we hit deadline, we should have already added safe cliques.
                         # For safety, just add the 2-cliques for unprocessed.
                         pass
            
            v = k + idx
            
            if not neighbors:
                continue
            
            # Determine Safe Set S_subset of neighbors
            # Conflict if internal edge (u, w) is already covered
            conflict_edges_for_v = set()
            for i in range(len(neighbors)):
                u = neighbors[i]
                for j in range(i + 1, len(neighbors)):
                    w = neighbors[j]
                    e = (min(u, w), max(u, w))
                    if e in covered_internal_edges:
                        conflict_edges_for_v.add((u, w))
            
            # Find Max IS in conflict graph
            # If no conflicts, S = all neighbors
            if not conflict_edges_for_v:
                S = list(neighbors)
            else:
                # Use IS finder
                # Note: neighbors is a set. max_independent_set expects list
                S = max_independent_set(list(neighbors), conflict_edges_for_v)
            
            # Add Clique {v} U S
            if S:
                clique = [v] + S
                cliques.append(clique)
                
                # Update covered_internal_edges
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        e = (min(S[i], S[j]), max(S[i], S[j]))
                        covered_internal_edges.add(e)
            
            # Handle remaining neighbors R = neighbors - S
            # These cross edges (v, u) must be covered.
            # We use 2-cliques {v, u}
            S_set = set(S)
            for u in neighbors:
                if u not in S_set:
                    cliques.append([v, u])
                    # No internal edges covered by 2-cliques involving independent vertex
                    # (Only 1 vertex from C)
            
        # 3. Cover remaining internal edges
        # All internal edges of K_k
        all_internal = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_internal.add((u, w))
        
        remaining_edges = all_internal - covered_internal_edges
        
        if not remaining_edges:
            return cliques
        
        # Greedy Clique Cover on remaining graph
        # Nodes: 0..k-1
        # Edges: remaining_edges
        
        # Build adj for remaining
        rem_adj = {u: set() for u in range(k)}
        for u, w in remaining_edges:
            rem_adj[u].add(w)
            rem_adj[w].add(u)
            
        active_nodes = set(range(k))
        
        while remaining_edges:
            if time.time() > deadline - 2:
                # Emergency fallback: add all remaining as 2-cliques
                for u, w in remaining_edges:
                    cliques.append([u, w])
                break
                
            # Find a "max" clique in remaining graph
            # Heuristic: Start with highest degree node
            # Sort active nodes by degree in rem_adj
            degs = [(len(rem_adj[u]), u) for u in active_nodes if rem_adj[u]]
            if not degs:
                break # Should not happen if remaining_edges not empty
            degs.sort(reverse=True)
            
            start_node = degs[0][1]
            current_clique = [start_node]
            current_clique_set = {start_node}
            
            # Greedily add neighbors that are connected to all in current_clique
            # Check candidates
            candidates = list(rem_adj[start_node])
            # Sort candidates by degree?
            candidates.sort(key=lambda x: len(rem_adj[x]), reverse=True)
            
            for u in candidates:
                if u not in current_clique_set:
                    # Check if u is connected to all in current_clique
                    is_connected = True
                    for v in current_clique:
                        if u not in rem_adj[v]:
                            is_connected = False
                            break
                    if is_connected:
                        current_clique.append(u)
                        current_clique_set.add(u)
            
            # Add this clique
            if len(current_clique) >= 2:
                cliques.append(sorted(current_clique))
                
                # Remove edges within this clique from remaining
                to_remove = []
                for i in range(len(current_clique)):
                    for j in range(i + 1, len(current_clique)):
                        e = (min(current_clique[i], current_clique[j]), max(current_clique[i], current_clique[j]))
                        if e in remaining_edges:
                            to_remove.append(e)
                
                for e in to_remove:
                    remaining_edges.remove(e)
                    u, w = e
                    rem_adj[u].remove(w)
                    rem_adj[w].remove(u)
            
            # Remove used nodes from active? No, they can be used in other cliques?
            # Wait, clique cover means every edge in exactly one clique.
            # Vertices can be in multiple cliques.
            # But if a node has no edges left, it doesn't need to be in any more.
            # So we just continue the loop on remaining_edges.
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > 110:
            break
        results.append(build_partition(*spec))
    
    return results
# EVOLVE-BLOCK-END
