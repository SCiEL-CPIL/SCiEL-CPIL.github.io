"""Clique partition of split graphs using structural properties and greedy clique partitioning."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def find_max_clique(adj, vertices):
        """
        Find a maximum clique in the subgraph induced by 'vertices' using 'adj' (adjacency sets).
        Simple greedy/backtracking for small k.
        """
        if not vertices:
            return []
        
        # Heuristic: Order vertices by degree in the induced subgraph, descending
        # For simplicity and speed with k<=30, we can use a recursive branch and bound
        # or just a simple greedy approach if exact is too slow.
        # Given k=30, exact is fine.
        
        best_clique = []
        
        def backtrack(candidates, current_clique):
            nonlocal best_clique
            if not candidates:
                if len(current_clique) > len(best_clique):
                    best_clique = current_clique[:]
                return
            
            # Pruning: if len(current_clique) + len(candidates) <= len(best_clique), return
            if len(current_clique) + len(candidates) <= len(best_clique):
                return
                
            # Try adding each vertex in candidates
            # Order candidates by degree to find large cliques faster
            # We need to be careful with recursion depth or use iterative.
            # k is small, so recursion is fine.
            
            for v in candidates:
                # New candidates are neighbors of v that are also in candidates
                new_candidates = [u for u in candidates if u > v and u in adj[v]]
                # Wait, standard Bron-Kerbosch uses sets.
                # Let's use a simpler recursive approach:
                # Pick v. New clique = current + [v].
                # New candidates = candidates intersected with neighbors of v.
                
                # To avoid duplicates, we enforce ordering: v must be greater than last added?
                # Or just use a set of candidates and remove v.
                
                # Let's stick to a simple recursive function that takes a list of candidates.
                # To ensure we don't re-explore, we process candidates in a specific order.
                
                # Actually, a standard maximum clique algorithm:
                # R = current clique
                # P = candidates
                # X = excluded (not needed if we just want one max clique, but helps for pruning)
                
                # Let's use a simple heuristic:
                # 1. Sort candidates by degree in P.
                # 2. Pick the one with highest degree.
                # 3. Recurse with P intersected with neighbors.
                
                # For k=30, a simple greedy might be enough, but let's do a bit better.
                # We'll use a recursive search that picks the vertex with the highest degree in the current candidate set.
                
                pass # Placeholder, will implement below

        # Implementation of Max Clique for small graphs
        def max_clique_search(P):
            # P is a set of vertices
            if not P:
                return set()
            
            # If P is a clique, return it
            is_clique = True
            p_list = list(P)
            for i in range(len(p_list)):
                for j in range(i+1, len(p_list)):
                    if p_list[j] not in adj[p_list[i]]:
                        is_clique = False
                        break
                if not is_clique:
                    break
            
            if is_clique:
                return set(P)
            
            best = set()
            # Greedy choice: pick vertex with max degree in P
            # Or branch on each vertex?
            # Branching on each vertex in P is 2^|P| worst case.
            # With |P| <= 30, this is too big for naive recursion if not pruned.
            
            # Let's use a simple greedy heuristic first, then improve if needed.
            # Greedy:
            # 1. Select vertex v with max degree in P.
            # 2. New P = P intersect N(v).
            # 3. Recurse.
            # This finds A clique, not necessarily THE maximum.
            
            # For the purpose of "minimizing total cliques", finding a reasonably large clique is good.
            # Let's use a simple greedy:
            current_clique = []
            P_copy = set(P)
            while P_copy:
                # Find vertex with max degree in P_copy
                max_deg = -1
                v = -1
                for u in P_copy:
                    deg = len(adj[u] & P_copy)
                    if deg > max_deg:
                        max_deg = deg
                        v = u
                if v == -1:
                    break
                current_clique.append(v)
                P_copy = P_copy & adj[v]
            
            return set(current_clique)

        # Use the greedy max clique
        return list(max_clique_search(set(vertices)))

    def partition_split_graph(n, k, edges):
        # Build adjacency for cross edges
        cross_adj = {v: set() for v in range(k, n)}
        for u, v in edges:
            if u < k and v >= k:
                cross_adj[v].add(u)
        
        cliques = []
        
        # Step 1: Cliques for independent vertices
        covered_c_edges = set()
        for v in range(k, n):
            neighbors = cross_adj[v]
            if neighbors:
                clique = [v] + sorted(neighbors)
                cliques.append(clique)
                # Mark clique edges within neighbors as covered
                neigh_list = sorted(neighbors)
                for i in range(len(neigh_list)):
                    for j in range(i + 1, len(neigh_list)):
                        covered_c_edges.add((neigh_list[i], neigh_list[j]))
        
        # Step 2: Cover uncovered C-edges
        # Build graph H of uncovered edges
        H_adj = {u: set() for u in range(k)}
        uncovered_edges = []
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_c_edges:
                    H_adj[u].add(w)
                    H_adj[w].add(u)
                    uncovered_edges.append((u, w))
        
        if not uncovered_edges:
            return cliques
            
        # Greedy partition of H
        # We need to remove edges from H as we form cliques
        # Represent H as a set of edges
        current_H_edges = set(uncovered_edges)
        
        while current_H_edges:
            # Find vertices involved
            verts_involved = set()
            for u, w in current_H_edges:
                verts_involved.add(u)
                verts_involved.add(w)
                
            # Find a max clique in the subgraph induced by verts_involved
            # Using the greedy helper
            # We need to pass the adjacency restricted to current_H_edges
            # Let's just use the greedy logic directly here for clarity
            
            # Greedy Max Clique on current_H_edges
            # 1. Pick vertex with max degree in current_H_edges
            # 2. Restrict to its neighbors
            # 3. Repeat
            
            # Build degree map
            deg = {v: 0 for v in verts_involved}
            for u, w in current_H_edges:
                deg[u] += 1
                deg[w] += 1
            
            # Select starting vertex
            if not verts_involved:
                break
                
            start_v = max(verts_involved, key=lambda x: deg[x])
            
            # Greedy expansion
            clique_verts = [start_v]
            candidates = set(H_adj[start_v]) # Neighbors in the original H
            # Candidates must also be in verts_involved and connected to start_v in current_H_edges
            # Actually, we need to maintain the set of candidates that form a clique with current clique.
            
            # Let's use a standard greedy clique finding:
            # C = {v0}
            # P = N(v0)
            # While P not empty:
            #   Pick v in P with max degree in P
            #   Add v to C
            #   P = P intersect N(v)
            
            C = [start_v]
            P = set(H_adj[start_v])
            # Ensure P only contains vertices that are in verts_involved (they are, by definition of H_adj)
            
            while P:
                # Find v in P with max degree in P
                # Degree in P means number of neighbors in P
                max_deg = -1
                v = -1
                for u in P:
                    d = len(H_adj[u] & P)
                    if d > max_deg:
                        max_deg = d
                        v = u
                
                if v == -1:
                    break
                
                C.append(v)
                P = P & H_adj[v]
            
            # Add this clique to the result
            clique_list = sorted(C)
            cliques.append(clique_list)
            
            # Remove edges covered by this clique from current_H_edges
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    u, w = clique_list[i], clique_list[j]
                    if u > w:
                        u, w = w, u
                    if (u, w) in current_H_edges:
                        current_H_edges.remove((u, w))
                        
            # Update verts_involved for next iteration?
            # The loop condition `while current_H_edges` handles it.
            # We re-calculate verts_involved at the start of the loop.
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        results.append(cliques)
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return results
# EVOLVE-BLOCK-END
