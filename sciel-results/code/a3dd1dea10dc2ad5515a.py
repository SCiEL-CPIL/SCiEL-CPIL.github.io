import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges
    
    def solve_instance(edges):
        if not edges:
            return []
        
        # Map edge to index for faster lookup if needed, but we work with tuples
        # Calculate disjointness count for sorting
        # disjoint_count[i] is number of edges disjoint from edges[i]
        n_edges = len(edges)
        disjoint_counts = [0] * n_edges
        
        # Precompute adjacency in complement line graph? 
        # For n=64, E~1100. E^2 = 1.2M. 
        # Let's just compute counts first.
        for i in range(n_edges):
            u1, v1 = edges[i]
            cnt = 0
            for j in range(n_edges):
                if i == j: continue
                u2, v2 = edges[j]
                if u1 != u2 and u1 != v2 and v1 != u2 and v1 != v2:
                    cnt += 1
            disjoint_counts[i] = cnt
            
        # Sort indices by disjoint count descending
        # High disjoint count means the edge is "far" from others, 
        # so it's more constrained in the coloring of the complement graph?
        # In coloring G_complement, vertices with high degree are harder to color.
        # So we process them first.
        order = sorted(range(n_edges), key=lambda x: disjoint_counts[x], reverse=True)
        
        colors = [] # List of lists of edge indices
        
        for idx in order:
            e = edges[idx]
            u, v = e
            placed = False
            for c_idx, clique in enumerate(colors):
                # Check if e intersects all edges in clique
                ok = True
                for j in clique:
                    u2, v2 = edges[j]
                    if u != u2 and u != v2 and v != u2 and v != v2:
                        ok = False
                        break
                if ok:
                    clique.append(idx)
                    placed = True
                    break
            if not placed:
                colors.append([idx])
                
        # Convert indices to vertex lists
        result = []
        for clique in colors:
            # The clique is a set of edge indices. 
            # The output requires "vertex lists". 
            # Wait, the task says: "two lists of cliques (vertex lists)".
            # "Every listed set must be a clique with at least two vertices"
            # This implies the output should be lists of vertices that form a clique in G.
            # If we have a set of edges that form a clique in L(G) (pairwise intersecting), 
            # do they always form a clique in G?
            # A clique in L(G) is a set of edges in G that are pairwise intersecting.
            # Does a set of pairwise intersecting edges in G always correspond to a clique in G?
            # No. Example: Triangle (1,2), (2,3), (1,3). They are pairwise intersecting. 
            # The vertices are {1,2,3}. They form a clique in G.
            # Example: Star (1,2), (1,3). Pairwise intersecting at 1. Vertices {1,2,3}. 
            # Are 2 and 3 connected? Not necessarily. 
            # So the set of edges (1,2), (1,3) is a clique in L(G) but the vertices {1,2,3} 
            # might not be a clique in G if edge (2,3) is missing.
            # The task requires: "Every listed set must be a clique ... in G".
            # This means the set of vertices must be a clique in G.
            # So we cannot just output the vertices of the edges in the L(G) clique 
            # unless those vertices actually form a clique in G.
            
            # Re-reading the task:
            # "Clique partitions of two fixed dense random graphs"
            # "Every listed set must be a clique with at least two vertices and every edge must lie in exactly one listed clique."
            # This implies the partition is a partition of the *edges* into sets, where each set is a *clique in G*.
            # If a set of edges forms a clique in G, then all edges in that set must be part of a complete subgraph (clique) on the vertices involved.
            # So, if we pick a set of edges, the union of their vertices must induce a complete graph, AND every edge in the set must be present in G.
            # Actually, if the set of vertices induces a complete graph, then *all* edges between them are in G. 
            # But the partition must cover *every edge* exactly once.
            # This implies that if we assign an edge (u,v) to a clique K, then K must be a clique in G containing u and v.
            # And if we assign another edge (u,w) to the same clique K, then w must be in K, and (u,w) must be in G (which it is), and (v,w) must be in G (so K is a clique).
            # So, essentially, we are partitioning the edges of G into cliques of G.
            # This is the same as finding a clique cover of the line graph L(G) where the cliques in L(G) must correspond to cliques in G?
            # No, a clique in L(G) is just pairwise intersecting edges. 
            # If we have pairwise intersecting edges (1,2), (1,3), (2,3), they form a triangle in G.
            # If we have (1,2), (1,3), they are pairwise intersecting. The vertices are {1,2,3}. 
            # If (2,3) is not in G, then {1,2,3} is not a clique in G. 
            # But the edge (1,2) and (1,3) are in G. 
            # Can we put them in the same "clique" for the partition?
            # The "clique" in the partition is a set of vertices. 
            # If the set of vertices is {1,2}, it contains edge (1,2). 
            # If the set of vertices is {1,3}, it contains edge (1,3).
            # We cannot have a set of vertices {1,2,3} if (2,3) is missing, because it wouldn't be a clique.
            # So, the partition must consist of vertex sets that are cliques in G.
            # Every edge must be in exactly one such vertex set.
            # This means if edge (u,v) is in clique K, then u,v in K and K is a clique in G.
            # This is equivalent to: Partition the edge set into subsets, such that each subset is the edge set of a clique in G.
            # Note: A clique K in G has $\binom{|K|}{2}$ edges. All these edges must be assigned to K.
            # So we are looking for a collection of cliques in G such that every edge of G belongs to exactly one clique.
            
            # This is different from coloring L(G). 
            # Coloring L(G) allows (1,2) and (1,3) to have the same color even if (2,3) is missing, 
            # because they intersect. But in our problem, if we assign them to the same "clique", 
            # that clique must be a valid clique in G. The smallest clique containing (1,2) and (1,3) 
            # is {1,2,3}, which requires (2,3). 
            # So, we can only group edges into a clique if they all belong to the same maximal clique in G?
            # No, they just need to belong to *some* clique. 
            # If we group (1,2) and (1,3), the clique must contain 1,2,3. So (2,3) must be in G.
            # So, effectively, we can only group edges that are part of the same "clique structure".
            # Specifically, if we assign edge e to a clique C, C must be a clique in G.
            # And all edges of C must be assigned to C? 
            # "Every edge must lie in exactly one listed clique."
            # If C is a clique, it has edges. If we list C, do we claim all edges of C?
            # Yes, "every edge must lie in exactly one listed clique". 
            # If C is listed, and e is an edge of C, then e lies in C. 
            # If e lies in C, it cannot lie in any other listed clique.
            # So, if we list a clique C, we "consume" all edges of C.
            # We need to select a set of cliques in G such that their edge sets form a partition of E(G).
            
            # This is a much harder problem. 
            # Greedy approach:
            # 1. Find all maximal cliques in G? Too many.
            # 2. Greedily pick a clique, remove its edges, repeat.
            # To minimize the number of cliques, we want large cliques.
            # So, at each step, find the largest clique in the remaining graph, add it to the partition, and remove its edges.
            
            # Let's implement this.
            
            if time.time() > deadline:
                # Fallback to trivial solution (each edge is a clique of size 2? No, must be clique in G. 
                # If we pick {u,v}, it's a clique. 
                # So the trivial solution is just all edges as K2 cliques.
                # But we need to return the best result found.
                # If we haven't started, return trivial.
                pass

        # Since the previous logic inside `solve_instance` was for coloring L(G) which is incorrect for the "vertex clique" requirement, 
        # I will rewrite `solve_instance` to use the "Greedy Maximal Clique" approach.
        
        # However, finding the *largest* clique is NP-hard. 
        # For n=64, we can use a heuristic.
        # Bron-Kerbosch with pivot to find all maximal cliques, then pick the largest? 
        # Or just a greedy heuristic: 
        # 1. Start with an empty set of cliques.
        # 2. While edges remain:
        #    a. Find a large clique in the remaining graph.
        #    b. Add it to the partition.
        #    c. Remove its edges from the remaining graph.
        
        # Heuristic for finding a large clique:
        # - Use a greedy vertex addition: 
        #   Start with a vertex of high degree.
        #   Add neighbors that are connected to all current vertices.
        #   Repeat until no more can be added.
        # - Try this from multiple starting vertices or random restarts?
        
        # Given the time limit, let's try a simple greedy:
        # 1. Compute degrees.
        # 2. Pick vertex with max degree.
        # 3. Build clique greedily.
        # 4. Remove edges.
        
        # This might be slow if we do it naively.
        # Let's optimize.
        
        # Actually, let's look at the score 3.146 from p41. 
        # The baseline is 1.0. 3.146 is much better. 
        # The baseline returns edges as cliques. 
        # If we can find even a few triangles, we reduce the count.
        
        # Let's implement the greedy clique removal.
        
        edges_set = set(edges)
        remaining_edges = set(edges)
        result_cliques = []
        
        # Adjacency list for faster intersection checks
        adj = {u: set() for u in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Helper to check if a set of vertices is a clique
        # and get its edges
        def get_clique_edges(vertices):
            verts = sorted(vertices)
            es = []
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    u, v = verts[i], verts[j]
                    if v in adj[u]:
                        es.append((u, v))
                    else:
                        return None # Not a clique
            return es
            
        # Greedy clique finder
        def find_greedy_clique(remaining_edges, adj):
            if not remaining_edges:
                return []
            
            # Pick a start vertex
            # Heuristic: vertex with max degree in remaining graph?
            # Let's just pick the first edge's u
            # Or better, find a dense area.
            
            # Let's try starting from each vertex in the first few edges?
            # Or just a simple greedy:
            # 1. Start with an edge (u,v) from remaining.
            # 2. Try to extend it.
            
            # A better heuristic: 
            # Sort vertices by degree in remaining graph descending.
            # Try to build a clique starting from the highest degree vertex.
            
            deg = {u: len(adj[u] & {x for x in range(n) if (min(u,x), max(u,x)) in remaining_edges or (min(x,u), max(x,u)) in remaining_edges}) for u in range(n)}
            # This calculation is expensive. 
            # Let's just use a simple local search.
            
            # Pick a random edge from remaining to start?
            # Let's pick the edge with the most common neighbors?
            
            # For simplicity and speed, let's just try to extend a clique from a vertex.
            # We'll try a few times.
            
            best_clique = []
            
            # Get a list of vertices present in remaining edges
            verts_in_rem = set()
            for u, v in remaining_edges:
                verts_in_rem.add(u)
                verts_in_rem.add(v)
                
            if not verts_in_rem:
                return []
                
            # Try starting from each vertex in a subset of verts_in_rem
            # To save time, limit the number of attempts
            candidates = list(verts_in_rem)
            # Sort by degree in original graph?
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            
            for start_v in candidates[:10]: # Try top 10
                current_clique = {start_v}
                # Neighbors of start_v in remaining graph
                # A neighbor w is valid if (start_v, w) is in remaining_edges
                # And w is connected to all in current_clique
                
                # Actually, we just need to find a clique in the subgraph induced by remaining_edges.
                
                # Greedy extension:
                # Maintain a set of candidates that are connected to all in current_clique.
                candidates_set = set(adj[start_v])
                # Filter candidates to those connected to start_v via remaining edge
                candidates_set = {v for v in candidates_set if (min(start_v, v), max(start_v, v)) in remaining_edges}
                
                while candidates_set:
                    # Pick a candidate that is connected to all in current_clique
                    # Since we maintain the property, all in candidates_set are connected to all in current_clique.
                    # Pick the one with highest degree in candidates_set?
                    # Or just pick the first one.
                    next_v = next(iter(candidates_set))
                    current_clique.add(next_v)
                    # Update candidates_set: must be connected to next_v
                    # And must be in remaining_edges with next_v
                    new_candidates = set()
                    for v in candidates_set:
                        if v == next_v: continue
                        if (min(next_v, v), max(next_v, v)) in remaining_edges:
                            # Also must be connected to all other in current_clique?
                            # Yes, but they were already checked when they were added to candidates_set.
                            # Wait, if we add next_v, the new candidates must be connected to next_v.
                            # They were already connected to all previous clique members.
                            new_candidates.add(v)
                    candidates_set = new_candidates
                    
                if len(current_clique) > len(best_clique):
                    best_clique = list(current_clique)
                    
            if not best_clique:
                # Fallback: just pick one edge
                e = next(iter(remaining_edges))
                best_clique = [e[0], e[1]]
                
            return best_clique

        while remaining_edges:
            if time.time() > deadline:
                break
                
            clique_verts = find_greedy_clique(remaining_edges, adj)
            if not clique_verts:
                break
                
            # Get edges of this clique
            clique_edges = get_clique_edges(clique_verts)
            if not clique_edges:
                # Should not happen if find_greedy_clique is correct
                continue
                
            # Check if all these edges are in remaining_edges
            # They should be, by construction.
            
            result_cliques.append(clique_verts)
            
            # Remove these edges from remaining
            for e in clique_edges:
                remaining_edges.discard(e)
                
        # If we didn't finish due to deadline, add remaining edges as K2 cliques
        if remaining_edges:
            for u, v in remaining_edges:
                result_cliques.append([u, v])
                
        return result_cliques

    # Generate instances
    res1 = solve_instance(instance(32, 0.6, 601))
    res2 = solve_instance(instance(64, 0.55, 602))
    
    return [res1, res2]

construct()
