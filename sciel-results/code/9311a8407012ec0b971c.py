import time
import random

def construct():
    start_time = time.time()
    
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = generate_graph(n, p, seed)
        if not edges:
            return []
        
        # Adjacency representation for quick lookup
        # adj[i] is a set of neighbors
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        active_edges = set(edges)
        cliques = []
        
        # To speed up, we can iterate through edges, but we need to be careful
        # that we don't pick an edge that has already been covered.
        # A simple way is to maintain a list of active edges and pop from it.
        # But checking if an edge is active is O(1) with a set.
        
        # We will repeatedly find a maximal clique and remove its edges.
        # To make the cliques large, we prioritize edges with high degree sums.
        
        while active_edges:
            # Pick an edge to start the clique.
            # Heuristic: pick the edge (u,v) where deg(u)+deg(v) is maximal among active edges.
            # For small N, we can just scan.
            best_edge = None
            best_score = -1
            for e in active_edges:
                u, v = e
                score = len(adj[u]) + len(adj[v])
                if score > best_score:
                    best_score = score
                    best_edge = e
            
            if best_edge is None:
                break
                
            u, v = best_edge
            # Build a maximal clique containing u and v
            clique = [u, v]
            # Candidates must be neighbors of all vertices in clique
            # Initially neighbors of u and v
            candidates = adj[u] & adj[v]
            candidates.discard(u)
            candidates.discard(v)
            
            # Greedily add vertices
            # We want to keep candidates that are connected to all current clique members
            # And we want to add the "best" ones first.
            
            # To ensure maximality, we process candidates.
            # We can sort candidates by degree to get larger cliques earlier.
            # But we must be careful: adding a vertex restricts future candidates.
            
            # Let's just iterate through candidates and add if possible.
            # To make it maximal, we should check all potential additions.
            # A simpler greedy:
            # While there is a candidate connected to all in clique:
            #   Add it.
            #   Update candidates: intersect with neighbors of the new vertex.
            
            # However, this simple intersection might miss a larger clique if we pick the wrong order?
            # Actually, for a fixed starting set, the greedy expansion where we intersect neighborhoods
            # yields a maximal clique. The size depends on the order.
            
            # Let's sort candidates by degree descending to try to add high-degree vertices first.
            # Note: adj sets are dynamic. We should use current degree.
            
            while candidates:
                # Pick candidate with highest degree
                w = max(candidates, key=lambda x: len(adj[x]))
                
                # Check if w is connected to all in clique
                # Since candidates are maintained as intersection of neighborhoods, 
                # they are connected to all in clique by construction if we update correctly.
                
                # Add w to clique
                clique.append(w)
                candidates.discard(w)
                
                # Update candidates: must be neighbors of w as well
                candidates = candidates & adj[w]
                
                # Safety check: ensure all in candidates are still connected to all in clique
                # This is guaranteed by the intersection property.
                
            # Now clique is maximal (no more candidates can be added)
            cliques.append(clique)
            
            # Remove all edges within this clique from active_edges and adj
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    c1, c2 = clique[i], clique[j]
                    if (c1, c2) in active_edges:
                        active_edges.remove((c1, c2))
                    elif (c2, c1) in active_edges:
                        active_edges.remove((c2, c1))
                    # Update adjacency
                    adj[c1].discard(c2)
                    adj[c2].discard(c1)
                    
            # Check time
            if time.time() - start_time > 100:
                # Fallback: add remaining edges as individual cliques to ensure validity
                for e in active_edges:
                    cliques.append([e[0], e[1]])
                break

        return cliques

    results = []
    for n, p, seed in [(32, 0.6, 601), (64, 0.55, 602)]:
        results.append(solve_instance(n, p, seed))
        
    return results
