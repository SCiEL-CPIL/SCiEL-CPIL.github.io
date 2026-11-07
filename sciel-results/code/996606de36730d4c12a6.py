import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        # K is 0..k-1, I is k..n-1
        edges_k = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges_cross = []
        # Map cross edges to neighbors for each v in I
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges_cross.append((u, v))
                    neighbors[v].append(u)
        return edges_k, edges_cross, neighbors

    def solve_instance(n, k, p, seed):
        edges_k, edges_cross, neighbors = instance(n, k, p, seed)
        
        # Set of available K-edges
        # Represent as frozenset of (u, v) with u < v
        available_k_edges = set((u, v) for u, v in edges_k)
        
        cliques = []
        
        # Process each v in I
        # Sort by degree to handle high degree ones first? 
        # Or just iterate.
        # To maximize matching size, maybe process vertices with larger neighborhoods first?
        vs = sorted([v for v in range(k, n)], key=lambda v: len(neighbors[v]), reverse=True)
        
        for v in vs:
            if time.time() > deadline:
                break # Return what we have? No, must be valid.
                # If we break, we must cover remaining edges with 2-cliques to be valid.
                # But we are in the middle of processing.
                # Better to check deadline before starting a vertex.
            
            N_v = neighbors[v]
            if not N_v:
                continue
                
            # Find maximal matching in the subgraph of available_k_edges induced by N_v
            # Nodes are N_v. Edges are pairs (u, w) in available_k_edges where u, w in N_v.
            # Simple greedy matching:
            # Sort edges in induced subgraph?
            # Just iterate through pairs in N_v.
            
            matched_nodes = set()
            pairs = []
            
            # Create list of potential edges
            potential_edges = []
            for i in range(len(N_v)):
                for j in range(i + 1, len(N_v)):
                    u, w = N_v[i], N_v[j]
                    edge = (min(u, w), max(u, w))
                    if edge in available_k_edges:
                        potential_edges.append(edge)
                        
            # Greedy matching: pick an edge, remove its nodes
            # To be a bit smarter, maybe pick edges with higher "degree" in the induced subgraph?
            # For now, random or sorted.
            random.shuffle(potential_edges)
            
            for edge in potential_edges:
                u, w = edge
                if u not in matched_nodes and w not in matched_nodes:
                    pairs.append((u, w, v))
                    matched_nodes.add(u)
                    matched_nodes.add(w)
                    available_k_edges.remove(edge)
            
            # Create cliques for pairs
            for u, w, v_node in pairs:
                cliques.append([u, w, v_node])
                
            # Create cliques for unmatched neighbors (singles)
            for u in N_v:
                if u not in matched_nodes:
                    cliques.append([u, v])
                    
        # Cover remaining K-edges
        # If we broke out of loop early, we still need to cover everything.
        # The loop above only processes v in I.
        # The remaining K-edges are in available_k_edges.
        
        # Greedy clique cover for remaining K-edges
        # Since k is small, we can do this.
        remaining_edges = list(available_k_edges)
        
        while remaining_edges:
            # Find a clique in the remaining graph
            # Nodes involved in remaining_edges
            nodes = set()
            for u, v in remaining_edges:
                nodes.add(u)
                nodes.add(v)
                
            # Try to find largest clique
            # Simple: start with a node, add neighbors
            # Or just pick an edge and see if it can be extended
            # Given small size, let's just pick the first edge and try to extend
            # Or simply: cover each edge as a 2-clique if we can't find a larger one quickly.
            
            # Let's try to find a maximal clique containing the first edge
            first_edge = remaining_edges[0]
            u, w = first_edge
            
            # Candidates for extension: nodes x such that (u,x) and (w,x) are in remaining_edges
            candidates = set()
            for e in remaining_edges:
                a, b = e
                if a == u and b != w:
                    candidates.add(b)
                elif b == u and a != w:
                    candidates.add(a)
                if a == w and b != u:
                    candidates.add(b)
                elif b == w and a != u:
                    candidates.add(a)
            
            # Check which candidates form a triangle with u, w
            # i.e., (c, u) and (c, w) in remaining_edges
            valid_c = []
            for c in candidates:
                if (min(u, c), max(u, c)) in remaining_edges and (min(w, c), max(w, c)) in remaining_edges:
                    valid_c.append(c)
            
            if valid_c:
                # Pick one, say valid_c[0]
                c = valid_c[0]
                clique = [u, w, c]
                # Remove edges covered by this clique
                edges_to_remove = set()
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        a, b = clique[i], clique[j]
                        edges_to_remove.add((min(a, b), max(a, b)))
                        
                remaining_edges = [e for e in remaining_edges if e not in edges_to_remove]
                cliques.append(clique)
            else:
                # Just cover the first edge
                cliques.append([u, w])
                remaining_edges.remove(first_edge)
                
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Emergency fallback: return valid but maybe not optimal?
            # If we run out of time, we can't really do much.
            # But the algorithm is fast.
            pass
        cliques = solve_instance(*spec)
        # Convert to list of lists of ints
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)
        
    return results
