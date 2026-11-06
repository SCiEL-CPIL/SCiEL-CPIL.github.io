import time
from random import Random

def construct():
    start_time = time.time()
    
    def get_graph(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = get_graph(n, p, seed)
        if not edges:
            return []
        
        # Initialize adjacency structures
        # neighbors[u] is a set of v such that (u,v) is in remaining_edges
        neighbors = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            neighbors[u].add(v)
            neighbors[v].add(u)
            # Store edge in canonical form (min, max)
            if u > v:
                u, v = v, u
            edge_set.add((u, v))
            
        cliques = []
        
        while edge_set:
            # Check time limit
            if time.time() - start_time > 110:
                # Fallback: add all remaining edges as size-2 cliques
                for u, v in edge_set:
                    cliques.append([u, v])
                break
            
            # Find best starting edge
            # Heuristic: Maximize the number of common neighbors in the remaining graph
            # This indicates a dense subgraph where a larger clique can be formed
            best_edge = None
            max_common = -1
            
            # Iterate through all remaining edges to find the one with highest potential
            for u, v in edge_set:
                # Common neighbors of u and v in the remaining graph
                # neighbors[u] and neighbors[v] are sets of vertices
                common = len(neighbors[u] & neighbors[v])
                if common > max_common:
                    max_common = common
                    best_edge = (u, v)
            
            if not best_edge:
                # Should not happen if edge_set is not empty
                break
            
            u, v = best_edge
            # Start clique with this edge
            C = [u, v]
            # Candidates for the next vertex are common neighbors of u and v
            candidates = neighbors[u] & neighbors[v]
            
            # Greedily grow the clique
            while candidates:
                # Pick the candidate with the highest degree in the remaining graph
                # This tends to keep the clique within a dense region
                best_w = None
                max_deg = -1
                for w in candidates:
                    deg = len(neighbors[w])
                    if deg > max_deg:
                        max_deg = deg
                        best_w = w
                
                if best_w is None:
                    break
                    
                C.append(best_w)
                # Update candidates: new vertex must be connected to best_w
                candidates = candidates & neighbors[best_w]
                
            # Add the formed clique to the result
            if len(C) >= 2:
                cliques.append(C)
                
                # Remove all edges within C from the remaining graph
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        a, b = C[i], C[j]
                        if a > b:
                            a, b = b, a
                        edge_to_remove = (a, b)
                        if edge_to_remove in edge_set:
                            edge_set.remove(edge_to_remove)
                            neighbors[a].remove(b)
                            neighbors[b].remove(a)
            else:
                # Fallback if clique growth failed (should not happen for edge start)
                cliques.append([u, v])
                edge_to_remove = (u, v) if u < v else (v, u)
                if edge_to_remove in edge_set:
                    edge_set.remove(edge_to_remove)
                    neighbors[u].remove(v)
                    neighbors[v].remove(u)
                
        return cliques

    res1 = solve_instance(32, 0.6, 601)
    res2 = solve_instance(64, 0.55, 602)
    
    return [res1, res2]
