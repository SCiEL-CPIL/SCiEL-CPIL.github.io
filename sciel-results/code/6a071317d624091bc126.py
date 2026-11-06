"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random

def construct():
    n = 16
    # Start with complete graph
    edges_set = set()
    for u in range(n):
        for v in range(u+1, n):
            edges_set.add((u, v))
    
    # Precompute subsets of size 8
    subsets = list(itertools.combinations(range(n), 8))
    
    # Q3 structure: vertices 0..7, edges if xor in {1,2,4}
    q3_adj = {i: [] for i in range(8)}
    q3_edges = []
    for i in range(8):
        for j in range(8):
            if i != j and (i ^ j) in (1, 2, 4):
                if i < j:
                    q3_edges.append((i, j))
                q3_adj[i].append(j)
                
    # Adjacency list for quick degree checks
    adj = {i: set() for i in range(n)}
    for u, v in edges_set:
        adj[u].add(v)
        adj[v].add(u)

    def has_q3(edge_set, adj):
        """
        Find a Q3 subgraph in the graph defined by edge_set.
        Returns a list of edges forming the Q3, or None.
        """
        # Shuffle subsets to avoid bias? Or just iterate.
        # Iterating in order is fine, but randomizing might help escape local structures?
        # Since we remove edges, the structure changes.
        for subset in subsets:
            subset_set = set(subset)
            # Check if induced subgraph has at least 12 edges
            edge_count = 0
            # Also check min degree in induced subgraph
            min_deg = 8
            for u in subset:
                deg_in_subset = 0
                for v in subset:
                    if u < v and (u, v) in edge_set:
                        edge_count += 1
                        deg_in_subset += 1
                        deg_in_subset += 1 # count for v? No, just count edges
                # Recalculate degree properly for pruning
                deg_u = 0
                for v in subset:
                    if u != v and (min(u,v), max(u,v)) in edge_set:
                        deg_u += 1
                if deg_u < 3:
                    min_deg = 0
                    break
            if min_deg == 0:
                continue
            if edge_count < 12:
                continue
                
            # Backtracking search for Q3 embedding
            # Map Q3 vertices (0..7) to subset vertices
            mapping = {}
            
            # Order Q3 vertices to prune: 
            # All Q3 vertices have degree 3. 
            # But we can fix the first vertex to reduce symmetry.
            # Let's just iterate subset vertices for the first Q3 vertex.
            
            def backtrack(idx):
                if idx == 8:
                    return True
                q_v = idx
                for g_v in subset:
                    if g_v in mapping.values():
                        continue
                    # Check if g_v has enough neighbors in the already mapped Q3 neighbors
                    compatible = True
                    for q_neighbor in q3_adj[q_v]:
                        if q_neighbor in mapping:
                            u, v = (mapping[q_neighbor], g_v)
                            if u > v: u, v = v, u
                            if (u, v) not in edge_set:
                                compatible = False
                                break
                    if compatible:
                        mapping[q_v] = g_v
                        if backtrack(idx + 1):
                            return True
                        del mapping[q_v]
                return False

            if backtrack(0):
                # Reconstruct edges
                q3_found_edges = []
                for i in range(8):
                    for j in q3_adj[i]:
                        if i < j:
                            u, v = mapping[i], mapping[j]
                            if u > v: u, v = v, u
                            q3_found_edges.append((u, v))
                return q3_found_edges
        return None

    # Greedy removal
    # Limit iterations to ensure time bound
    # 120s limit. 
    # If has_q3 is ~1ms, we can do ~1000 calls.
    # But has_q3 can be slower.
    max_iterations = 500
    for _ in range(max_iterations):
        q3_edges = has_q3(edges_set, adj)
        if q3_edges is None:
            break
        
        # Find the edge in q3_edges to remove
        # Heuristic: remove edge with max degree sum in current graph
        best_edge = None
        best_score = -1
        for u, v in q3_edges:
            score = len(adj[u]) + len(adj[v])
            if score > best_score:
                best_score = score
                best_edge = (u, v)
        
        if best_edge:
            u, v = best_edge
            edges_set.remove(best_edge)
            adj[u].remove(v)
            adj[v].remove(u)
            
    return [list(e) for e in edges_set]
# EVOLVE-BLOCK-END
