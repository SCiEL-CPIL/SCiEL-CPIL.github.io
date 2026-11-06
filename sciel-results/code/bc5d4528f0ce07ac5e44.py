"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    n = 16
    # Start with complete graph
    edges_set = set()
    for u in range(n):
        for v in range(u+1, n):
            edges_set.add((u, v))
    
    # Q3 structure: vertices 0..7, edges if xor in {1,2,4}
    q3_adj = {i: [] for i in range(8)}
    for i in range(8):
        for j in range(8):
            if i != j and (i ^ j) in (1, 2, 4):
                q3_adj[i].append(j)
                
    def has_q3(edge_set):
        """
        Find a Q3 subgraph in the graph defined by edge_set.
        Returns a list of edges forming the Q3, or None.
        """
        for subset in itertools.combinations(range(n), 8):
            subset_set = set(subset)
            # Check if induced subgraph has at least 12 edges
            edge_count = 0
            for u in subset:
                for v in subset:
                    if u < v and (u, v) in edge_set:
                        edge_count += 1
            if edge_count < 12:
                continue
                
            # Backtracking search for Q3 embedding
            mapping = {}
            
            def backtrack(idx):
                if idx == 8:
                    return True
                q_v = idx
                for g_v in subset:
                    if g_v in mapping.values():
                        continue
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
                q3_edges = []
                for i in range(8):
                    for j in q3_adj[i]:
                        if i < j:
                            u, v = mapping[i], mapping[j]
                            if u > v: u, v = v, u
                            q3_edges.append((u, v))
                return q3_edges
        return None

    # Greedy removal
    max_iterations = 200
    for _ in range(max_iterations):
        q3_edges = has_q3(edges_set)
        if q3_edges is None:
            break
        
        # Find the edge in q3_edges to remove
        # Heuristic: remove edge with max degree sum
        best_edge = None
        best_score = -1
        for u, v in q3_edges:
            deg_u = sum(1 for w in range(n) if (min(u,w), max(u,w)) in edges_set)
            deg_v = sum(1 for w in range(n) if (min(v,w), max(v,w)) in edges_set)
            score = deg_u + deg_v
            if score > best_score:
                best_score = score
                best_edge = (u, v)
        
        if best_edge:
            edges_set.remove(best_edge)
            
    return [list(e) for e in edges_set]
# EVOLVE-BLOCK-END
