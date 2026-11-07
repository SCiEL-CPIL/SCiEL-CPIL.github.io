import time
import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    results = []
    start_time = time.time()
    deadline = start_time + 115.0

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        rng = random.Random(seed)
        
        # Build edge set
        edge_set = set()
        # C-C edges
        for u in range(k):
            for v in range(u + 1, k):
                edge_set.add(frozenset((u, v)))
        
        # C-I edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edge_set.add(frozenset((u, v)))
        
        uncovered = set(edge_set)
        cliques = []
        
        # Helper: Greedy Maximal Clique in induced subgraph of 'nodes' using 'uncovered' edges
        def get_maximal_clique(nodes):
            if not nodes:
                return set()
            
            # Helper to get neighbors in 'nodes' using 'uncovered'
            def get_neighbors(u, node_set):
                neighbors = set()
                for v in node_set:
                    if u != v and frozenset((u, v)) in uncovered:
                        neighbors.add(v)
                return neighbors

            # Pick node with max degree in induced subgraph
            best_node = None
            max_deg = -1
            for u in nodes:
                deg = len(get_neighbors(u, nodes))
                if deg > max_deg:
                    max_deg = deg
                    best_node = u
            
            S = {best_node}
            candidates = get_neighbors(best_node, nodes)
            
            while candidates:
                best_u = None
                max_deg_u = -1
                for u in candidates:
                    deg = len(get_neighbors(u, candidates))
                    if deg > max_deg_u:
                        max_deg_u = deg
                        best_u = u
                
                S.add(best_u)
                candidates = get_neighbors(best_u, candidates)
                
            return S

        while uncovered:
            if time.time() > deadline:
                # Fallback: just add remaining edges as pairs to ensure validity
                # This might be suboptimal but ensures we finish
                for e in list(uncovered):
                    cliques.append(sorted(e))
                    uncovered.discard(e)
                break

            best_clique = None
            best_size = 0
            
            # Try to find a clique involving an I vertex
            # Iterate I vertices
            for v in range(k, n):
                # Check if v has any uncovered edges
                # Neighbors of v in uncovered
                N_v = set()
                for u in range(k):
                    if frozenset((u, v)) in uncovered:
                        N_v.add(u)
                
                if not N_v:
                    continue
                
                # Find maximal clique in N_v using uncovered edges
                S = get_maximal_clique(N_v)
                if not S:
                    continue
                
                clique = {v} | S
                # Verify all edges in clique are in uncovered
                valid = True
                for u, w in combinations(sorted(clique), 2):
                    if frozenset((u, w)) not in uncovered:
                        valid = False
                        break
                
                if valid and len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
            
            # If no I vertex clique found, or we want to check C-C cliques too?
            # The loop above finds the *largest* clique involving *any* I vertex.
            # If best_clique is None, it means no I vertex has uncovered edges?
            # Or the maximal clique was empty?
            # If uncovered is not empty, there must be some edges.
            # If all edges are C-C, then N_v will be empty for all v.
            # In that case, best_clique remains None.
            
            if best_clique is None:
                # Must be C-C edges only
                # Pick any edge
                u, w = next(iter(uncovered))
                # Find maximal clique containing (u,w)
                # Nodes must be connected to both u and w
                candidates = set()
                for x in range(k):
                    if x != u and x != w:
                        if frozenset((u, x)) in uncovered and frozenset((w, x)) in uncovered:
                            candidates.add(x)
                
                # Add u and w to candidates? No, we start with {u, w}
                # We need a clique containing u and w.
                # So we look for nodes connected to all in {u, w} + previous picks.
                
                # Let's just use the helper on the intersection of neighbors of u and w
                # But the helper picks one node. We need to force u and w.
                
                # Simple approach:
                # Start S = {u, w}
                # Candidates = nodes connected to both u and w
                S = {u, w}
                candidates = set()
                for x in range(k):
                    if x not in S:
                        if frozenset((u, x)) in uncovered and frozenset((w, x)) in uncovered:
                            candidates.add(x)
                
                # Now expand
                while candidates:
                    # Pick best from candidates
                    best_x = None
                    max_deg_x = -1
                    for x in candidates:
                        # Degree in candidates
                        deg = 0
                        for y in candidates:
                            if x != y and frozenset((x, y)) in uncovered:
                                deg += 1
                        # Also check connection to S? 
                        # By definition of candidates, they are connected to u and w.
                        # But do they need to be connected to other nodes in S added later?
                        # Yes, the "candidates" set must be updated to only include nodes connected to the new node.
                        
                        if deg > max_deg_x:
                            max_deg_x = deg
                            best_x = x
                    
                    S.add(best_x)
                    new_candidates = set()
                    for x in candidates:
                        if x != best_x and frozenset((best_x, x)) in uncovered:
                            new_candidates.add(x)
                    candidates = new_candidates
                
                best_clique = S
                best_size = len(S)

            # Add clique to results
            cliques.append(sorted(best_clique))
            
            # Remove edges from uncovered
            for u, w in combinations(sorted(best_clique), 2):
                uncovered.discard(frozenset((u, w)))

        results.append([[int(x) for x in cl] for cl in cliques])

    return results
