        def clique_partition_greedy(n, k, p, seed):
            rng = Random(seed)
            # Generate graph
            cross_edges = set()
            cross_adj = defaultdict(set)
            for w in range(k, n):
                for u in range(k):
                    if rng.random() < p:
                        cross_edges.add((u, w))
                        cross_adj[u].add(w)
            
            cc_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
            
            available_cc = set(cc_edges)
            available_cross = set(cross_edges)
            
            cliques = []
            
            # Greedy loop
            # We process w's. To be fair, we can shuffle or prioritize high-degree w.
            # Let's try a few iterations of the greedy pass.
            
            # Optimization: 
            # For a fixed w, we want a max matching in the graph of available_cc edges among N_K(w).
            # Since the graph is a subset of a complete graph, we can use a simple greedy matching 
            # or Hopcroft-Karp if we build the graph explicitly.
            # Given small k (<=30), simple greedy matching is fine.
            
            def get_matching_for_w(w, available_cc_set, cross_adj_dict):
                neighbors = cross_adj_dict.get(w, set())
                # Filter neighbors to those that are part of some available_cc edge?
                # Actually, we just need to pick pairs from neighbors that are in available_cc_set.
                # Build a list of available edges among neighbors
                edges = []
                neigh_list = list(neighbors)
                for i in range(len(neigh_list)):
                    for j in range(i+1, len(neigh_list)):
                        u, v = neigh_list[i], neigh_list[j]
                        if (u, v) in available_cc_set:
                            edges.append((u, v))
                
                # Find max matching in this graph
                # Vertices are in neigh_list
                # Use simple greedy: sort edges by... ? 
                # Or just random order and pick if disjoint.
                # To get max matching, we should use a proper algorithm.
                # But k is small. Let's use a simple DFS-based matching on the subgraph.
                
                # Build adj for matching
                match = {} # u -> v
                adj_match = defaultdict(list)
                for u, v in edges:
                    adj_match[u].append(v)
                    adj_match[v].append(u)
                
                # Kuhn's algorithm on this subgraph
                def dfs(u, visited):
                    for v in adj_match.get(u, []):
                        if v not in visited:
                            visited.add(v)
                            if v not in match or dfs(match[v], visited):
                                match[v] = u
                                match[u] = v
                                return True
                    return False
                
                # We need to match all possible vertices
                # The graph is small.
                # To ensure we get a MAXIMUM matching, we iterate over vertices.
                # But the graph changes as we pick edges? No, we pick a set of disjoint edges.
                # Kuhn's finds max matching in bipartite graph. This is general graph.
                # For general graph, Kuhn's doesn't work directly.
                # However, the graph is defined by available_cc edges.
                # Since available_cc is a subset of a clique, the graph is dense.
                # We can just use a greedy heuristic for the matching inside w:
                # Sort edges by... nothing, just pick randomly?
                # Or, since we want to maximize the number of edges, and all edges are equivalent 
                # in terms of weight (1), any maximal matching is good?
                # No, maximal != maximum.
                # But for a dense graph, greedy is close to max.
                
                # Let's just use a simple greedy: 
                # Pick edges one by one.
                # To be deterministic and fast, sort by (u,v).
                edges_sorted = sorted(edges)
                used_in_match = set()
                matching = []
                for u, v in edges_sorted:
                    if u not in used_in_match and v not in used_in_match:
                        matching.append((u, v))
                        used_in_match.add(u)
                        used_in_match.add(v)
                
                return matching
            
            # Main loop
            # We will do multiple passes to allow re-optimization?
            # Actually, the greedy "pick max matching for w" is destructive.
            # Once we pick edges for w, they are gone.
            # The order of w matters.
            # Let's try processing w in order of decreasing degree.
            
            w_order = sorted(range(k, n), key=lambda w: len(cross_adj.get(w, set())), reverse=True)
            
            for w in w_order:
                matching = get_matching_for_w(w, available_cc, cross_adj)
                for u, v in matching:
                    cliques.append([u, v, w])
                    available_cc.discard((u, v))
                    available_cross.discard((u, w))
                    available_cross.discard((v, w))
            
            # Cover remaining
            for u, v in available_cc:
                cliques.append([u, v])
            
            for u, w in available_cross:
                cliques.append([u, w])
                
            return cliques
        