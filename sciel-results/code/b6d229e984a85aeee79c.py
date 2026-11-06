    def solve(n, p, seed):
        edges = generate_edges(n, p, seed)
        if not edges:
            return []

        # adjacency as sets
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # track uncovered edges
        uncovered = set(edges)
        result = []

        def get_edge(u, v):
            return (u, v) if u < v else (v, u)

        while uncovered:
            # Find candidate vertices incident to uncovered edges
            cand_vertices = set()
            for u, v in uncovered:
                cand_vertices.add(u)
                cand_vertices.add(v)

            # Calculate degree in uncovered subgraph for sorting seeds
            deg_uncov = {v: 0 for v in cand_vertices}
            for u, v in uncovered:
                deg_uncov[u] += 1
                deg_uncov[v] += 1

            # Sort seeds by degree descending
            seeds = sorted(cand_vertices, key=lambda x: -deg_uncov[x])
            
            best_clique = None
            best_covered_edges = set()
            max_gain = 0

            # Try building a clique from top seeds
            # Limit number of seeds to try to ensure speed
            num_seeds_to_try = min(len(seeds), 15)
            
            for start_v in seeds[:num_seeds_to_try]:
                current_clique = {start_v}
                current_covered = set()
                
                # Potential vertices: neighbors of start_v
                potential = set(adj[start_v])
                
                while True:
                    best_next = None
                    best_gain = 0
                    best_new_edges = set()
                    
                    # Iterate through potential vertices
                    for v in potential:
                        if v in current_clique:
                            continue
                        
                        # Count uncovered edges from v to current_clique
                        gain = 0
                        new_edges = []
                        for u in current_clique:
                            e = get_edge(v, u)
                            if e in uncovered:
                                gain += 1
                                new_edges.append(e)
                                
                        if gain > best_gain:
                            best_gain = gain
                            best_next = v
                            best_new_edges = new_edges
                            
                    if best_next is None:
                        break
                    
                    current_clique.add(best_next)
                    current_covered.update(best_new_edges)
                    
                    # Update potential: intersect with neighbors of best_next
                    potential = potential & adj[best_next]
                    
                    # If potential is empty, no more vertices can be added
                    if not potential:
                        break

                # Evaluate result from this seed
                if len(current_clique) >= 2:
                    if len(current_covered) > max_gain:
                        max_gain = len(current_covered)
                        best_clique = current_clique
                        best_covered_edges = current_covered
                        
            if best_clique:
                result.append(sorted(best_clique))
                for e in best_covered_edges:
                    uncovered.discard(e)
            else:
                # Fallback: pick a single edge
                e = next(iter(uncovered))
                result.append(sorted([e[0], e[1]]))
                uncovered.discard(e)

        return result
