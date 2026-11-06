    def construct():
        import random
        from collections import defaultdict

        def instance(n, p, seed):
            rng = random.Random(seed)
            order = list(range(n))
            rng.shuffle(order)
            edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
            edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
            return sorted(edges)

        def decompose(edges):
            if not edges:
                return []
            
            # Build adjacency list
            adj = defaultdict(list)
            deg = defaultdict(int)
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
            
            # Find connected components
            visited = set()
            components = []
            for node in range(max(adj.keys()) + 1 if adj else 0):
                if node not in visited and deg[node] > 0:
                    # BFS/DFS to find component
                    stack = [node]
                    comp_nodes = set()
                    while stack:
                        n = stack.pop()
                        if n in visited:
                            continue
                        visited.add(n)
                        comp_nodes.add(n)
                        for nei in adj[n]:
                            if nei not in visited:
                                stack.append(nei)
                    components.append(comp_nodes)
            
            all_pieces = []
            
            for comp in components:
                # Get edges within this component
                comp_edges = [e for e in edges if e[0] in comp and e[1] in comp]
                if not comp_edges:
                    continue
                
                # Check degrees within component
                odd_nodes = [n for n in comp if deg[n] % 2 != 0]
                
                virtual_edges = set()
                # Pair odd nodes
                for i in range(0, len(odd_nodes), 2):
                    u, v = odd_nodes[i], odd_nodes[i+1]
                    virtual_edges.add(tuple(sorted((u, v))))
                    # Add to adj for Eulerian tour
                    adj[u].append(v)
                    adj[v].append(u)
                
                # Find Eulerian circuit for this component
                # Start node: any node in comp with degree > 0
                start = next(iter(comp))
                
                # Hierholzer's algorithm
                # We need to pop edges.
                # To ensure determinism, we should sort adjacency lists?
                # The prompt doesn't require deterministic output, just valid and minimal.
                # But for debugging, it's nice.
                # However, popping from list is O(1).
                
                # We need a copy of adj to not modify the original?
                # We are modifying adj in place.
                # Since we process components separately, we can just use the global adj.
                # But we added virtual edges.
                
                stack = [start]
                circuit = []
                
                while stack:
                    v = stack[-1]
                    if adj[v]:
                        u = adj[v].pop()
                        stack.append(u)
                    else:
                        circuit.append(stack.pop())
                
                circuit.reverse()
                
                # Extract cycles from circuit
                # circuit is a list of vertices
                # Decompose into simple cycles
                def split_walk(walk):
                    if len(walk) < 2:
                        return []
                    
                    # Check if simple cycle
                    # walk[0] == walk[-1]
                    # Inner vertices distinct
                    inner = walk[1:-1]
                    if len(set(inner)) == len(inner) and walk[0] not in inner:
                        return [walk[:-1]]
                    
                    seen = {}
                    for idx, v in enumerate(walk):
                        if v in seen:
                            start_idx = seen[v]
                            cycle = walk[start_idx : idx+1]
                            new_walk = walk[:start_idx] + walk[idx:]
                            sub_cycles = split_walk(new_walk)
                            return [cycle] + sub_cycles
                        seen[v] = idx
                    return []
                
                raw_cycles = split_walk(circuit)
                
                for cyc in raw_cycles:
                    # Check for virtual edges
                    has_virtual = False
                    for i in range(len(cyc)):
                        u = cyc[i]
                        v = cyc[(i+1) % len(cyc)]
                        e = tuple(sorted((u, v)))
                        if e in virtual_edges:
                            has_virtual = True
                            break
                    
                    if not has_virtual:
                        # Valid cycle
                        all_pieces.append(cyc)
                    else:
                        # Break into paths
                        # We need to traverse the cycle and split at virtual edges
                        # A cycle is a list of vertices.
                        # We can iterate through edges.
                        segments = []
                        current_segment = []
                        
                        # To make it a list of edges, we can just iterate
                        # and group consecutive real edges.
                        
                        # Let's convert cycle to list of edges
                        cycle_edges = []
                        for i in range(len(cyc)):
                            u = cyc[i]
                            v = cyc[(i+1) % len(cyc)]
                            cycle_edges.append((u, v))
                        
                        # Now split cycle_edges into segments of real edges
                        # A segment is a list of edges that are not virtual.
                        # Since virtual edges break the cycle, the segments are paths.
                        
                        current_path = []
                        for u, v in cycle_edges:
                            e = tuple(sorted((u, v)))
                            if e in virtual_edges:
                                if current_path:
                                    segments.append(current_path)
                                    current_path = []
                            else:
                                current_path.append((u, v))
                        
                        if current_path:
                            segments.append(current_path)
                        
                        # Each segment is a path (list of edges).
                        # Decompose each path into single edges.
                        for path in segments:
                            for u, v in path:
                                all_pieces.append([u, v])
            
            return all_pieces

        results = []
        for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
            edges = instance(*spec)
            pieces = decompose(edges)
            results.append(pieces)
        
        return results
    