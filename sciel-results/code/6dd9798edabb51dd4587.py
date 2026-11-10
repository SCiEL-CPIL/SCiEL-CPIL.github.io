"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_best_decomposition(edges, n):
        """Find decomposition with minimum pieces using path-based cycle merging."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not edge_set:
            return []
        
        best_pieces = None
        best_count = len(edge_set) + 1
        
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            
            remaining = set(edge_set)
            pieces = []
            
            # Build adjacency for remaining edges
            adj = defaultdict(set)
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)
            
            # Phase 1: Find maximum set of edge-disjoint paths via greedy matching on line graph
            # We'll find paths that cover edges, then try to merge them into cycles
            
            # Strategy: Repeatedly find long paths in the 2-core, then check if endpoints can be connected
            # to form a cycle. If yes, make it a cycle. If no, keep as path (but we need single edges or cycles).
            # Actually, the problem requires pieces to be either single edges or cycles.
            # So we need to extract cycles, not paths.
            
            # Revised strategy: 
            # 1. Find the 2-core
            # 2. In the 2-core, repeatedly find cycles using a smarter method
            # 3. For remaining edges, try to find cycles or leave as single edges
            
            # Get 2-core
            def get_2_core(edge_list):
                if not edge_list:
                    return set(), set()
                adj_local = defaultdict(set)
                for u, v in edge_list:
                    adj_local[u].add(v)
                    adj_local[v].add(u)
                
                changed = True
                while changed:
                    changed = False
                    to_remove = set()
                    for u in list(adj_local.keys()):
                        if len(adj_local[u]) < 2:
                            to_remove.add(u)
                    if to_remove:
                        changed = True
                        for u in to_remove:
                            for v in adj_local[u]:
                                adj_local[v].discard(u)
                            del adj_local[u]
                
                core_edges = set()
                for u in adj_local:
                    for v in adj_local[u]:
                        if u < v:
                            core_edges.add((u, v))
                return core_edges, adj_local
            
            core_edges, core_adj = get_2_core(remaining)
            remaining_core = set(core_edges)
            
            # Extract cycles from 2-core using improved method
            while remaining_core:
                if time.time() > deadline - 15:
                    break
                
                # Find a cycle in the 2-core
                # Use a method that tries to find longer cycles by exploring multiple options
                cycle = None
                
                # Try starting from different vertices
                candidates = [u for u in core_adj if len([nb for nb in core_adj[u] if (min(u, nb), max(u, nb)) in remaining_core]) >= 2]
                
                if candidates:
                    # Sort by degree in remaining_core
                    candidates.sort(key=lambda u: len([nb for nb in core_adj[u] if (min(u, nb), max(u, nb)) in remaining_core]), reverse=True)
                    
                    for start in candidates[:5]:
                        # Try to find a cycle starting at start
                        for attempt in range(20):
                            path = [start]
                            visited = {start}
                            current = start
                            found = False
                            
                            for _ in range(50):
                                neighbors = [nb for nb in core_adj[current] if (min(current, nb), max(current, nb)) in remaining_core]
                                if not neighbors:
                                    break
                                
                                unvisited = [nb for nb in neighbors if nb not in visited]
                                
                                if unvisited:
                                    # Prefer neighbors that can lead back to start
                                    unvisited.sort(key=lambda x: len(core_adj[x]), reverse=True)
                                    next_node = random.choice(unvisited)
                                elif start in neighbors:
                                    next_node = start
                                    found = True
                                else:
                                    break
                                
                                if found:
                                    if len(path) >= 3:
                                        if cycle is None or len(path) > len(cycle):
                                            cycle = path[:]
                                    break
                                
                                path.append(next_node)
                                visited.add(next_node)
                                current = next_node
                            
                            if found and len(path) >= 3:
                                if cycle is None or len(path) > len(cycle):
                                    cycle = path[:]
                        if cycle:
                            break
                
                if cycle and len(cycle) >= 3:
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((u, v)))
                        remaining_core.discard(e)
                        remaining.discard(e)
                    pieces.append(cycle)
                else:
                    break
            
            # Phase 2: Handle remaining edges
            while remaining:
                if time.time() > deadline - 15:
                    break
                
                # Try to find a cycle in remaining edges
                found_cycle = False
                edges_list = list(remaining)
                random.shuffle(edges_list)
                
                for e in edges_list[:10]:
                    u, v = e
                    # BFS from v to u avoiding edge (u,v)
                    parent = {v: None}
                    queue = deque([v])
                    
                    while queue:
                        current = queue.popleft()
                        if current == u:
                            # Found a path from v to u
                            path = []
                            node = u
                            while node != v:
                                path.append(node)
                                node = parent[node]
                            path.reverse()
                            cycle = [u] + path
                            if len(cycle) >= 3:
                                for i in range(len(cycle)):
                                    cu, cv = cycle[i], cycle[(i + 1) % len(cycle)]
                                    ce = tuple(sorted((cu, cv)))
                                    remaining.discard(ce)
                                pieces.append(cycle)
                                found_cycle = True
                                break
                            break
                        for nb in adj[current]:
                            if nb not in parent:
                                ne = tuple(sorted((current, nb)))
                                if ne == e:
                                    continue
                                if ne in remaining:
                                    parent[nb] = current
                                    queue.append(nb)
                    
                    if found_cycle:
                        break
                
                if not found_cycle:
                    # Take a single edge
                    e = tuple(sorted(random.choice(list(remaining))))
                    remaining.discard(e)
                    pieces.append(list(e))
            
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = [p[:] for p in pieces]
        
        return best_pieces if best_pieces else [list(e) for e in edge_set]
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = find_best_decomposition(edges, n)
        results.append(pieces)
        if time.time() > deadline - 5:
            for spec2 in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))[len(results):]:
                n2, p2, seed2 = spec2
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
# EVOLVE-BLOCK-END
