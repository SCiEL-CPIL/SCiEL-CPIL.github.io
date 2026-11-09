"""Decompose three fixed random graphs into few cycles and edges (Erdős-Gallai style)."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict

def construct():
    t_start = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, n):
        if not edges:
            return []

        remaining = set(edges)
        pieces = []

        def build_adj(rem):
            adj = defaultdict(list)
            for u, v in rem:
                adj[u].append(v)
                adj[v].append(u)
            return adj

        def find_cycle_bfs(start, adj, rem, max_depth=8, deadline=None):
            """BFS to find a cycle through start with up to max_depth vertices."""
            # BFS from start, track parent to reconstruct path
            # We want to find a path from some neighbor of start back to start
            visited = {start: None}  # node -> (parent, edge)
            queue = [start]
            found = False
            end_node = None
            
            while queue and not found:
                if deadline and time.time() > deadline:
                    return None
                node = queue.pop(0)
                if len([x for x in queue]) > 2000:
                    break
                for nb in adj[node]:
                    e = tuple(sorted((node, nb)))
                    if e not in rem:
                        continue
                    if nb == start:
                        # Found a cycle: reconstruct path from node back to start
                        path = [node]
                        cur = node
                        while cur != start:
                            cur = visited[cur][0]
                            path.append(cur)
                        path.reverse()
                        return path
                    if nb not in visited:
                        visited[nb] = (node, e)
                        queue.append(nb)
            return None

        def find_long_cycle_backtrack(start, adj, rem, target_len, deadline=None, max_nodes=50000):
            """Backtracking to find a cycle of exactly target_len through start."""
            count = [0]
            result = [None]
            
            def bt(path, visited):
                if result[0] is not None:
                    return
                count[0] += 1
                if count[0] > max_nodes:
                    return
                if deadline and time.time() > deadline:
                    return
                if len(path) == target_len:
                    last = path[-1]
                    if start in adj[last]:
                        e = tuple(sorted((last, start)))
                        if e in rem:
                            result[0] = list(path)
                            return
                    return
                node = path[-1]
                for nb in adj[node]:
                    if nb == start and len(path) >= 3:
                        # Can close the cycle
                        e = tuple(sorted((node, start)))
                        if e in rem:
                            result[0] = list(path)
                            return
                    if nb not in visited and len(path) < target_len:
                        visited.add(nb)
                        path.append(nb)
                        bt(path, visited)
                        if result[0] is not None:
                            return
                        path.pop()
                        visited.discard(nb)
            
            bt([start], {start})
            return result[0]

        def greedy_decompose(rem, n, per_graph_deadline):
            """Main greedy loop: repeatedly extract cycles from rem."""
            local_pieces = []
            
            while rem:
                if time.time() > per_graph_deadline:
                    break
                
                adj = build_adj(rem)
                
                # Compute degrees
                deg = defaultdict(int)
                for u, v in rem:
                    deg[u] += 1
                    deg[v] += 1
                
                if not deg:
                    break
                
                # Pick start vertex: highest degree
                start = max(deg, key=lambda x: deg[x])
                
                if deg[start] < 2:
                    # No cycle possible through this vertex
                    # Just take an edge
                    edge = None
                    for u, v in rem:
                        if u == start or v == start:
                            edge = (u, v)
                            break
                    if edge is None:
                        edge = next(iter(rem))
                    local_pieces.append(list(edge))
                    rem.discard(edge)
                    continue
                
                # Try to find the longest cycle through start
                cycle = None
                
                # Strategy: try longer cycles first with backtracking, then fall back to BFS
                # Limit backtracking depth to keep it fast
                max_bt_len = min(n, 12)
                
                for target_len in range(max_bt_len, 2, -1):
                    if time.time() > per_graph_deadline:
                        break
                    cycle = find_long_cycle_backtrack(
                        start, adj, rem, target_len, 
                        deadline=per_graph_deadline, max_nodes=30000
                    )
                    if cycle:
                        break
                
                if not cycle:
                    # Fall back to BFS for any cycle
                    cycle = find_cycle_bfs(start, adj, rem, max_depth=min(n, 10), deadline=per_graph_deadline)
                
                if cycle and len(cycle) >= 3:
                    # Validate all edges exist in rem
                    valid = True
                    cycle_edges = set()
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((u, v)))
                        if e not in rem:
                            valid = False
                            break
                        cycle_edges.add(e)
                    
                    if valid:
                        local_pieces.append(cycle)
                        rem -= cycle_edges
                    else:
                        # Invalid cycle, try BFS fallback
                        cycle2 = find_cycle_bfs(start, adj, rem, max_depth=min(n, 8), deadline=per_graph_deadline)
                        if cycle2 and len(cycle2) >= 3:
                            valid2 = True
                            cycle_edges2 = set()
                            for i in range(len(cycle2)):
                                u, v = cycle2[i], cycle2[(i + 1) % len(cycle2)]
                                e = tuple(sorted((u, v)))
                                if e not in rem:
                                    valid2 = False
                                    break
                                cycle_edges2.add(e)
                            if valid2:
                                local_pieces.append(cycle2)
                                rem -= cycle_edges2
                            else:
                                # Just take an edge
                                edge = None
                                for u, v in rem:
                                    if u == start or v == start:
                                        edge = (u, v)
                                        break
                                if edge is None:
                                    edge = next(iter(rem))
                                local_pieces.append(list(edge))
                                rem.discard(edge)
                        else:
                            # No cycle found, take an edge
                            edge = None
                            for u, v in rem:
                                if u == start or v == start:
                                    edge = (u, v)
                                    break
                            if edge is None:
                                edge = next(iter(rem))
                            local_pieces.append(list(edge))
                            rem.discard(edge)
                else:
                    # No cycle found, take an edge
                    edge = None
                    for u, v in rem:
                        if u == start or v == start:
                            edge = (u, v)
                            break
                    if edge is None:
                        edge = next(iter(rem))
                    local_pieces.append(list(edge))
                    rem.discard(edge)
            
            # Add remaining edges as single pieces
            for e in rem:
                local_pieces.append(list(e))
            
            return local_pieces

        return greedy_decompose(remaining, n, DEADLINE - 2.0)

    results = []
    for i, spec in enumerate(((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))):
        if time.time() > DEADLINE - 5:
            # Not enough time, just return edges
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        n = spec[0]
        # Allocate time budget per graph
        remaining_time = DEADLINE - time.time() - 2.0
        per_graph_budget = remaining_time / (3 - i)
        if per_graph_budget < 1.0:
            per_graph_budget = 1.0
        graph_deadline = time.time() + per_graph_budget
        pieces = decompose(edges, n)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
