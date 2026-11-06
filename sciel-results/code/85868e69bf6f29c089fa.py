import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for 120s limit
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(adj_list, n, remaining_edges, time_limit):
        best_len = 0
        best_cycle = []
        
        starts = list(range(n))
        random.shuffle(starts)
        
        for start in starts:
            if time.time() > time_limit:
                break
            if not adj_list[start]:
                continue
            
            visited = {start: 0}
            parent = {start: None}
            in_stack = {start}
            
            def dfs(u):
                nonlocal best_len, best_cycle
                if best_len == n:
                    return
                for v in adj_list[u]:
                    if (min(u, v), max(u, v)) not in remaining_edges:
                        continue
                    if v not in visited:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        in_stack.add(v)
                        dfs(v)
                        in_stack.remove(v)
                    elif v in in_stack:
                        length = visited[u] - visited[v] + 1
                        if length >= 3 and length > best_len:
                            # Reconstruct
                            cycle = [u]
                            cur = u
                            while cur != v:
                                cur = parent[cur]
                                cycle.append(cur)
                            best_cycle = cycle
                            best_len = length
                            if best_len == n:
                                return

            dfs(start)
            
        return best_cycle

    def decompose_once(edges, n, time_limit):
        adj_list = {i: [] for i in range(n)}
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        for i in adj_list:
            random.shuffle(adj_list[i])
            
        pieces = []
        remaining_edges = set(edges)
        
        while remaining_edges:
            if time.time() > time_limit:
                for e in remaining_edges:
                    pieces.append(list(e))
                return pieces
                
            best_cycle = find_longest_cycle(adj_list, n, remaining_edges, time_limit)
            if best_cycle:
                pieces.append(best_cycle)
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    e = (min(u, v), max(u, v))
                    if e in remaining_edges:
                        remaining_edges.discard(e)
                        adj_list[u].remove(v)
                        adj_list[v].remove(u)
            else:
                for e in remaining_edges:
                    pieces.append(list(e))
                break

        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        edges = instance(*spec)
        n = spec[0]
        
        best_pieces = None
        best_score = float('inf')
        
        # Allocate time for this graph
        # Total deadline is 130s.
        # We have 3 graphs.
        # Let's reserve 10s for the last graph to be safe?
        # Or just split remaining time equally?
        # Simpler: set a sub-deadline for each graph.
        # If we finish early, we can use the leftover for the next?
        # No, `deadline` is global.
        # Let's just use a fraction of remaining time.
        
        remaining_time = deadline - time.time()
        # Spend at most 30s per graph, or remaining time if less.
        graph_time_limit = min(30.0, remaining_time)
        graph_deadline = time.time() + graph_time_limit
        
        # Ensure we have at least some time
        if graph_deadline > deadline:
            graph_deadline = deadline - 1.0

        # Run restarts
        while time.time() < graph_deadline:
            pieces = decompose_once(edges, n, graph_deadline)
            score = len(pieces)
            if best_pieces is None or score < best_score:
                best_pieces = pieces
                best_score = score
                
            # Early stop if we hit a good lower bound
            # Min pieces >= E/3
            if best_score <= len(edges) / 3.0 + 1:
                break

        if best_pieces is None:
            # Fallback if no time or error
            best_pieces = [list(e) for e in edges]
            
        results.append(best_pieces)
        
    return results
