import time
import random
from collections import defaultdict

_DEADLINE_SECONDS = 145.0

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_longest_cycle(adj, deg, n, time_limit, rng):
    best_cycle = None
    best_len = 0
    
    candidates = [v for v in range(n) if deg[v] >= 2]
    if not candidates:
        return None
    
    candidates.sort(key=lambda v: -deg[v])
    
    start_time = time.time()
    
    # Perform several randomized DFS runs
    # Each run tries to find a long cycle
    max_restarts = 100
    # We will break early if we find a Hamiltonian cycle (len == n)
    
    for _ in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        if best_len == n:
            break
            
        start = rng.choice(candidates)
        
        # Iterative DFS with backtracking
        # Stack: (node, path, visited_set)
        # We use a list as a stack.
        # To allow backtracking and exploring other branches, we need to be careful.
        # Standard DFS explores one path to the end.
        # To find the *longest* cycle, we need to explore *all* simple paths? 
        # That is $O(N!)$ in worst case.
        # But with N=30 and random graphs, the search space is much smaller due to backtracking and pruning.
        
        # Let's use a recursive-like structure with explicit stack.
        # To avoid copying large lists, we can use a single path list and modify it.
        # But `visited` set needs to be updated.
        
        # Let's try a simple recursive DFS with a depth limit or time check.
        # Python recursion limit might be an issue, but N=30 is small.
        # However, recursion overhead is high.
        
        # Let's stick to iterative.
        # We will implement a DFS that backtracks.
        
        # State: (node, path, visited)
        # We push states onto the stack.
        # When we pop a state, we process it.
        # If we find a cycle, we update best.
        # We do NOT stop. We continue popping.
        # This explores the entire search tree (up to time limit).
        
        stack = [(start, [start], {start})]
        
        while stack:
            if time.time() - start_time > time_limit:
                return best_cycle
                
            node, path, visited = stack.pop()
            
            # Check for cycle closure
            # If node is start? No, we check neighbors.
            
            # Pruning: If max possible length is less than best_len, skip?
            # Max possible length = len(path) + (n - len(visited))
            if len(path) + (n - len(visited)) <= best_len:
                continue
                
            neighbors = list(adj[node])
            # Randomize neighbors to explore different paths
            rng.shuffle(neighbors)
            
            for nb in neighbors:
                if nb == start and len(path) >= 3:
                    # Found a cycle
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
                        if best_len == n:
                            return best_cycle
                elif nb not in visited:
                    # Extend path
                    new_path = path + [nb]
                    new_visited = visited | {nb}
                    # Push to stack
                    # To prioritize deeper paths, we could push in reverse order?
                    # DFS is LIFO.
                    stack.append((nb, new_path, new_visited))
                    
    return best_cycle

def _decompose_graph(edges, n, time_limit):
    if not edges:
        return []

    adj = defaultdict(set)
    deg = [0] * n
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    pieces = []
    start_time = time.time()
    rng = random.Random(42)
    
    while True:
        if time.time() - start_time > time_limit:
            break

        remaining = sum(len(adj[u]) for u in range(n)) // 2
        if remaining == 0:
            break

        # Find a long cycle
        # Allocate half the remaining time for this search step?
        # Or just a fixed small chunk?
        # If we spend too much time finding one cycle, we might not finish the graph.
        # Let's use a dynamic time limit: 10% of remaining time, min 0.1s.
        step_time = max(0.1, (time_limit - (time.time() - start_time)) * 0.1)
        
        cycle = _find_longest_cycle(adj, deg, n, step_time, rng)
        
        if cycle:
            pieces.append(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                adj[u].discard(v)
                adj[v].discard(u)
                deg[u] -= 1
                deg[v] -= 1
        else:
            # No cycle found, take a single edge
            edge_found = None
            for u in range(n):
                if adj[u]:
                    v = next(iter(adj[u]))
                    edge_found = (u, v)
                    break
            if edge_found is None:
                break
            u, v = edge_found
            pieces.append([u, v])
            adj[u].discard(v)
            adj[v].discard(u)
            deg[u] -= 1
            deg[v] -= 1

    # Add remaining edges
    for u in range(n):
        for v in adj[u]:
            if u < v:
                pieces.append([u, v])

    return pieces

def construct():
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for n, p, seed in specs:
        elapsed = time.time() - start_time
        remaining = _DEADLINE_SECONDS - elapsed
        if remaining < 5:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        
        graph_time = remaining * 0.95
        
        pieces = _decompose_graph(edges, n, graph_time)
        
        results.append(pieces)

    return results
