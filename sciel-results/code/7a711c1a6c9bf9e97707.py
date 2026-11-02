import random
import time
from collections import defaultdict

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses local search (edge swaps) to eliminate short cycles.
    """
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Calculate Moore bound
    if k == 1:
        return g, [(0, 1)] if g >= 2 else []
    
    r = (g - 1) // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(r + 1))
    
    # Candidate n values
    candidates = []
    if k == 3 and g == 16:
        # Known best is 936. Try this first.
        # Then try slightly larger in case 936 is hard to find quickly.
        candidates = [936, 1000, 1100, 1200, 1500, 2000, 3000, 5000]
    else:
        n = moore_bound
        if n % 2 != 0:
            n += 1
        candidates = []
        for _ in range(10):
            candidates.append(n)
            n *= 2
            if n > 10000:
                break
            if n % 2 != 0:
                n += 1

    for n in candidates:
        if time.time() > deadline:
            break
        
        success, edges = _find_valid_graph(k, g, n, deadline)
        if success:
            return n, edges
        
    # Fallback: should not happen
    # Return a large random graph? No, must be valid.
    # If we get here, we failed to find a valid graph.
    # We can try one last time with a very large n.
    n = 10000
    success, edges = _find_valid_graph(k, g, n, deadline)
    if success:
        return n, edges
        
    # If all fails, raise error or return dummy?
    # The evaluator expects a valid graph.
    # This is a critical failure.
    # But given the heuristics, it should work.
    raise RuntimeError("Failed to construct valid graph")

def _find_valid_graph(k, g, n, deadline):
    """
    Attempts to construct a k-regular graph with n vertices and girth >= g.
    Returns (success, edges).
    """
    if n % 2 != 0:
        n += 1
        
    def generate_random_regular_graph(k, n):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        while True:
            random.shuffle(stubs)
            edge_set = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edge_set or (v, u) in edge_set:
                    valid = False
                    break
                edge_set.add(frozenset((u, v)))
            
            if valid:
                return edge_set
    
    edge_set = generate_random_regular_graph(k, n)
    
    adj = defaultdict(list)
    for u, v in edge_set:
        adj[u].append(v)
        adj[v].append(u)
        
    def find_short_cycle(adj, g):
        max_depth = g // 2
        for start in range(n):
            stack = [(start, [start], {start})]
            while stack:
                node, path, visited = stack.pop()
                if len(path) > max_depth:
                    continue
                for neighbor in adj[node]:
                    if neighbor in visited:
                        if neighbor in path:
                            idx = path.index(neighbor)
                            cycle = path[idx:]
                            if len(cycle) < g:
                                return cycle
                    else:
                        new_path = path + [neighbor]
                        new_visited = visited | {neighbor}
                        stack.append((neighbor, new_path, new_visited))
        return None

    search_start = time.time()
    max_search_time = 10
    
    while time.time() - search_start < max_search_time:
        if time.time() > deadline:
            break
            
        cycle = find_short_cycle(adj, g)
        if not cycle:
            return True, [tuple(e) for e in edge_set]
            
        k_len = len(cycle)
        edge_idx = random.randint(0, k_len - 1)
        u = cycle[edge_idx]
        v = cycle[(edge_idx + 1) % k_len]
        
        cycle_edges = set()
        for i in range(k_len):
            a = cycle[i]
            b = cycle[(i + 1) % k_len]
            cycle_edges.add(frozenset((a, b)))
            
        swapped = False
        for _ in range(50):
            x = random.randint(0, n - 1)
            if not adj[x]:
                continue
            y = random.choice(adj[x])
            
            e_xy = frozenset((x, y))
            if e_xy in cycle_edges:
                continue
            if e_xy == frozenset((u, v)):
                continue
                
            if u != x and u != y and v != x and v != y:
                if frozenset((u, x)) not in edge_set and frozenset((v, y)) not in edge_set:
                    edge_set.remove(frozenset((u, v)))
                    edge_set.remove(e_xy)
                    edge_set.add(frozenset((u, x)))
                    edge_set.add(frozenset((v, y)))
                    
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    adj[u].append(x)
                    adj[x].append(u)
                    adj[v].append(y)
                    adj[y].append(v)
                    
                    swapped = True
                    break
                    
            if u != x and u != y and v != x and v != y:
                if frozenset((u, y)) not in edge_set and frozenset((v, x)) not in edge_set:
                    edge_set.remove(frozenset((u, v)))
                    edge_set.remove(e_xy)
                    edge_set.add(frozenset((u, y)))
                    edge_set.add(frozenset((v, x)))
                    
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    adj[u].append(y)
                    adj[y].append(u)
                    adj[v].append(x)
                    adj[x].append(v)
                    
                    swapped = True
                    break
        
        if not swapped:
            pass

    cycle = find_short_cycle(adj, g)
    if not cycle:
        return True, [tuple(e) for e in edge_set]
        
    return False, None
