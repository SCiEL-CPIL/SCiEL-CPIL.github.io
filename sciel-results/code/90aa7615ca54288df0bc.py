import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Calculate Moore Bound
    if g % 2 == 1:
        r = (g - 1) // 2
        lb = 1 + k * sum((k-1)**i for i in range(r))
    else:
        r = g // 2
        lb = 2 * sum((k-1)**i for i in range(r-1))
        
    # Ensure lb is at least 1
    lb = max(lb, 1)
    
    # Fallback: LUW graph
    def get_luw_graph(k, g):
        # Re-implementing the seed program's logic for fallback
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Try to find a better graph using random search
    # We iterate n from LB upwards
    # We give more time to larger n if small n fails quickly
    
    current_n = lb
    # Cap the search range to avoid infinite loops if LB is very small and we can't find anything
    # But we want to find the smallest n, so we must start at LB.
    
    # If LB is very small, random search might fail. 
    # We will try a range.
    
    while current_n < lb + 5000:
        # Check time
        if time.time() > deadline - 5.0:
            break
            
        # Determine time budget
        # Give small n less time, as they are less likely to succeed quickly
        if current_n < lb + 50:
            time_budget = 2.0
        elif current_n < lb + 200:
            time_budget = 5.0
        else:
            time_budget = 10.0
            
        if time.time() + time_budget > deadline:
            time_budget = max(0, deadline - time.time() - 1.0)
        if time_budget < 0.5:
            break
            
        result = try_random_search(k, g, current_n, time_budget)
        if result is not None:
            return result
            
        current_n += 1
        
    # Fallback to LUW
    return get_luw_graph(k, g)

def try_random_search(k, g, n, time_limit):
    start_time = time.time()
    
    # 1. Generate a random k-regular graph
    graph = None
    max_attempts = 100
    for _ in range(max_attempts):
        graph = generate_random_regular_graph(n, k)
        if graph is not None:
            break
            
    if graph is None:
        return None
        
    # 2. Check if it's already valid
    if check_girth(graph, n, g):
        return n, graph_to_edges(graph)
        
    # 3. Local Search
    # We maintain the graph as a list of sets of neighbors
    # graph[i] is a set of neighbors of i
    
    while time.time() - start_time < time_limit:
        # Find a short cycle
        cycle = find_short_cycle(graph, n, g)
        if cycle is None:
            # No short cycle found, graph is valid
            return n, graph_to_edges(graph)
            
        # Pick an edge in the cycle to break
        # cycle is a list of vertices [v0, v1, ..., vL-1]
        # Edges are (v0, v1), (v1, v2), ...
        edge_idx = random.randint(0, len(cycle) - 1)
        u = cycle[edge_idx]
        v = cycle[(edge_idx + 1) % len(cycle)]
        
        # Find a random edge (x, y) not in the cycle
        # We can just pick random vertices
        attempts = 0
        while attempts < 10:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            # Ensure x != y
            if x == y:
                attempts += 1
                continue
            # Ensure (x, y) is an edge
            if y not in graph[x]:
                attempts += 1
                continue
            # Ensure (x, y) is not in the cycle
            # Check if x and y are adjacent in cycle
            in_cycle = False
            for i in range(len(cycle)):
                a = cycle[i]
                b = cycle[(i + 1) % len(cycle)]
                if (a == x and b == y) or (a == y and b == x):
                    in_cycle = True
                    break
            if in_cycle:
                attempts += 1
                continue
                
            # Propose swap: remove (u, v) and (x, y), add (u, x) and (v, y)
            # Check validity
            if is_valid_swap(graph, u, v, x, y):
                # Check if new edges create short cycles
                if not creates_short_cycle(graph, u, x, g, n) and not creates_short_cycle(graph, v, y, g, n):
                    # Perform swap
                    perform_swap(graph, u, v, x, y)
                    break
            attempts += 1
        else:
            # If we failed to find a good swap, try a random swap anyway?
            # Or just break and restart?
            # For now, just continue to next iteration
            continue
            
    return None

def generate_random_regular_graph(n, k):
    if n * k % 2 != 0:
        return None
    if k >= n:
        return None
        
    # Create half-edges
    half_edges = []
    for i in range(n):
        for _ in range(k):
            half_edges.append(i)
            
    random.shuffle(half_edges)
    
    # Pair them
    graph = [set() for _ in range(n)]
    for i in range(0, len(half_edges), 2):
        u = half_edges[i]
        v = half_edges[i+1]
        if u == v:
            return None # Loop
        if v in graph[u]:
            return None # Multi-edge
        graph[u].add(v)
        graph[v].add(u)
        
    return graph

def check_girth(graph, n, g):
    # BFS from each node to depth (g-1)//2
    depth = (g - 1) // 2
    for start in range(n):
        visited = {start: 0}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            if dist >= depth:
                continue
            for neighbor in graph[node]:
                if neighbor == start and dist + 1 > 1:
                    # Cycle back to start
                    if dist + 1 < g:
                        return False
                elif neighbor in visited:
                    # Cycle found
                    if visited[neighbor] + dist + 1 < g:
                        return False
                else:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
    return True

def find_short_cycle(graph, n, g):
    # Return the cycle as a list of vertices
    depth = (g - 1) // 2
    for start in range(n):
        visited = {start: 0}
        parent = {start: -1}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            if dist >= depth:
                continue
            for neighbor in graph[node]:
                if neighbor == start and dist + 1 > 1:
                    # Found cycle back to start
                    if dist + 1 < g:
                        # Reconstruct cycle
                        cycle = []
                        cur = node
                        while cur != start:
                            cycle.append(cur)
                            cur = parent[cur]
                        cycle.append(start)
                        return cycle
                elif neighbor in visited:
                    if visited[neighbor] + dist + 1 < g:
                        # Found cycle
                        # Reconstruct
                        cycle = []
                        cur = node
                        while cur != neighbor:
                            cycle.append(cur)
                            cur = parent[cur]
                        cycle.append(neighbor)
                        return cycle
                else:
                    visited[neighbor] = dist + 1
                    parent[neighbor] = node
                    queue.append((neighbor, dist + 1))
    return None

def is_valid_swap(graph, u, v, x, y):
    # We are removing (u, v) and (x, y)
    # Adding (u, x) and (v, y)
    # Check loops
    if u == x or v == y:
        return False
    # Check multi-edges
    if x in graph[u] and x != v:
        return False
    if y in graph[v] and y != x:
        return False
    return True

def creates_short_cycle(graph, u, v, g, n):
    # Check if adding edge (u, v) creates a cycle of length < g
    # This is equivalent to checking if dist(u, v) < g-1 in G \ {(u, v)}
    # But we are in the process of adding it, so we check the current graph
    # Actually, we want to know if there is a path of length < g-1 between u and v
    # that does not use the edge (u, v) (which doesn't exist yet)
    # So just check dist(u, v) in current graph
    depth = g - 2
    visited = {u: 0}
    queue = deque([(u, 0)])
    while queue:
        node, dist = queue.popleft()
        if dist >= depth:
            continue
        for neighbor in graph[node]:
            if neighbor == v:
                if dist + 1 < g:
                    return True
            if neighbor not in visited:
                visited[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
    return False

def perform_swap(graph, u, v, x, y):
    graph[u].remove(v)
    graph[v].remove(u)
    graph[x].remove(y)
    graph[y].remove(x)
    
    graph[u].add(x)
    graph[x].add(u)
    graph[v].add(y)
    graph[y].add(v)

def graph_to_edges(graph):
    edges = []
    n = len(graph)
    for i in range(n):
        for j in graph[i]:
            if i < j:
                edges.append((i, j))
    return edges
