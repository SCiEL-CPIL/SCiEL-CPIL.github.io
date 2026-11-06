import time
import random
import itertools

def has_c5(graph):
    """Check if graph has a C5 subgraph. graph is set of edges (u,v) with u<v."""
    n = max(max(u, v) for u, v in graph) + 1 if graph else 0
    adj = [set() for _ in range(n)]
    for u, v in graph:
        adj[u].add(v)
        adj[v].add(u)
        
    # Check all 5-tuples
    # This is slow for n=13? 13P5 = 154440. Manageable.
    for perm in itertools.permutations(range(n), 5):
        if (perm[0], perm[1]) in graph or (perm[1], perm[0]) in graph:
            pass # Just check edges
        # Faster: check adj
        if (perm[1] in adj[perm[0]] and 
            perm[2] in adj[perm[1]] and 
            perm[3] in adj[perm[2]] and 
            perm[4] in adj[perm[3]] and 
            perm[0] in adj[perm[4]]):
            return True
    return False

def alpha_size(graph, n):
    """Find max independent set size. n is small (13)."""
    adj = [set() for _ in range(n)]
    for u, v in graph:
        adj[u].add(v)
        adj[v].add(u)
        
    # Simple backtracking or brute force for n=13
    # 2^13 = 8192 subsets.
    max_alpha = 0
    for mask in range(1 << n):
        is_independent = True
        for i in range(n):
            if mask & (1 << i):
                for j in range(i+1, n):
                    if mask & (1 << j):
                        if j in adj[i]:
                            is_independent = False
                            break
                if not is_independent:
                    break
        if is_independent:
            cnt = bin(mask).count('1')
            if cnt > max_alpha:
                max_alpha = cnt
    return max_alpha

def find_13_vertex_graph():
    start_time = time.time()
    deadline = start_time + 100 # Leave time for construction
    
    # Random search for C5-free graph with alpha <= 3
    # Start with random edges, try to add/remove to satisfy constraints
    
    # Better: Start with a known structure or random dense graph and prune C5s?
    # Or random sparse graph and add edges to reduce alpha?
    
    # Let's try random graphs with p=0.4
    for _ in range(1000):
        if time.time() > deadline:
            break
        edges = set()
        n = 13
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.4:
                    edges.add((u, v))
        
        # Check C5
        if has_c5(edges):
            continue
            
        # Check Alpha
        if alpha_size(edges, n) <= 3:
            return list(edges)
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find the 13-vertex component
    comp13 = find_13_vertex_graph()
    
    if comp13 is not None and time.time() < deadline - 5:
        # Construct 33-vertex graph
        n = 33
        edges = []
        
        # Add 13-vertex component (vertices 0-12)
        for u, v in comp13:
            edges.append([u, v])
            
        # Add 5 K4s (vertices 13-32)
        for b in range(5):
            base = 13 + 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
                    
        return {'n': n, 'edges': edges}
    else:
        # Fallback to 32-vertex graph
        n = 32
        edges = []
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return {'n': n, 'edges': edges}
