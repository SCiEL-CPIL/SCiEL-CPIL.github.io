"""Construct a 4-regular graph of girth at least 10 with as few vertices as possible."""
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START

def _verify_girth(edges, n, g):
    """Verify that the graph has girth at least g using BFS from every vertex
       with non-backtracking constraint."""
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do a BFS tracking (current, previous) to avoid backtracking
    # We look for the shortest cycle through each vertex
    for start in range(n):
        # dist[(u, prev)] = distance from start
        # We want to find if there's a cycle of length < g
        # A cycle of length L through start means we can reach start from some neighbor
        # without backtracking in exactly L steps
        # BFS on state (node, prev_node)
        dist = {}
        queue = deque()
        for nb in adj[start]:
            dist[(nb, start)] = 1
            queue.append((nb, start))
        
        while queue:
            u, prev = queue.popleft()
            d = dist[(u, prev)]
            if d >= g - 1:
                break
            for w in adj[u]:
                if w == prev:
                    continue
                if w == start:
                    # Found a cycle of length d+1
                    if d + 1 < g:
                        return False
                    continue
                if (w, u) not in dist:
                    dist[(w, u)] = d + 1
                    queue.append((w, u))
    return True

def _is_valid(edges, n, k, g):
    """Check if the graph is k-regular, simple, and has girth >= g."""
    if n < 2:
        return False
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if u in adj[v] or v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Check girth
    if not _verify_girth(edges, n, g):
        return False
    return True

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    For k=4, g=10, we try to find a small such graph.
    
    Strategy: 
    1. Try known constructions if possible.
    2. Use randomized search: start with a k-regular graph, 
       then iteratively fix short cycles by edge swaps.
    """
    if k != 4 or g != 10:
        # Fallback to a simple construction for other parameters
        # Use the Lazebnik-Ustimenko-Woldar style construction
        return _lub_construct(k, g)
    
    # For (4, 10): try to find a 4-regular graph of girth >= 10
    # Moore bound for (4, 10): 
    # For even girth g, n >= 1 + k * (1 + (k-1) + ... + (k-1)^{(g/2-1)})
    # For g=10: n >= 1 + 4*(1 + 3 + 9 + 27 + 81) = 1 + 4*121 = 485? 
    # Wait, let me recalculate. The Moore bound for a k-regular graph of girth g:
    # If g is even: n >= 1 + k * sum_{i=0}^{g/2-1} (k-1)^i
    # For k=4, g=10: n >= 1 + 4 * (1 + 3 + 9 + 27 + 81) = 1 + 4*121 = 485
    # Hmm, but the task says the smallest known has 320 vertices and lower bound is 243.
    # Let me recheck the Moore bound formula.
    # Actually, the Moore bound for cage n(k,g) where g is the girth:
    # For even g = 2r: n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
    # For odd g = 2r+1: n >= k * sum_{i=0}^{r} (k-1)^i
    # For k=4, g=10 (even, r=5): n >= 1 + 4*(1+3+9+27+81) = 1+4*121 = 485
    # But the task says lower bound is 243 and best known is 320. 
    # This suggests the Moore bound calculation might be different or I'm misunderstanding.
    # Let me just proceed with finding any valid graph.
    
    # Try a randomized approach:
    # Start with a random 4-regular graph on a reasonable number of vertices
    # and try to increase its girth through edge swaps.
    
    # We'll try different sizes and use random restarts
    best_n = None
    best_edges = None
    
    # Try sizes from a lower estimate upward
    # The task says lower bound is 243, best known 320
    # Let's try to find something under 320 if possible
    
    random.seed(42)
    
    for n in range(244, 321):
        if n % 2 != 0 and k * n % 2 != 0:
            continue  # k*n must be even
        
        # Try multiple random initializations
        for attempt in range(50):
            edges = _random_k_regular(n, k)
            if edges is None:
                continue
            
            # Try to improve girth through edge swaps
            edges = _improve_girth(edges, n, k, g, max_iters=5000)
            
            if _is_valid(edges, n, k, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                    break
        
        if best_n is not None and best_n <= n:
            break
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback: use a construction that definitely works
    return _lub_construct(k, g)

def _random_k_regular(n, k):
    """Generate a random k-regular graph on n vertices."""
    if n * k % 2 != 0:
        return None
    if k >= n:
        return None
    
    # Use the configuration model / pairing model
    # Create a list of half-edges
    half_edges = []
    for v in range(n):
        for _ in range(k):
            half_edges.append(v)
    
    random.shuffle(half_edges)
    
    # Pair them up
    edges = set()
    valid = True
    for i in range(0, len(half_edges), 2):
        u, v = half_edges[i], half_edges[i+1]
        if u == v:
            valid = False
            break
        if (u, v) in edges or (v, u) in edges:
            valid = False
            break
        edges.add((min(u,v), max(u,v)))
    
    if not valid:
        return None
    
    return list(edges)

def _improve_girth(edges, n, k, g, max_iters=5000):
    """Try to increase girth by performing edge swaps."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    def find_short_cycle():
        """Find a cycle of length < g, or return None."""
        for start in range(n):
            # BFS with non-backtracking
            dist = {}
            parent = {}
            queue = deque()
            for nb in adj[start]:
                dist[(nb, start)] = 1
                parent[(nb, start)] = None
                queue.append((nb, start))
            
            while queue:
                u, prev = queue.popleft()
                d = dist[(u, prev)]
                if d >= g - 1:
                    break
                for w in adj[u]:
                    if w == prev:
                        continue
                    if w == start:
                        if d + 1 < g:
                            # Found a short cycle, reconstruct it
                            cycle = [start]
                            cur = (u, prev)
                            while cur is not None:
                                node, par = cur
                                cycle.append(node)
                                cur = parent.get((node, par))
                            cycle.reverse()
                            # Convert to edge list
                            cycle_edges = []
                            for i in range(len(cycle)):
                                a, b = cycle[i], cycle[(i+1) % len(cycle)]
                                cycle_edges.append((min(a,b), max(a,b)))
                            return cycle_edges
                        continue
                    if (w, u) not in dist:
                        dist[(w, u)] = d + 1
                        parent[(w, u)] = (u, prev)
                        queue.append((w, u))
        return None
    
    for it in range(max_iters):
        cycle = find_short_cycle()
        if cycle is None:
            break
        
        # Try to break the cycle by swapping an edge
        # Pick a random edge in the cycle and try to swap it
        random.shuffle(cycle)
        swapped = False
        for edge in cycle:
            u, v = edge
            # Try to replace (u,v) with (u,w) and (v,x) for some w,x
            # such that the new edges don't create new short cycles
            # Simple approach: try all possible partners
            candidates = []
            for w in adj[u]:
                if w == v:
                    continue
                for x in adj[v]:
                    if x == u:
                        continue
                    if (min(u,w), max(u,w)) in edge_set:
                        continue
                    if (min(v,x), max(v,x)) in edge_set:
                        continue
                    if w == x:
                        continue
                    candidates.append((w, x))
            
            if not candidates:
                continue
            
            random.shuffle(candidates)
            for w, x in candidates:
                # Try the swap
                new_edges = list(edge_set)
                new_edges.remove(edge)
                new_edges.append((min(u,w), max(u,w)))
                new_edges.append((min(v,x), max(v,x)))
                
                # Check if this improves things (heuristic: just try it)
                # For simplicity, accept if it doesn't immediately create a shorter cycle
                # This is expensive to check fully, so we'll just accept randomly
                # and rely on the outer loop to verify
                edge_set = set(new_edges)
                # Update adj
                adj[u].remove(v)
                adj[v].remove(u)
                adj[u].add(w)
                adj[v].add(x)
                adj[w].add(u)
                adj[x].add(v)
                swapped = True
                break
            
            if swapped:
                break
        
        if not swapped:
            # Stuck, try a random perturbation
            # Pick a random edge and swap it with a random non-edge
            edge = random.choice(list(edge_set))
            u, v = edge
            possible_w = [w for w in range(n) if w not in adj[u] and w != v]
            possible_x = [x for x in range(n) if x not in adj[v] and x != u]
            if possible_w and possible_x:
                w = random.choice(possible_w)
                x = random.choice(possible_x)
                if w != x:
                    edge_set.remove(edge)
                    edge_set.add((min(u,w), max(u,w)))
                    edge_set.add((min(v,x), max(v,x)))
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[u].add(w)
                    adj[v].add(x)
                    adj[w].add(u)
                    adj[x].add(v)
    
    return list(edge_set)

def _lub_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
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

# EVOLVE-BLOCK-END
