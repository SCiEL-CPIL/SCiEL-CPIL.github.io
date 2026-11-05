"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _biconnected_components(n, edges):
    """Find biconnected components using Tarjan's algorithm."""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    index_counter = [0]
    index = {}
    lowlink = {}
    root = None
    stack = []
    visited = [False] * n
    bccs = []
    
    def strongconnect(v):
        nonlocal index_counter, index, lowlink, root, stack, visited, bccs
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        visited[v] = True
        
        children = 0
        for w in adj[v]:
            if w not in index:
                children += 1
                stack.append((v, w))
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
                
                if lowlink[w] >= index[v]:
                    bcc = []
                    while True:
                        x, y = stack.pop()
                        bcc.append((x, y) if x < y else (y, x))
                        if (x, y) == (v, w) or (y, x) == (v, w):
                            break
                    bccs.append(bcc)
            elif w != v: # and not parent
                if visited[w] and index[w] < index[v]:
                    stack.append((v, w))
                    lowlink[v] = min(lowlink[v], index[w])
                    
    for v in range(n):
        if not visited[v]:
            root = v
            strongconnect(v)
            
    return bccs

def _find_cycle_in_graph(adj, n):
    """Find any cycle in the graph. Returns list of vertices or None."""
    visited = [False] * n
    parent = [-1] * n
    
    def dfs(u):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                parent[v] = u
                cycle = dfs(v)
                if cycle:
                    return cycle
            elif v != parent[u]:
                # Back edge found
                cycle = [v]
                cur = u
                while cur != v:
                    cycle.append(cur)
                    cur = parent[cur]
                cycle.append(v)
                return cycle
        return None
    
    for u in range(n):
        if not visited[u]:
            cycle = dfs(u)
            if cycle:
                return cycle
    return None

def _decompose_bcc(bcc_edges, n, deadline):
    """Decompose a biconnected component into cycles and single edges."""
    if not bcc_edges:
        return []
    
    adj = defaultdict(set)
    remaining = set(bcc_edges)
    for u, v in bcc_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    pieces = []
    
    # Greedily extract cycles
    while remaining:
        if time.time() > deadline:
            break
            
        # Find a cycle
        cycle = _find_cycle_in_graph(adj, n)
        if not cycle:
            break
            
        # Remove cycle edges
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i+1]
            e = (u, v) if u < v else (v, u)
            if e in remaining:
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
        # Note: The cycle found by DFS is a simple cycle, so the last edge (last, first) is also in the graph
        # But my _find_cycle_in_graph returns a list where the path is from v to u... 
        # Let's ensure the closing edge is removed too.
        # The cycle returned is [v, ..., u, v]? No, my dfs returns [v, ..., u, v] ? 
        # Let's check: cycle = [v], then append cur... until cur==v.
        # So it ends with v. The edges are (cycle[i], cycle[i+1]).
        # The last edge is (cycle[-2], cycle[-1]) which is (u, v).
        # The cycle in the graph is the path + the back edge.
        # The back edge is (u, v) where u is the current node and v is the ancestor.
        # In the list `cycle`, the vertices are v, ..., u, v.
        # So the edges are (v, next), ..., (prev, u), (u, v).
        # My loop `for i in range(len(cycle)-1)` handles (v, next)...(prev, u).
        # It misses (u, v) if the list is [v, ..., u, v].
        # Wait, if cycle is [1, 2, 3, 1], len=4.
        # i=0: (1,2)
        # i=1: (2,3)
        # i=2: (3,1)
        # This covers all edges.
        
        pieces.append(cycle)
        
    # Remaining edges are single edges
    for u, v in remaining:
        pieces.append([u, v])
        
    return pieces

def _improve_single_cycles(pieces, n, deadline):
    """Find cycles in the graph formed by single edges and replace them."""
    # Identify single edges
    single_edges = set()
    cycle_pieces = []
    
    for piece in pieces:
        if len(piece) == 2:
            u, v = piece
            single_edges.add((u, v) if u < v else (v, u))
        else:
            cycle_pieces.append(piece)
            
    if not single_edges:
        return pieces
        
    # Build adj for single edges
    adj = defaultdict(set)
    for u, v in single_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    improved_pieces = list(cycle_pieces)
    remaining_singles = set(single_edges)
    
    # Extract cycles from single edges
    while remaining_singles:
        if time.time() > deadline:
            break
            
        cycle = _find_cycle_in_graph(adj, n)
        if not cycle:
            break
            
        # Remove edges from remaining_singles
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i+1]
            e = (u, v) if u < v else (v, u)
            remaining_singles.discard(e)
            adj[u].discard(v)
            adj[v].discard(u)
            
        improved_pieces.append(cycle)
        
    # Add back remaining singles
    for u, v in remaining_singles:
        improved_pieces.append([u, v])
        
    return improved_pieces

def _decompose(edges, n):
    """Decompose the entire graph."""
    deadline = time.time() + 110 # Safety margin
    
    bccs = _biconnected_components(n, edges)
    
    all_pieces = []
    for bcc in bccs:
        bcc_pieces = _decompose_bcc(bcc, n, deadline)
        all_pieces.extend(bcc_pieces)
        
    # Global improvement: Cycle the singles
    all_pieces = _improve_single_cycles(all_pieces, n, deadline)
    
    return all_pieces

def construct():
    deadline = time.time() + 120
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n)
        results.append(pieces)
        
        if time.time() > deadline - 5:
            for n2, p2, seed2 in specs[len(results):]:
                edges2 = _instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
            
    return results
# EVOLVE-BLOCK-END
