"""Construct a dense triangle-free, non-3-colorable graph on 16 vertices."""
import time
import random
from itertools import combinations

DEADLINE = 110.0  # safety margin before 120s wall-clock kill

def _is_triangle_free(adj):
    """Check if graph with adjacency sets is triangle-free."""
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[u]:
                if w <= v:
                    continue
                if w in adj[v]:
                    return False
    return True

def _is_not_3_colorable(adj):
    """Exact backtracking 3-colorability check. Returns True if NOT 3-colorable."""
    n = len(adj)
    color = [-1] * n
    
    # Order vertices by degree descending for better pruning
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    pos = {v: i for i, v in enumerate(order)}
    
    def backtrack(idx):
        if idx == n:
            return True  # successfully colored
        v = order[idx]
        used = set()
        for u in adj[v]:
            if color[u] != -1:
                used.add(color[u])
        for c in range(3):
            if c not in used:
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return not backtrack(0)

def _edge_count(adj):
    return sum(len(adj[i]) for i in range(len(adj))) // 2

def _random_triangle_free_graph(n=16, seed=None):
    """Generate a random triangle-free graph by random edge addition."""
    if seed is not None:
        random.seed(seed)
    adj = [set() for _ in range(n)]
    # Start with a random matching or empty
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.4:
                adj[u].add(v)
                adj[v].add(u)
    # Remove triangles
    changed = True
    while changed:
        changed = False
        for u in range(n):
            for v in list(adj[u]):
                if v <= u:
                    continue
                for w in list(adj[u]):
                    if w <= v:
                        continue
                    if w in adj[v]:
                        # Remove one edge to break triangle
                        if random.random() < 0.5:
                            adj[u].discard(v)
                            adj[v].discard(u)
                        else:
                            adj[u].discard(w)
                            adj[w].discard(u)
                        changed = True
                        break
    return adj

def _try_add_edge(adj, u, v):
    """Try to add edge (u,v), check triangle-free. Returns True if added."""
    if v in adj[u]:
        return True
    # Check no common neighbor
    for w in adj[u]:
        if w in adj[v]:
            return False
    adj[u].add(v)
    adj[v].add(u)
    return True

def _try_remove_edge(adj, u, v):
    """Remove edge (u,v)."""
    adj[u].discard(v)
    adj[v].discard(u)

def _local_search(adj, deadline, rng):
    """Local search: try to add edges while maintaining triangle-free and non-3-colorable."""
    n = len(adj)
    best_adj = [set(s) for s in adj]
    best_edges = _edge_count(best_adj)
    
    # Try all pairs to add
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    while time.time() < deadline:
        rng.shuffle(all_pairs)
        improved = False
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if v in adj[u]:
                continue
            # Check triangle-free
            can_add = True
            for w in adj[u]:
                if w in adj[v]:
                    can_add = False
                    break
            if not can_add:
                continue
            # Try adding
            adj[u].add(v)
            adj[v].add(u)
            if _is_not_3_colorable(adj):
                best_adj = [set(s) for s in adj]
                best_edges = _edge_count(adj)
                improved = True
            else:
                adj[u].discard(v)
                adj[v].discard(u)
        if not improved:
            break
    
    return best_adj, best_edges

def _hill_climbing_from(adj, deadline, rng):
    """Hill climbing: start from given adj, try to add edges greedily."""
    n = len(adj)
    best_adj = [set(s) for s in adj]
    best_edges = _edge_count(best_adj)
    
    while time.time() < deadline:
        improved = False
        # Try adding each missing edge
        for u in range(n):
            if time.time() > deadline:
                break
            for v in range(u + 1, n):
                if v in adj[u]:
                    continue
                # Check triangle-free
                can_add = True
                for w in adj[u]:
                    if w in adj[v]:
                        can_add = False
                        break
                if not can_add:
                    continue
                # Try adding
                adj[u].add(v)
                adj[v].add(u)
                if _is_not_3_colorable(adj):
                    best_adj = [set(s) for s in adj]
                    best_edges = _edge_count(adj)
                    improved = True
                else:
                    adj[u].discard(v)
                    adj[v].discard(u)
        if not improved:
            break
    
    return best_adj, best_edges

def _mycielski_of_c5():
    """Mycielski graph of C5 (Grötzsch graph) on vertices 0..10."""
    adj = [set() for _ in range(11)]
    # C5: 0-1-2-3-4-0
    for i in range(5):
        adj[i].add((i + 1) % 5)
        adj[(i + 1) % 5].add(i)
    # Mycielski: add vertices 5..9, each 5+i adjacent to neighbors of i
    for i in range(5):
        for j in [(i - 1) % 5, (i + 1) % 5]:
            adj[5 + i].add(j)
            adj[j].add(5 + i)
    # Vertex 10 adjacent to all of 5..9
    for i in range(5):
        adj[10].add(5 + i)
        adj[5 + i].add(10)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(42)
    
    best_adj = None
    best_edges = 0
    
    # Start with Mycielski of C5 (Grötzsch graph) + 5 isolated vertices
    adj_base = _mycielski_of_c5()
    n = 16
    adj = [set(s) for s in adj_base] + [set() for _ in range(5)]
    
    # First, try to connect isolated vertices to existing structure
    # Each new vertex v can connect to an independent set in existing graph
    for new_v in range(11, 16):
        # Find maximum independent set in current graph that new_v can connect to
        # Actually, new_v can connect to a set S where S is independent in G
        # and no vertex in S is adjacent to any other vertex in S (trivially true for independent set)
        # The constraint is: if new_v connects to S, then for any edge (u,w) in G, not both u,w in S
        # which is exactly that S is independent.
        # To maximize edges, connect new_v to a maximum independent set.
        
        # Simple heuristic: try to add edges one by one
        for u in range(new_v):
            if time.time() > deadline:
                break
            # Check if adding (new_v, u) keeps triangle-free
            can_add = True
            for w in adj[new_v]:
                if w in adj[u]:
                    can_add = False
                    break
            if can_add:
                adj[new_v].add(u)
                adj[u].add(new_v)
    
    # Now do local search to add more edges
    best_adj, best_edges = _hill_climbing_from(adj, deadline, rng)
    
    # Try multiple random restarts
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        # Generate a random triangle-free graph
        adj_rand = _random_triangle_free_graph(16, seed=rng.randint(0, 10**6))
        
        # Check if non-3-colorable
        if not _is_not_3_colorable(adj_rand):
            continue
        
        # Try to improve
        adj_improved, edges_improved = _hill_climbing_from(adj_rand, deadline - 2, rng)
        
        if edges_improved > best_edges:
            best_edges = edges_improved
            best_adj = adj_improved
    
    # Also try starting from Grötzsch + various connections
    if time.time() < deadline - 5:
        for trial in range(20):
            if time.time() > deadline - 2:
                break
            adj_try = _mycielski_of_c5()
            adj_try = [set(s) for s in adj_try] + [set() for _ in range(5)]
            # Randomly connect new vertices
            for new_v in range(11, 16):
                candidates = list(range(new_v))
                rng.shuffle(candidates)
                for u in candidates:
                    if time.time() > deadline - 2:
                        break
                    can_add = True
                    for w in adj_try[new_v]:
                        if w in adj[u]:
                            can_add = False
                            break
                    if can_add:
                        adj_try[new_v].add(u)
                        adj_try[u].add(new_v)
            
            if _is_not_3_colorable(adj_try):
                adj_imp, ed = _hill_climbing_from(adj_try, deadline - 2, rng)
                if ed > best_edges:
                    best_edges = ed
                    best_adj = adj_imp
    
    # Final result
    if best_adj is None:
        # Fallback: return Grötzsch graph with isolated vertices
        adj = _mycielski_of_c5()
        adj = [set(s) for s in adj] + [set() for _ in range(5)]
        best_adj = adj
    
    # Convert to edge list
    edges = []
    for u in range(16):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    
    return edges
