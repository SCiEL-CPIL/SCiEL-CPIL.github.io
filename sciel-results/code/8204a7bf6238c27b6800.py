"""K2,3-free graph on 20 vertices: maximize edges via greedy + enhanced local search."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n = 20
    best_edges = []
    best_score = 0
    
    # Strategy 1: Multiple greedy restarts
    for seed in range(300):
        if time.time() > deadline - 30.0:
            break
        
        edges = _greedy_construct(n, seed)
        score = len(edges)
        
        if score > best_score:
            best_score = score
            best_edges = edges[:]
    
    # Strategy 2: Enhanced local search from best greedy result
    if time.time() < deadline - 20.0 and best_edges:
        improved = _enhanced_local_search(n, best_edges, deadline)
        if len(improved) > best_score:
            best_score = len(improved)
            best_edges = improved
    
    # Strategy 3: More greedy restarts with different heuristics + local search
    for seed in range(300, 1200):
        if time.time() > deadline - 10.0:
            break
        
        edges = _greedy_construct_heuristic(n, seed)
        score = len(edges)
        
        if score > best_score:
            best_score = score
            best_edges = edges[:]
        
        # Quick enhanced local search on promising results
        if score >= best_score - 3 and time.time() < deadline - 5.0:
            improved = _enhanced_local_search(n, edges, deadline)
            if len(improved) > best_score:
                best_score = len(improved)
                best_edges = improved[:]
    
    return best_edges


def _greedy_construct(n, seed):
    """
    Greedy construction of K2,3-free graph.
    Start empty, add edges in random order if they don't create K2,3.
    """
    random.seed(seed)
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_edges)
    
    adj = [set() for _ in range(n)]
    edges = []
    
    for u, v in all_edges:
        if _can_add_edge(adj, u, v, n):
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
    
    return edges


def _greedy_construct_heuristic(n, seed):
    """
    Greedy with heuristic ordering: prefer edges that keep more flexibility.
    Use a priority based on current degrees.
    """
    random.seed(seed)
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    adj = [set() for _ in range(n)]
    edges = []
    remaining = set(all_edges)
    
    while remaining and len(edges) < 190:
        # Score each remaining edge by how "safe" it is
        best_edge = None
        best_score = -1
        
        # Sample a subset for speed
        candidates = list(remaining)
        if len(candidates) > 100:
            candidates = random.sample(candidates, 100)
        
        for u, v in candidates:
            if not _can_add_edge(adj, u, v, n):
                continue
            
            # Heuristic: prefer edges between lower-degree vertices
            # This keeps options open for future edges
            score = -(len(adj[u]) + len(adj[v]))
            
            if score > best_score:
                best_score = score
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        edges.append([u, v])
        remaining.discard((u, v))
    
    return edges


def _can_add_edge(adj, u, v, n):
    """
    Check if adding edge (u,v) keeps the graph K2,3-free.
    K2,3: two vertices with 3+ common neighbors.
    """
    # After adding (u,v), we need to check:
    # 1. Pair (u,v): common neighbors = adj[u] ∩ adj[v]
    #    (u and v are not in each other's adj yet, so this is just the intersection)
    common_uv = adj[u] & adj[v]
    if len(common_uv) >= 3:
        return False
    
    # 2. For each w in adj[v]: pair (u,w)
    #    New common neighbors of u and w = (adj[u] ∪ {v}) ∩ adj[w]
    #    = (adj[u] ∩ adj[w]) ∪ ({v} ∩ adj[w])
    #    Since v is in adj[v] and we're checking w in adj[v], v is NOT necessarily in adj[w]
    #    Wait: w is in adj[v], so v is in adj[w] (undirected). So {v} ∩ adj[w] = {v}
    for w in adj[v]:
        if w == u:
            continue
        common = adj[u] & adj[w]
        # v is in adj[w] because w in adj[v] and graph is undirected
        common = common | {v}
        if len(common) >= 3:
            return False
    
    # 3. For each w in adj[u]: pair (v,w)
    #    New common neighbors of v and w = (adj[v] ∪ {u}) ∩ adj[w]
    #    = (adj[v] ∩ adj[w]) ∪ ({u} ∩ adj[w])
    #    u is in adj[w] because w in adj[u] and graph is undirected
    for w in adj[u]:
        if w == v:
            continue
        common = adj[v] & adj[w]
        common = common | {u}
        if len(common) >= 3:
            return False
    
    return True


def _enhanced_local_search(n, edges, deadline):
    """
    Enhanced local search with edge swaps and 2-opt moves.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    current_edges = [e[:] for e in edges]
    current_score = len(current_edges)
    
    # Phase 1: Add all possible missing edges
    missing = [(u, v) for u in range(n) for v in range(u + 1, n) if v not in adj[u]]
    random.shuffle(missing)
    
    added = True
    while added and time.time() < deadline - 5.0:
        added = False
        for u, v in missing:
            if v in adj[u]:  # already added
                continue
            if _can_add_edge(adj, u, v, n):
                adj[u].add(v)
                adj[v].add(u)
                current_edges.append([u, v])
                added = True
    
    # Phase 2: Edge swap moves - remove one edge, try to add two
    for iteration in range(200):
        if time.time() > deadline - 3.0:
            break
        
        # Pick a random edge to remove
        if not current_edges:
            break
        
        idx = random.randint(0, len(current_edges) - 1)
        u, v = current_edges[idx]
        
        # Remove it
        adj[u].discard(v)
        adj[v].discard(u)
        current_edges.pop(idx)
        
        # Try to add all possible missing edges
        missing2 = [(a, b) for a in range(n) for b in range(a + 1, n) if b not in adj[a]]
        random.shuffle(missing2)
        
        for a, b in missing2:
            if _can_add_edge(adj, a, b, n):
                adj[a].add(b)
                adj[b].add(a)
                current_edges.append([a, b])
        
        # If we didn't gain edges, the move was neutral or worse
        # We keep the state as is (may have gained 1+ edges)
    
    # Phase 3: 2-opt moves - remove two edges, try to add three
    for iteration in range(100):
        if time.time() > deadline - 3.0:
            break
        
        if len(current_edges) < 2:
            break
        
        # Pick two random edges to remove
        idx1 = random.randint(0, len(current_edges) - 1)
        idx2 = random.randint(0, len(current_edges) - 1)
        if idx1 == idx2:
            continue
        
        u1, v1 = current_edges[idx1]
        u2, v2 = current_edges[idx2]
        
        # Remove them
        adj[u1].discard(v1)
        adj[v1].discard(u1)
        adj[u2].discard(v2)
        adj[v2].discard(u2)
        current_edges.pop(max(idx1, idx2))
        current_edges.pop(min(idx1, idx2))
        
        # Try to add all possible missing edges
        missing3 = [(a, b) for a in range(n) for b in range(a + 1, n) if b not in adj[a]]
        random.shuffle(missing3)
        
        for a, b in missing3:
            if _can_add_edge(adj, a, b, n):
                adj[a].add(b)
                adj[b].add(a)
                current_edges.append([a, b])
    
    return current_edges


# EVOLVE-BLOCK-END
