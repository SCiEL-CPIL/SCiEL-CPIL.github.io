"""Rewritten program: fast branch-and-bound for largest triangle-free induced subgraph
   with improved local search on K4-free graphs on 18 vertices."""

import random
import time
import math

N = 18
DEADLINE = 115.0  # seconds

# Precompute popcount for all 2^18 masks
_popcount = [0] * (1 << N)
for _i in range(1, 1 << N):
    _popcount[_i] = _popcount[_i >> 1] + (_i & 1)


def _edge_set_to_edges(es):
    return [[u, v] for (u, v) in sorted(es)]


def _build_adj_mask(es):
    """Return adjacency as list of bitmasks."""
    adj = [0] * N
    for u, v in es:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def _is_k4_free_mask(adj):
    """Check if adjacency bitmasks form a K4-free graph."""
    for i in range(N):
        for j in range(i + 1, N):
            if (adj[i] >> j) & 1:
                common = adj[i] & adj[j]
                # For any two common neighbors k, l, check if k-l is an edge
                # K4 exists iff there's a triangle in the common neighborhood
                k_bits = common
                while k_bits:
                    k = (k_bits & -k_bits).bit_length() - 1
                    k_bits &= k_bits - 1
                    l_bits = k_bits & adj[k]
                    if l_bits:
                        return False
    return True


def _largest_triangle_free_mask(adj):
    """
    Find the largest subset of vertices inducing a triangle-free subgraph.
    Uses branch-and-bound with bitmasks.
    """
    # Precompute all triangle masks
    tri_masks = []
    for i in range(N):
        for j in range(i + 1, N):
            if (adj[i] >> j) & 1:
                common = adj[i] & adj[j]
                k_bits = common
                while k_bits:
                    k = (k_bits & -k_bits).bit_length() - 1
                    k_bits &= k_bits - 1
                    if k > j:
                        tri_masks.append((1 << i) | (1 << j) | (1 << k))
    
    best = 0
    
    # Branch and bound: try to build a large triangle-free set
    # We process vertices in order, deciding to include or exclude
    # Prune if current_size + remaining <= best
    
    def branch(pos, current_mask, current_size):
        nonlocal best
        if current_size + (N - pos) <= best:
            return
        if pos == N:
            if current_size > best:
                best = current_size
            return
        
        # Option 1: exclude vertex pos
        branch(pos + 1, current_mask, current_size)
        
        # Option 2: include vertex pos (if it doesn't create a triangle)
        new_mask = current_mask | (1 << pos)
        # Check if adding vertex pos creates any triangle with existing vertices in current_mask
        valid = True
        for tm in tri_masks:
            if (new_mask & tm) == tm:
                valid = False
                break
        if valid:
            branch(pos + 1, new_mask, current_size + 1)
    
    branch(0, 0, 0)
    return best


def _score_graph(es):
    """Return the largest triangle-free induced subgraph size."""
    adj = _build_adj_mask(es)
    return _largest_triangle_free_mask(adj)


def _initial_graph():
    """Start with a random K4-free graph using random tripartite structure."""
    parts = list(range(N))
    random.shuffle(parts)
    sizes = [6, 6, 6]
    p0 = parts[:6]
    p1 = parts[6:12]
    p2 = parts[12:18]
    
    es = set()
    for a in p0:
        for b in p1:
            es.add((min(a, b), max(a, b)))
    for a in p0:
        for b in p2:
            es.add((min(a, b), max(a, b)))
    for a in p1:
        for b in p2:
            es.add((min(a, b), max(a, b)))
    
    return es


def _try_add_edge(es, u, v):
    """Try to add edge (u,v) if it keeps the graph K4-free."""
    key = (min(u, v), max(u, v))
    if key in es:
        return es
    
    adj = _build_adj_mask(es)
    common = adj[u] & adj[v]
    
    # Check if common neighborhood has any edge (would create K4)
    k_bits = common
    while k_bits:
        k = (k_bits & -k_bits).bit_length() - 1
        k_bits &= k_bits - 1
        l_bits = k_bits & adj[k]
        if l_bits:
            return es  # Would create K4
    
    es.add(key)
    return es


def _remove_random_edge(es):
    if not es:
        return es
    e = random.choice(list(es))
    es.discard(e)
    return es


def construct():
    start_time = time.time()
    random.seed(42)
    
    # Try multiple random initial structures
    best_score = N  # worst case
    best_es = set()
    
    # Generate several diverse initial graphs
    initial_graphs = []
    
    # 1. Complete tripartite K6,6,6
    g1 = _initial_graph()
    initial_graphs.append(g1)
    
    # 2. Random K4-free: start with empty, add random edges
    for trial in range(5):
        es = set()
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random.shuffle(all_edges)
        for e in all_edges:
            new_es = _try_add_edge(set(es), e[0], e[1])
            if new_es != es:
                es = new_es
        initial_graphs.append(es)
    
    # 3. Turan-like: complete 4-partite with parts of size ~4-5
    for trial in range(3):
        parts = list(range(N))
        random.shuffle(parts)
        # 4 parts of sizes 5,5,4,4 or similar
        p0 = parts[:5]
        p1 = parts[5:10]
        p2 = parts[10:14]
        p3 = parts[14:18]
        es = set()
        for a in p0:
            for b in p1:
                es.add((min(a, b), max(a, b)))
            for b in p2:
                es.add((min(a, b), max(a, b)))
            for b in p3:
                es.add((min(a, b), max(a, b)))
        for a in p1:
            for b in p2:
                es.add((min(a, b), max(a, b)))
            for b in p3:
                es.add((min(a, b), max(a, b)))
        for a in p2:
            for b in p3:
                es.add((min(a, b), max(a, b)))
        # Verify K4-free
        adj = _build_adj_mask(es)
        if _is_k4_free_mask(adj):
            initial_graphs.append(es)
    
    # Evaluate all initial graphs
    for es in initial_graphs:
        if not es:
            continue
        adj = _build_adj_mask(es)
        if not _is_k4_free_mask(adj):
            continue
        score = _largest_triangle_free_mask(adj)
        if score < best_score:
            best_score = score
            best_es = set(es)
    
    # Local search with simulated annealing
    current_temp = 1.0
    cooling = 0.999
    min_temp = 0.01
    es = set(best_es)
    
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() - start_time < DEADLINE:
        # Try a move
        move_type = random.random()
        new_es = set(es)
        
        if move_type < 0.5:
            # Try to add a random edge
            u, v = random.choice(all_edges)
            new_es = _try_add_edge(new_es, u, v)
            if new_es == es:
                continue
        else:
            # Remove a random edge
            new_es = _remove_random_edge(new_es)
            if new_es == es:
                continue
        
        # Verify K4-free
        adj = _build_adj_mask(new_es)
        if not _is_k4_free_mask(adj):
            continue
        
        new_score = _largest_triangle_free_mask(adj)
        
        if new_score < best_score:
            best_score = new_score
            best_es = set(new_es)
            es = new_es
        elif new_score <= best_score:
            es = new_es
        else:
            delta = new_score - best_score
            if random.random() < math.exp(-delta / current_temp):
                es = new_es
        
        current_temp = max(min_temp, current_temp * cooling)
        
        if best_score <= 5:
            break
    
    return _edge_set_to_edges(best_es)
