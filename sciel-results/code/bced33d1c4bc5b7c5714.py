"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def _compute_book_sizes(edges, n=40):
    """Compute the book size for each edge in the graph."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    max_book = 0
    for u, v in edges:
        common = adj[u] & adj[v]
        if len(common) > max_book:
            max_book = len(common)
    return max_book

def _simulate_annealing(n=40, target_edges=401, time_limit=140.0, seed=42):
    """Simulated annealing with efficient incremental updates."""
    start_time = time.time()
    random.seed(seed)
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_possible)
    current_edges = all_possible[:target_edges]
    edge_set = set((min(u,v), max(u,v)) for u,v in current_edges)
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in current_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Compute initial book sizes for all edges
    book_sizes = {}
    for u, v in current_edges:
        key = (min(u,v), max(u,v))
        book_sizes[key] = len(adj[u] & adj[v])
    
    current_max = max(book_sizes.values()) if book_sizes else 0
    
    best_edges = [list(e) for e in current_edges]
    best_max = current_max
    
    # SA parameters
    T0 = 8.0
    Tend = 0.05
    alpha = 0.9995
    
    T = T0
    while T > Tend and time.time() - start_time < time_limit:
        # Pick a random edge to remove
        remove_idx = random.randint(0, len(current_edges) - 1)
        ru, rv = current_edges[remove_idx]
        rkey = (min(ru, rv), max(ru, rv))
        
        # Pick a random non-edge to add
        cu, cv = random.choice(all_possible)
        ckey = (min(cu, cv), max(cu, cv))
        if ckey in edge_set:
            continue
        
        # Identify affected edges: those sharing a vertex with ru,rv or cu,cv
        affected = set()
        affected.add(rkey)
        affected.add(ckey)
        for w in adj[ru]:
            k = (min(ru, w), max(ru, w))
            if k in edge_set and k != rkey:
                affected.add(k)
        for w in adj[rv]:
            k = (min(rv, w), max(rv, w))
            if k in edge_set and k != rkey:
                affected.add(k)
        for w in adj[cu]:
            k = (min(cu, w), max(cu, w))
            if k in edge_set:
                affected.add(k)
        for w in adj[cv]:
            k = (min(cv, w), max(cv, w))
            if k in edge_set:
                affected.add(k)
        
        # Save old book sizes for affected edges
        old_books = {k: book_sizes.get(k, 0) for k in affected}
        
        # Apply the swap: remove rkey, add ckey
        del book_sizes[rkey]
        edge_set.remove(rkey)
        adj[ru].remove(rv)
        adj[rv].remove(ru)
        
        edge_set.add(ckey)
        adj[cu].add(cv)
        adj[cv].add(cu)
        current_edges[remove_idx] = [cu, cv]
        
        # Recompute book sizes only for affected edges
        new_max = 0
        for k in affected:
            if k in edge_set:
                u, v = k
                bs = len(adj[u] & adj[v])
                book_sizes[k] = bs
                if bs > new_max:
                    new_max = bs
            else:
                if k in book_sizes:
                    del book_sizes[k]
        
        # If the removed edge was the max, we need to check non-affected edges too
        # But for efficiency, we check: if old max edge is still present and unaffected, max unchanged
        # Simplest correct approach: if we deleted the max edge, recompute global max
        # Optimization: track if rkey was the unique max
        was_max_edge = old_books.get(rkey, 0) == current_max
        if was_max_edge:
            # Need to find new max from all edges
            new_max = max(book_sizes.values()) if book_sizes else 0
        else:
            # Max edge unaffected, so new_max = max(old non-affected max, new affected max)
            # We don't track non-affected max separately, so use a cheaper check:
            # new_max is at most max(current_max, max of affected new books)
            # But if current_max edge is not affected, it's still current_max
            # So new_max = max(current_max, max_new_affected)
            # We already computed new_max as max of affected; combine with current_max
            if current_max > new_max:
                new_max = current_max
        
        delta = new_max - current_max
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_edges = [list(e) for e in current_edges]
        else:
            # Revert
            # Remove ckey
            edge_set.remove(ckey)
            adj[cu].remove(cv)
            adj[cv].remove(cu)
            if ckey in book_sizes:
                del book_sizes[ckey]
            
            # Restore rkey
            edge_set.add(rkey)
            adj[ru].add(rv)
            adj[rv].add(ru)
            
            # Restore book sizes for affected
            for k in affected:
                if k in old_books:
                    if k in edge_set:
                        book_sizes[k] = old_books[k]
                    else:
                        if k in book_sizes:
                            del book_sizes[k]
                else:
                    if k in book_sizes:
                        del book_sizes[k]
            
            current_edges[remove_idx] = [ru, rv]
        
        T *= alpha
    
    return best_edges, best_max

def construct():
    start_time = time.time()
    n = 40
    target_edges = 401
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_edges = None
    best_score = float('inf')
    
    # Try multiple SA runs with different seeds
    seeds = [42, 123, 456, 789, 1000, 2024, 7, 99]
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
        time_limit = min(17.0, deadline - time.time() - 1.0)
        if time_limit < 2.0:
            break
        edges, score = _simulate_annealing(n, target_edges, time_limit=time_limit, seed=seed)
        if score < best_score:
            best_score = score
            best_edges = edges
    
    if best_edges is None:
        # Fallback: bipartite + 1 edge
        best_edges = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        best_score = _compute_book_sizes(best_edges, n)
    
    return best_edges
# EVOLVE-BLOCK-END
