import random
import time
import sys

def construct():
    n = 20
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    def popcount(x):
        return bin(x).count("1")
        
    def is_valid_add(u, v):
        # Check if adding edge (u, v) keeps the graph K2,3-free
        # 1. Check pair (u, v)
        # Common neighbors of u and v in current graph: adj[u] & adj[v]
        # After adding, common neighbors are still adj[u] & adj[v] (since u,v are not neighbors of themselves)
        # Wait, if u and v are connected, do they count as common neighbors of each other? No.
        # So count = popcount(adj[u] & adj[v])
        if popcount(adj[u] & adj[v]) > 2:
            return False
            
        # 2. Check pairs (u, w) for all w != u, v
        # New adj[u] will have bit v set.
        # For a fixed w, new common neighbors of u and w are:
        # (adj[u] | (1 << v)) & adj[w]
        # This is (adj[u] & adj[w]) | (adj[w] & (1 << v))
        # We need popcount of this to be <= 2.
        
        # Optimization: 
        # current_common_uw = adj[u] & adj[w]
        # new_common_uw = current_common_uw | (1 << v if (adj[w] >> v) & 1 else 0)
        # Note: if v is already in adj[u], this logic is wrong, but we only call this for missing edges.
        # So v is NOT in adj[u].
        
        # Similarly for (v, w).
        
        for w in range(n):
            if w == u or w == v:
                continue
            
            # Check pair (u, w)
            # current common: adj[u] & adj[w]
            # added common: v if v is in adj[w]
            # So new count = popcount(adj[u] & adj[w]) + (1 if (adj[w] >> v) & 1 else 0)
            # But wait, does v intersect with adj[u] & adj[w]?
            # v is not in adj[u] (since edge is missing).
            # So the sets are disjoint.
            cuw = popcount(adj[u] & adj[w])
            if (adj[w] >> v) & 1:
                cuw += 1
            if cuw > 2:
                return False
                
            # Check pair (v, w)
            cvw = popcount(adj[v] & adj[w])
            if (adj[w] >> u) & 1:
                cvw += 1
            if cvw > 2:
                return False
                
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def count_edges():
        return sum(popcount(a) for a in adj) // 2

    # Initial state: Start with seed or random
    # Let's start with the seed: 5 disjoint K4s
    for k in range(5):
        base = 4 * k
        for a in range(4):
            for b in range(a + 1, 4):
                add_edge(base + a, base + b)
                
    best_edges = count_edges()
    best_adj = adj[:]
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Local Search
    # Strategy: Try to add random edges. If stuck, do random swaps.
    # To be more aggressive, we can use a "temperature" or just random restarts if stuck for too long.
    
    current_edges = best_edges
    current_adj = best_adj[:]
    
    # Greedy phase: try to add as many edges as possible
    # Randomize order of missing edges
    missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (current_adj[i] >> j) & 1]
    random.shuffle(missing)
    
    for u, v in missing:
        if time.time() > deadline:
            break
        if is_valid_add(u, v):
            add_edge(u, v)
            current_edges += 1
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = current_adj[:]
                
    # Randomization/Annealing phase
    # We will perform random moves:
    # 1. Pick a random missing edge (u, v).
    # 2. If valid, add it.
    # 3. If invalid, pick a random existing edge (a, b).
    #    Remove (a, b).
    #    Try to add (u, v).
    #    If valid, keep swap.
    #    If invalid, revert (add (a, b) back).
    
    # To make it faster, precompute list of existing edges?
    # Or just pick random u, v and check.
    
    # We need to efficiently pick a random missing edge and a random existing edge.
    # For N=20, total edges 190.
    
    while time.time() < deadline:
        # Pick random missing edge
        while True:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u < v and not (current_adj[u] >> v) & 1:
                break
        
        if is_valid_add(u, v):
            add_edge(u, v)
            current_edges += 1
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = current_adj[:]
        else:
            # Try a swap
            # Pick random existing edge
            while True:
                a = random.randint(0, n - 1)
                b = random.randint(0, n - 1)
                if a < b and (current_adj[a] >> b) & 1:
                    break
            
            # Remove (a, b)
            remove_edge(a, b)
            current_edges -= 1
            
            # Try to add (u, v)
            if is_valid_add(u, v):
                add_edge(u, v)
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = current_adj[:]
            else:
                # Revert
                add_edge(a, b)
                current_edges += 1
                
    # Reconstruct best_adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
