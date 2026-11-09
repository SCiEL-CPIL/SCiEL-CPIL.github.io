import time
import random
import numpy as np

N = 40
DEADLINE = 115  # Seconds, safety margin for 120s limit

def _creates_c6(adj, u, v):
    """
    Checks if adding edge (u, v) creates a C6.
    Path: u - a - b - c - d - v
    """
    nu = adj[u]
    nv = adj[v]
    
    # Iterate a in N(u)
    a_mask = nu
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a
        
        if a == v: continue
        
        # Iterate d in N(v)
        d_mask = nv
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d
            
            if d == u or d == a: continue
            
            # We need b in N(a) and c in N(d) such that (b, c) is an edge.
            # b, c must be distinct from u, v, a, d.
            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << d)
            
            # Candidates for b: neighbors of a, excluding forbidden
            b_mask = adj[a] & ~forbidden
            if not b_mask: continue
            
            # Candidates for c: neighbors of d, excluding forbidden
            c_mask = adj[d] & ~forbidden
            if not c_mask: continue
            
            # Check if any b in b_mask is connected to any c in c_mask
            bm = b_mask
            while bm:
                lsb_b = bm & -bm
                b = lsb_b.bit_length() - 1
                bm ^= lsb_b
                if (adj[b] & c_mask):
                    return True
                        
    return False

def _local_search(adj, count, start_time, deadline):
    """
    Attempts to improve the graph by removing one edge and adding more than one.
    Returns the improved adjacency list and count.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # List of all possible edges
        all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        current_edges_set = set()
        for i in range(N):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if j > i:
                    current_edges_set.add((i, j))
                mask ^= lsb
        
        # Try removing each edge
        for e in list(current_edges_set):
            if time.time() >= deadline:
                break
            
            u, v = e
            # Temporarily remove e
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            count -= 1
            
            # Try to add as many other edges as possible
            added_edges = []
            for f in all_edges:
                if f == e:
                    continue
                if f in current_edges_set:
                    continue # Already in graph (excluding e)
                
                fu, fv = f
                if not _creates_c6(adj, fu, fv):
                    # Can add f
                    added_edges.append(f)
                    # Note: We don't actually add it to adj yet to keep the state clean for checking other f's?
                    # No, adding f might block other g's. But we want to maximize the number of added edges.
                    # Greedily adding them might block others. 
                    # However, usually, if we can add k edges, we want to add all of them.
                    # Let's just count them for now to see if it's profitable.
                    # Actually, we should add them to check if further edges become possible?
                    # For a 1-swap, we just check if the set of addable edges > 1.
                    # If we add one, it might block another.
                    # So we should check the maximum independent set of addable edges? That's hard.
                    # Simple heuristic: Just count how many are *individually* addable in the state without e.
                    # If count > 1, we swap.
            
            if len(added_edges) > 1:
                # Perform the swap: Add all added_edges
                for f in added_edges:
                    fu, fv = f
                    adj[fu] |= (1 << fv)
                    adj[fv] |= (1 << fu)
                    current_edges_set.add(f)
                count += len(added_edges)
                current_edges_set.remove(e)
                improved = True
            else:
                # Restore e
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
    
    return adj, count

def construct():
    start_time = time.time()
    best_edges = []
    best_count = 0
    
    # Generate all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Run random restarts until deadline
    while time.time() - start_time < DEADLINE:
        random.shuffle(all_edges)
        adj = [0] * N
        count = 0
        
        for u, v in all_edges:
            if not _creates_c6(adj, u, v):
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        
        # Local search refinement
        # Only do local search if we have enough time and the graph is good enough
        if count > best_count and time.time() - start_time < DEADLINE - 10:
            adj, count = _local_search(adj, count, start_time, DEADLINE - 5)
        
        if count > best_count:
            best_count = count
            # Reconstruct edge list
            best_edges = []
            for i in range(N):
                for j in range(i+1, N):
                    if (adj[i] >> j) & 1:
                        best_edges.append([i, j])
                        
    return best_edges
