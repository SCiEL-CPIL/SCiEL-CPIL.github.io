import time
import random
import itertools
import sys

DEADLINE = 150
SAFETY_MARGIN = 15
START_TIME = None

def find_violation_fast(adj_masks, n):
    """
    Return a violating vertex-subset (size >= 6) whose induced subgraph is regular, or None.
    Optimized: precompute adjacency bitmasks, check size-6 first, early exit on mismatch.
    """
    # Check size-6 subsets first
    for combo in itertools.combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        
        # Calculate degree of first vertex in this subset
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        
        # Check if all other vertices have the same degree
        is_regular = True
        for v in combo[1:]:
            if (adj_masks[v] & mask).bit_count() != d0:
                is_regular = False
                break
        
        if is_regular:
            return combo

    # Check larger subsets (less numerous)
    for k in range(7, n + 1):
        for combo in itertools.combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            d0 = (adj_masks[combo[0]] & mask).bit_count()
            
            is_regular = True
            for v in combo[1:]:
                if (adj_masks[v] & mask).bit_count() != d0:
                    is_regular = False
                    break
            
            if is_regular:
                return combo
    return None

def get_edges_from_masks(adj_masks, n):
    """Convert adjacency masks to edge list."""
    e = []
    for u in range(n):
        mask = adj_masks[u]
        for v in range(u + 1, n):
            if mask & (1 << v):
                e.append([u, v])
    return e

def local_search(adj_masks, n, rng, deadline):
    """
    Flip edges to eliminate regular induced subgraphs. 
    Uses a short-term strategy to allow for more restarts in the main loop.
    """
    cur = adj_masks[:]
    
    # Check if already valid
    viol = find_violation_fast(cur, n)
    if viol is None:
        return n, get_edges_from_masks(cur, n), True

    # Limit local search iterations to avoid getting stuck
    # We rely on the outer loop to restart if this fails
    max_iters = 200 
    it = 0
    while it < max_iters:
        if time.time() > deadline:
            break

        viol = find_violation_fast(cur, n)
        if viol is None:
            return n, get_edges_from_masks(cur, n), True

        vset = set(viol)
        # Collect candidate edges: prefer edges within the violating set
        internal_edges = []
        internal_nonedges = []
        for u in vset:
            for v in vset:
                if u < v:
                    if cur[u] & (1 << v):
                        internal_edges.append((u, v))
                    else:
                        internal_nonedges.append((u, v))
        
        # Strategy: flip an edge in the violating set
        # Randomly choose between removing an edge and adding a non-edge
        # This helps escape local minima where one type of flip doesn't work
        if internal_edges and internal_nonedges:
            if rng.random() < 0.5:
                u, v = rng.choice(internal_edges)
            else:
                u, v = rng.choice(internal_nonedges)
        elif internal_edges:
            u, v = rng.choice(internal_edges)
        elif internal_nonedges:
            u, v = rng.choice(internal_nonedges)
        else:
            # Should not happen if vset size >= 2
            u, v = rng.sample(range(n), 2)
            if u > v: u, v = v, u

        cur[u] ^= (1 << v)
        cur[v] ^= (1 << u)
        it += 1

    # Final check
    viol = find_violation_fast(cur, n)
    return n, get_edges_from_masks(cur, n), viol is None

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    deadline = START_TIME + DEADLINE - SAFETY_MARGIN

    best_n = 0
    best_edges = []
    best_valid = False

    rng = random.Random(12345)

    # Try n from 22 down to 6
    # We prioritize higher n
    for n in range(22, 5, -1):
        if time.time() > deadline:
            break

        # For n=21 and 22, we need many restarts
        # For smaller n, fewer restarts needed
        if n >= 21:
            # Tight deadline per restart to allow many attempts
            # We don't know how many fit, so we just loop until deadline
            pass
        
        while time.time() < deadline:
            # Generate random graph
            # Avoid p=0.5 where regular subgraphs are most likely? 
            # Actually, random graphs at p=0.5 are "most random".
            # Let's try a range of p.
            p = rng.uniform(0.2, 0.8)
            
            adj_masks = [0] * n
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < p:
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                        edges.append([u, v])

            # Check if valid
            viol = find_violation_fast(adj_masks, n)
            if viol is None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    best_valid = True
                # Found a valid graph for this n, try to find a larger one?
                # The loop structure `for n in range(22, 5, -1)` means we break out 
                # and try n-1 only if we didn't find a valid one for n.
                # But if we found a valid one for n, we should stop trying n 
                # and move to n-1? No, we want the MAX n.
                # So if we find a valid graph for n, we can break the inner while loop
                # and proceed to the next (smaller) n? 
                # Wait, if we find a valid graph for n=21, we are done with n=21.
                # We don't need to find another one for n=21.
                # We should break the inner loop and let the outer loop decrement n.
                # But if we find n=22, we break inner, outer decrements to 21, 
                # but we already have 22, so we should probably just return immediately?
                # The task is to MAXIMIZE n.
                # So if we find a valid graph for n, we should record it and 
                # only continue if we are trying to find a larger one?
                # No, the loop goes DOWN. So we find the largest possible n first.
                # If we find a valid graph for n=22, we are done.
                # If we fail n=22, we try n=21.
                # So breaking the inner loop is correct.
                break

            # If invalid, try local search
            new_n, new_edges, new_valid = local_search(adj_masks, n, rng, deadline)
            if new_valid:
                if n > best_n:
                    best_n = n
                    best_edges = new_edges
                    best_valid = True
                break

    if not best_valid or best_n < 5:
        # Fallback: return a small valid graph
        # A graph with no edges on 5 vertices is valid (no regular induced subgraph of size >=6)
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': best_edges}
