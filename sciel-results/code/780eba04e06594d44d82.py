import time
import random
import itertools

# Global state for deadline management
START_TIME = None
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    deadline = START_TIME + DEADLINE - SAFETY_MARGIN
    
    # We want to maximize n. 
    # Known: n=20 is possible. n=21 is the target for improvement.
    # We try from 22 down to 6.
    
    best_n = 0
    best_edges = []
    
    # Fixed seed for reproducibility in the "seed program" context, 
    # but we can vary it if we want to explore. 
    # Inspiration 1 used a fixed seed 12345. 
    # Let's use a fixed seed to ensure consistent behavior, 
    # but allow multiple attempts with different p values.
    base_seed = 12345
    rng = random.Random(base_seed)
    
    for n in range(22, 5, -1):
        if time.time() > deadline:
            break
            
        # If we already found a graph for this n or larger, we can skip 
        # but we are iterating down, so if we found n=21, we return immediately.
        # If we didn't find n=22, we try n=21.
        
        # Number of attempts for this n
        # Larger n is harder, but bitmask is fast. 
        # Let's give it a reasonable number of attempts.
        max_attempts = 20 if n <= 20 else 10
        
        found_valid = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Vary edge probability p
            # Regular induced subgraphs are common at p=0.5. 
            # Avoiding p=0.5 might help, but local search should fix it.
            # Let's sample p from a distribution that avoids 0.5 slightly or just uniform.
            p = rng.uniform(0.1, 0.9)
            
            # Generate random graph
            adj_masks, edges = random_graph(n, p, rng)
            
            # Check for violation
            viol = find_violation_fast(adj_masks, n)
            
            if viol is None:
                # Valid graph found
                best_n = n
                best_edges = edges
                found_valid = True
                break # Move to next n (smaller) only if we didn't find a larger one? 
                      # Actually, we iterate down. If we find n=21, we return it.
                      # If we find n=20, we continue to n=19? No, we want max n.
                      # So if we find a valid graph for n, we should return it immediately 
                      # because we are checking from largest to smallest.
                return {'n': n, 'edges': edges}
            
            # If violation found, try local search to repair
            new_adj_masks, new_edges, is_valid = local_search(adj_masks, n, rng, deadline)
            
            if is_valid:
                best_n = n
                best_edges = new_edges
                return {'n': n, 'edges': new_edges}
        
        # If we exhausted attempts for this n and didn't find a valid one, 
        # continue to smaller n.
        
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
        
    # Fallback: trivial graph
    return {'n': 5, 'edges': []}

def random_graph(n, p, rng):
    """
    Generates a random graph on n vertices with edge probability p.
    Returns (adj_masks, edges) where adj_masks is a list of ints.
    """
    adj_masks = [0] * n
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.append([u, v])
    return adj_masks, edges

def get_edges_from_masks(adj_masks, n):
    """
    Converts adjacency masks to a list of edges.
    """
    edges = []
    for u in range(n):
        mask = adj_masks[u]
        for v in range(u + 1, n):
            if mask & (1 << v):
                edges.append([u, v])
    return edges

def find_violation_fast(adj_masks, n):
    """
    Finds a subset of size >= 6 that induces a regular graph.
    Returns the subset as a tuple, or None if no such subset exists.
    Optimized for speed using bitmasks.
    """
    vertices = range(n)
    
    # Check size 6 first (most numerous, most likely to exist)
    for combo in itertools.combinations(vertices, 6):
        # Create a bitmask for the subset
        mask = 0
        for v in combo:
            mask |= (1 << v)
        
        # Compute degree of the first vertex in the subset
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        
        # Check if all other vertices have the same degree
        is_regular = True
        for v in combo[1:]:
            if (adj_masks[v] & mask).bit_count() != d0:
                is_regular = False
                break
        
        if is_regular:
            return combo
            
    # Check larger subsets
    # For n=21, C(21,7) is 116280, C(21,8) is 203490... 
    # This can get expensive. 
    # However, if we eliminate all size-6 regular subgraphs, 
    # it is *possible* that a size-7+ regular subgraph exists.
    # But typically, if you break the small ones, the large ones break too 
    # or are less likely to be regular.
    # To be safe, we must check all k >= 6.
    
    for k in range(7, n + 1):
        for combo in itertools.combinations(vertices, k):
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

def local_search(adj_masks, n, rng, deadline):
    """
    Attempts to fix a graph by flipping edges in violating subgraphs.
    Returns (adj_masks, edges, is_valid)
    """
    cur_masks = adj_masks[:]
    
    # Initial check
    viol = find_violation_fast(cur_masks, n)
    if viol is None:
        return cur_masks, get_edges_from_masks(cur_masks, n), True
    
    max_iters = 500
    it = 0
    
    while it < max_iters:
        if time.time() > deadline:
            break
            
        viol = find_violation_fast(cur_masks, n)
        if viol is None:
            return cur_masks, get_edges_from_masks(cur_masks, n), True
            
        # Get the violating set
        vset = set(viol)
        
        # Candidate edges to flip:
        # 1. Edges inside the violating set (to break regularity)
        # 2. Non-edges inside the violating set (to break regularity)
        
        internal_edges = []
        internal_nonedges = []
        
        for u in vset:
            for v in vset:
                if u < v:
                    if cur_masks[u] & (1 << v):
                        internal_edges.append((u, v))
                    else:
                        internal_nonedges.append((u, v))
        
        # Strategy: Flip an edge within the violating set.
        # Removing an edge reduces degree of 2 vertices.
        # Adding an edge increases degree of 2 vertices.
        # This disrupts the equality of degrees.
        
        candidates = []
        if internal_edges:
            candidates.extend(internal_edges)
        if internal_nonedges:
            candidates.extend(internal_nonedges)
            
        if not candidates:
            # Should not happen for k>=6, but fallback to random
            u = rng.randint(0, n-1)
            v = rng.randint(0, n-1)
            if u == v:
                v = (v+1) % n
            candidates = [(u, v)]
            
        u, v = rng.choice(candidates)
        
        # Flip the edge
        cur_masks[u] ^= (1 << v)
        cur_masks[v] ^= (1 << u)
        
        it += 1
        
    # Final check
    viol = find_violation_fast(cur_masks, n)
    return cur_masks, get_edges_from_masks(cur_masks, n), viol is None

# Keep the interface exactly as requested
# The task says: construct() -> {'n': n, 'edges': [[u, v], ...]}
# The current program has a nested function structure, but the entry point is construct().
# I have defined construct() at the top level.

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Result: n={res['n']}, edges={len(res['edges'])}")
