import random
import time
import itertools
import sys

def popcount(x):
    return bin(x).count('1')

def count_bad_subsets(adj, n):
    """
    Counts the number of induced subgraphs on k>=6 vertices that are regular.
    adj: list of n integers, each is a bitmask of neighbors.
    n: number of vertices.
    """
    bad_count = 0
    # Iterate over all subset sizes k from 6 to n
    for k in range(6, n + 1):
        # Iterate over all combinations of k vertices
        # Using itertools.combinations is slower than bit iteration for large n, 
        # but for n=20, k=6..20, it's manageable.
        # However, generating combinations of size 10 from 20 is 184k.
        # Total subsets ~ 485k.
        # For each subset, we check regularity.
        for combo in itertools.combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            # Check if induced subgraph on 'mask' is regular
            # Degree of v in mask is popcount(adj[v] & mask)
            # We need all degrees to be equal.
            # Optimization: check first vertex's degree, then compare others.
            first_deg = popcount(adj[combo[0]] & mask)
            is_regular = True
            for v in combo[1:]:
                if popcount(adj[v] & mask) != first_deg:
                    is_regular = False
                    break
            if is_regular:
                bad_count += 1
    return bad_count

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_n = 5
    best_edges = []
    
    # We aim for n=20. If we can't find it, we fall back to 5.
    # We could try n=20, then 19, etc., but 20 is the target.
    
    target_n = 20
    
    # Try to find a valid graph for target_n
    found = False
    best_score = float('inf')
    best_adj = None
    
    # Multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialize random graph
        adj = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_score = count_bad_subsets(adj, target_n)
        
        if current_score == 0:
            # Found a valid graph
            edges = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': target_n, 'edges': edges}
        
        if current_score < best_score:
            best_score = current_score
            best_adj = adj[:]
            
        # Hill climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try flipping random edges
            # Shuffle order of edges to avoid bias? Or just pick random pair.
            # Picking random pair is O(1).
            # To be thorough, we could iterate all pairs, but that's 190 pairs.
            # 190 * cost_of_count. Cost of count is ~5ms. 190*5ms = 1s per full pass.
            # This is too slow for many passes.
            # So we do random restarts and random moves.
            
            # Pick a random edge
            u = random.randint(0, target_n - 1)
            v = random.randint(0, target_n - 1)
            if u == v:
                continue
                
            # Flip edge (u, v)
            mask_uv = (1 << u) | (1 << v)
            # Check if edge exists
            has_edge = (adj[u] >> v) & 1
            
            # Remove edge
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            new_score = count_bad_subsets(adj, target_n)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score == 0:
                    edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    return {'n': target_n, 'edges': edges}
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
    # If we didn't find a valid graph for 20, check if we have a good best_adj?
    # No, best_adj is not necessarily valid.
    # We must return a VALID graph.
    # We know n=5 is valid.
    # We could try n=10, 15 etc if we have time, but n=20 is the goal.
    # If we fail n=20, we return n=5.
    
    # As a last resort, try a smaller n if time permits?
    # The task says "Maximize n".
    # If we can't get 20, maybe we can get 10?
    # But finding a valid graph for 20 is the primary goal.
    
    # Fallback: Empty graph on 5 vertices is always valid.
    return {'n': 5, 'edges': []}
