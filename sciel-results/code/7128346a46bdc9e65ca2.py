import random
import time

# Precomputation of 4-sets and disjoint pairs
# We need to check if any 4+4 bipartite subgraph has >= 12 edges.
# This is equivalent to containing Q3.

def _precompute_sets():
    sets = []
    set_masks = []
    # Generate all combinations of 4 vertices from 16
    from itertools import combinations
    for combo in combinations(range(16), 4):
        sets.append(combo)
        mask = 0
        for v in combo:
            mask |= (1 << v)
        set_masks.append(mask)
    return sets, set_masks

SETS, SET_MASKS = _precompute_sets()
NUM_SETS = len(SETS)

# Precompute disjoint pairs (A, B)
# A and B are indices into SETS
# We store (v1, v2, v3, v4, mask_B) for fast access
PAIRS = []
for i in range(NUM_SETS):
    mask_i = SET_MASKS[i]
    for j in range(i + 1, NUM_SETS):
        # Check if disjoint
        if (mask_i & SET_MASKS[j]) == 0:
            # We can treat (i, j) and (j, i) as the same for counting, 
            # but the edge count is symmetric.
            # However, the vertices in A are the ones we sum over.
            # e(G[A,B]) = sum_{u in A} deg(u, B)
            # This is symmetric in A and B.
            # So we just need one direction.
            # But to be safe and simple, we just store the vertices of the first set
            # and the mask of the second set.
            # Since the sum is symmetric, it doesn't matter which is A and which is B.
            v1, v2, v3, v4 = SETS[i]
            mask_b = SET_MASKS[j]
            PAIRS.append((v1, v2, v3, v4, mask_b))

def _find_q3(adj):
    """
    Returns True if the graph (adj list of bitmasks) contains a Q3 subgraph.
    A Q3 subgraph exists iff there is a 4+4 bipartite subgraph with >= 12 edges.
    """
    # Fast path: if total edges < 12, impossible. 
    # But we don't track total edges easily here without extra cost.
    # Just iterate pairs.
    for v1, v2, v3, v4, mask_b in PAIRS:
        # Count edges from {v1,v2,v3,v4} to mask_b
        # Using bit_count for speed
        c = (adj[v1] & mask_b).bit_count()
        if c < 12: # Early exit? No, we need sum of 4.
             c += (adj[v2] & mask_b).bit_count()
             if c < 12:
                 c += (adj[v3] & mask_b).bit_count()
                 if c < 12:
                     c += (adj[v4] & mask_b).bit_count()
                     if c >= 12:
                         return True
                 else:
                     return True
             else:
                 return True
        else:
            return True
    return False

def _count_edges(adj):
    total = 0
    for mask in adj:
        total += mask.bit_count()
    return total // 2

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_adj = None
    best_count = 0
    
    rng = random.Random(42)
    
    # Number of restarts depends on time, but we loop until deadline
    restart = 0
    
    while True:
        if time.time() > deadline:
            break
        restart += 1
        
        # Initialize adj
        adj = [0] * n
        
        # Strategy: Start with a random graph and remove edges to make it Q3-free
        # Or start empty and add? 
        # Greedy addition is slow to start but gives a valid graph.
        # Random dense + removal is faster to get a decent base.
        
        # Let's try: Random graph with p=0.8
        for u, v in all_edges:
            if rng.random() < 0.8:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Remove edges to make it Q3-free
        # This might take a while if we check _find_q3 every time.
        # We can be smarter: if _find_q3 is True, find a "conflict" and remove an edge from it?
        # _find_q3 doesn't return the conflict.
        # Let's just randomly remove edges until valid.
        
        while _find_q3(adj):
            if time.time() > deadline:
                break
            # Pick a random edge to remove
            # To be more effective, we could pick an edge from a dense area, but random is okay for now
            u, v = rng.choice(all_edges)
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        if _find_q3(adj):
            continue # If we ran out of time making it valid, skip
            
        cur_count = _count_edges(adj)
        if cur_count > best_count:
            best_count = cur_count
            best_adj = list(adj)
            
        # Local Search: Saturation + Kicks
        # We will perform local search for a fixed duration or number of steps
        # Given the speed of _find_q3, we can do many steps.
        
        while time.time() < deadline - 2.0: # Leave 2s for next restart or exit
            
            # 1. Saturation: Try to add any missing edge
            improved = False
            # Shuffle missing edges for randomness in tie-breaking
            missing = [(u, v) for u, v in all_edges if not (adj[u] & (1 << v))]
            rng.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline - 2.0:
                    break
                # Try adding
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if not _find_q3(adj):
                    cur_count += 1
                    improved = True
                    if cur_count > best_count:
                        best_count = cur_count
                        best_adj = list(adj)
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
            # If no improvement, we are at a local maximum
            if not improved:
                # 2. Kick: Remove some random edges to escape
                # Number of edges to remove: 1 to 3
                num_kicks = rng.randint(1, 3)
                existing = [(u, v) for u, v in all_edges if adj[u] & (1 << v)]
                if not existing:
                    break # Empty graph, can't kick
                
                for _ in range(num_kicks):
                    if not existing:
                        break
                    u, v = rng.choice(existing)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    cur_count -= 1
                    # Remove from existing list to avoid re-picking? 
                    # It's okay to re-pick, it just means we might remove fewer unique edges.
                    # For simplicity, we don't update 'existing' list immediately, 
                    # but we check adj in the loop condition? 
                    # Actually, rng.choice(existing) might pick an edge we just removed.
                    # That's fine, it just wastes a kick.
                    
                # After kick, loop back to saturation

    # Convert best_adj to edge list
    if best_adj is None:
        # Fallback: return empty graph or a simple valid graph
        # Should not happen if we have time
        return []
        
    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lb = mask & (-mask)
            j = lb.bit_length() - 1
            mask ^= lb
            if i < j:
                edges.append([i, j])
                
    return edges
