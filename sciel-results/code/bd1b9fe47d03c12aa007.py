"""
Search for a large graph with no regular induced subgraph on 6+ vertices.
Uses bitmask representation and optimized bit-counting for speed.
"""
import time
import random
import itertools

def construct():
    start = time.time()
    deadline = start + 140  # 10s safety margin before 150s kill
    
    best_n = 0
    best_edges = []
    
    # Prioritize finding n=20, then try n=21 if time permits
    targets = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]
    
    # If we find a valid n=20, we can attempt n=21
    # But n=21 is much harder. Let's stick to finding the best possible n efficiently.
    
    for target_n in targets:
        if time.time() > deadline - 15:
            break
        
        found = False
        # Try more seeds for smaller n if n=20 fails quickly?
        # No, just try a reasonable number of seeds.
        max_seeds = 1000 if target_n <= 18 else 500
        
        for seed in range(max_seeds):
            if time.time() > deadline - 15:
                break
            
            rng = random.Random(seed)
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if rng.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            if _is_valid(adj, target_n, deadline):
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                if target_n > best_n:
                    best_n = target_n
                    best_edges = edges
                found = True
                break
        
        if found:
            # If we found a valid graph for this target_n, 
            # should we try a larger one? 
            # The loop iterates 20, 19... so if we find 20, we break.
            # If we want to try 21, we need to add it to the list.
            # Let's add 21 to the front if we have time?
            # The loop structure breaks on first found.
            # So if we find 20, we return 20.
            # To try 21, we should check if we have time after finding 20.
            # But the loop breaks.
            
            # Let's restructure: 
            # Try 20. If found, store it.
            # If time remains, try 21.
            # If 21 found, store it.
            # Return best.
            
            # Since the loop goes 20 down to 10, and breaks on first found,
            # it will always return the highest n it finds in this specific order.
            # So if it finds 20, it stops. It won't try 19.
            # This is correct for maximizing n.
            
            # To try 21, we need to modify the list to include 21.
            # But 21 is very hard. 
            # Let's just ensure we find 20 reliably and fast.
            
            break
    
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: trivially valid small graph
    return {'n': 5, 'edges': []}


def _is_valid(adj, n, deadline):
    """
    Check if no subset of size >= 6 induces a regular graph.
    Returns True if valid, False otherwise.
    Uses early termination and bit_count.
    """
    
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            # Time check
            if time.time() > deadline:
                # If we run out of time, we assume valid? 
                # No, that's dangerous. 
                # But if we don't return, we might kill the process.
                # Returning True is a gamble. Returning False is safe but might miss a valid graph.
                # Given the deadline is strict, we should probably return False to be safe?
                # Or break out and let the caller handle it?
                # The caller checks time before starting a new seed.
                # If we exceed deadline here, we are in trouble.
                # Let's return False to stop the search for this candidate.
                return False
            
            # Compute mask
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Check regularity
            # Get degree of first vertex
            v0 = subset[0]
            d0 = (adj[v0] & mask).bit_count()
            
            is_regular = True
            for v in subset[1:]:
                dv = (adj[v] & mask).bit_count()
                if dv != d0:
                    is_regular = False
                    break
            
            if is_regular:
                return False
    
    return True
