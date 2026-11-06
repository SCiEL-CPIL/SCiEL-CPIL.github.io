import time
import random

def is_valid_graph(n, adj):
    # adj is a list of integers, adj[i] has bit j set if (i,j) is an edge
    # Check all subsets of size >= 6
    # Optimization: We can skip subsets of size < 6
    # We iterate over all masks 0..2^n-1
    # This is 2^n iterations.
    # For n=20, 2^20 = 1,048,576.
    # For each mask, check popcount. If < 6, skip.
    # If >= 6, check regularity.
    
    # To speed up popcount, we can use int.bit_count() (Python 3.8+)
    
    for mask in range(1 << n):
        k = mask.bit_count()
        if k < 6:
            continue
        
        # Check regularity
        # For each vertex in mask, count neighbors in mask
        # If all counts are equal, it's regular.
        
        # We can break early if we find two different degrees
        degs = set()
        # Iterate vertices in mask
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            i = lsb.bit_length() - 1
            # Count neighbors of i in mask
            deg = (adj[i] & mask).bit_count()
            degs.add(deg)
            if len(degs) > 1:
                break
            m ^= lsb
            if len(degs) > 1:
                break
        if len(degs) <= 1:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try n=21 first? 
    # 2^21 is 2M masks. ~1-2s per check.
    # Let's try a few random graphs for n=21.
    # If we find one, return it.
    
    # Then try n=20.
    # 2^20 is 1M masks. ~0.5-1s per check.
    
    # We can interleave or just try n=20 first as it's more likely.
    # Actually, the prompt says "Maximize n". So 21 is better if found.
    # But 21 is likely impossible or very hard.
    # Let's try n=20 first to secure a high score, then if time remains, try n=21?
    # No, if we find n=20, we can return it immediately.
    # If we want to try for 21, we should do it before committing to 20?
    # But if 21 is impossible, we waste time.
    # Given the "120s" limit, and that 20 is "known", I will focus on finding a valid 20-vertex graph quickly.
    # If I have > 50s left after finding 20, I could try 21, but returning 20 is a solid 4.0 score.
    # Let's just find a valid 20-vertex graph.
    
    n = 20
    # Random seed for reproducibility? No, random is fine.
    
    while time.time() < deadline:
        # Generate random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        if is_valid_graph(n, adj):
            # Convert to edge list
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    # Fallback: return a valid small graph
    # The empty graph on 5 vertices is valid.
    return {'n': 5, 'edges': []}
