import time
import random
import sys

def construct():
    """
    Finds a graph on n vertices with no regular induced subgraph on >= 6 vertices.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # We aim for n=20 first, as it's known to exist.
    # If we find one quickly, we might try 21, but 20 is the safe bet for high score.
    # Actually, the score is n. So 20 is better than 5. 21 would be better than 20.
    # Let's try to find n=20.
    
    n = 20
    
    def popcount(x):
        return x.bit_count()

    def is_regular_subset(mask, adj_masks):
        """
        Check if the induced subgraph on vertices in 'mask' is regular.
        """
        if popcount(mask) < 6:
            return False
        
        # Calculate degrees of vertices in the subset
        degs = []
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            v = lsb.bit_length() - 1
            # Degree of v in the induced subgraph is popcount(adj_masks[v] & mask)
            deg = popcount(adj_masks[v] & mask)
            degs.append(deg)
            m ^= lsb
            
        # Check if all degrees are equal
        if not degs:
            return False
        first_deg = degs[0]
        for d in degs[1:]:
            if d != first_deg:
                return False
        return True

    def find_bad_subset(adj_masks):
        """
        Find any subset of size >= 6 that induces a regular subgraph.
        Returns the mask of the subset, or None if no such subset exists.
        """
        # Iterate through all masks
        # Optimization: We can skip masks with < 6 bits.
        # For n=20, 2^20 is 1,048,576. This is fast enough.
        
        # To speed up, we can iterate by size? No, simple loop is fine.
        # But checking is_regular_subset is O(k).
        
        # We can pre-compute nothing, just loop.
        
        # Start from masks with at least 6 bits.
        # Simplest: loop all, check bit count.
        
        # Better: Generate combinations?
        # itertools.combinations is slow in Python for 20 choose 6.
        # Bitmask iteration is faster.
        
        for mask in range(1, 1 << n):
            if popcount(mask) < 6:
                continue
            if is_regular_subset(mask, adj_masks):
                return mask
        return None

    def flip_edge(adj_masks, u, v):
        """
        Flip the edge between u and v in the adjacency masks.
        """
        adj_masks[u] ^= (1 << v)
        adj_masks[v] ^= (1 << u)

    def random_graph(n):
        adj_masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
        return adj_masks

    # Search loop
    best_n = 5
    best_edges = []
    
    # Try to find a solution for n=20
    # We will do a local search.
    
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        adj_masks = random_graph(n)
        
        # Local search
        steps = 0
        max_steps = 5000  # Limit steps per restart to avoid infinite loops on bad random inits
        
        while steps < max_steps:
            if time.time() > deadline:
                break
                
            bad_mask = find_bad_subset(adj_masks)
            
            if bad_mask is None:
                # We found a valid graph!
                # Construct the edge list
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj_masks[i] >> j) & 1:
                            edges.append([i, j])
                
                return {'n': n, 'edges': edges}
            
            # If we found a bad subset, try to fix it.
            # Get vertices in the bad subset
            vertices = []
            m = bad_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                vertices.append(v)
                m ^= lsb
            
            # Strategy: Flip a random edge inside the bad subset.
            # This will break the regularity of this specific subset.
            if len(vertices) >= 2:
                u = random.choice(vertices)
                v = random.choice(vertices)
                if u != v:
                    flip_edge(adj_masks, u, v)
            else:
                # Should not happen for size >= 6
                break
                
            steps += 1
            
            # Occasionally, if we seem stuck (e.g., many steps without finding solution),
            # we might want to do a larger perturbation.
            # But simple hill climbing on "find one, fix one" usually works for these small n.
            # If we cycle, we'll hit max_steps and restart.

    # If we didn't find n=20, return the best we have (or just n=5 as fallback)
    # But we should try to return a valid graph if possible.
    # The empty graph on 5 is always valid.
    return {'n': 5, 'edges': []}
