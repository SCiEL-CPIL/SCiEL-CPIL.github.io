import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    n = 18
    # Initialize with a random K4-free graph or a structured one
    # Let's start with a random graph and remove edges to make it K4-free
    # Or start with K6,6,6 which is K4-free and has a known score.
    
    adj = [0] * n
    
    # Helper to check K4-freeness
    def is_k4_free():
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    # Check if any edge exists within common
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        if adj[w] & common:
                            return False
                        c -= lsb
        return True

    # Helper to compute max triangle-free induced subgraph size using B&B
    def max_tf_size():
        # We want the largest k such that there exists a subset of size k with no triangles.
        # We try k from 18 down to 1.
        
        # Precompute triangles? No, check on the fly is better for B&B pruning.
        
        def can_add(v, current_mask):
            # Check if adding v to current_mask creates a triangle
            # Triangle (v, u, w) where u, w in current_mask
            # This happens if there is an edge (u, w) in current_mask such that
            # (u, v) and (w, v) are edges.
            # Equivalent: For all u in current_mask, (adj[v] & adj[u] & current_mask) == 0
            c = current_mask
            while c:
                lsb = c & -c
                u = lsb.bit_length() - 1
                if adj[v] & adj[u] & current_mask:
                    return False
                c -= lsb
            return True

        def search(size, current_mask, available_mask, count):
            # current_mask: bitmask of selected vertices
            # available_mask: bitmask of vertices that can still be selected (subset of original)
            # count: number of vertices selected so far
            # We want to reach 'size'
            
            if count == size:
                return True
            
            # Pruning: if not enough available vertices
            if count + bin(available_mask).count('1') < size:
                return False
            
            # Iterate through available vertices
            a = available_mask
            while a:
                lsb = a & -a
                v = lsb.bit_length() - 1
                a -= lsb
                
                # Try adding v
                if can_add(v, current_mask):
                    if search(size, current_mask | (1 << v), available_mask & ~(1 << v), count + 1):
                        return True
            return False

        # Try sizes from 18 down to 1
        for k in range(18, 0, -1):
            # To make B&B more effective, we can order vertices or use a better initial available set.
            # Here we just try all subsets of size k.
            # The search space is C(18, k).
            # For k=18, it's 1. For k=17, it's 18. For k=10, it's 43758.
            # The pruning in `search` should cut this down significantly if triangles are dense.
            
            # Optimization: If k is large, the search is fast. If k is small, it might be slower?
            # Actually, finding a TF set of size 1 is trivial. Finding size 10 is harder.
            # But we stop as soon as we find one.
            
            # We need to be careful with recursion depth and time.
            # If this takes too long, we might timeout.
            # Let's add a time check inside the search?
            # Python recursion overhead is high.
            
            # Alternative: Use a greedy upper bound?
            # No, we need the exact max.
            
            # Let's try the search.
            # To speed up, we can shuffle available vertices? No, determinism is better for reproducibility?
            # Randomness in search order might help avoid bad local minima in B&B?
            # Let's stick to simple iteration first.
            
            if search(k, 0, (1 << n) - 1, 0):
                return k
        return 0

    # Initial Graph: Random
    # Generate a random graph and remove edges to ensure K4-free
    # Or just start with K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    
    current_score = max_tf_size()
    best_score = current_score
    best_adj = adj[:]
    
    # SA Parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Pick random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        # Flip
        mask = 1 << v
        if adj[u] & mask:
            adj[u] &= ~mask
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= mask
            adj[v] |= 1 << u
            
        # Check K4
        if not is_k4_free():
            # Revert
            if adj[u] & mask:
                adj[u] &= ~mask
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= mask
                adj[v] |= 1 << u
            continue
            
        # Compute score
        new_score = max_tf_size()
        
        if new_score < best_score:
            best_score = new_score
            best_adj = adj[:]
            
        # Accept/Reject
        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            current_score = new_score
        else:
            # Revert
            if adj[u] & mask:
                adj[u] &= ~mask
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= mask
                adj[v] |= 1 << u
            # current_score remains the same as before the flip (conceptually)
            # But we didn't update it in the accepted branch if we rejected?
            # Wait, if we reject, we revert adj. The score of the reverted adj is the old current_score.
            # So we don't need to recompute.
            
        T *= cooling
        
    # Return edges of best_adj
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
