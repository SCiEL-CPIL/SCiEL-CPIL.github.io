import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    def time_left():
        return deadline - time.time()
    
    def is_valid(adj_mask, n):
        # Check K4
        # For each pair (u, v) with edge, check common neighbors for an edge
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    # Common neighbors of u and v
                    common = adj_mask[u] & adj_mask[v]
                    # Clear u and v from common
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    # Check if any pair in common has an edge
                    # We can iterate through bits of common
                    c = common
                    while c:
                        # Get lowest set bit
                        lsb = c & -c
                        w_idx = lsb.bit_length() - 1
                        c ^= lsb
                        # Check if w is connected to any other in common
                        # (common & adj_mask[w_idx]) should have a bit set other than w itself?
                        # Actually, we just need to check if there is an edge between w and some x in common.
                        # Let's just check if (adj_mask[w_idx] & common) has any bits set.
                        # But wait, w is in common, so adj_mask[w_idx] & common will include w if self-loop? No, simple graph.
                        # But it will include w if we didn't remove it? We removed u,v.
                        # If there is an x in common such that w-x is an edge, then (adj_mask[w_idx] & common) != 0.
                        # However, this might count w itself if not careful? No, no self loops.
                        # But it might count w if w is in common? Yes.
                        # We need to check if there is an edge between two *distinct* vertices in common.
                        # So, for a specific w, check if (adj_mask[w_idx] & (common & ~(1 << w_idx))) != 0.
                        if (adj_mask[w_idx] & (common & ~(1 << w_idx))) != 0:
                            return False
        # Check Independent Set of size 6
        # Backtracking search
        # Heuristic: order vertices by degree (ascending) to find independent set faster?
        # Or just use a simple DFS with bitmasks.
        
        def find_independent_6(candidates_mask, size):
            if size == 0:
                return True
            if candidates_mask == 0:
                return False
            # Pruning: if popcount(candidates_mask) < size, return False
            if bin(candidates_mask).count('1') < size:
                return False
            
            # Pick a vertex. Let's pick the one with the lowest degree in the candidate set?
            # Or just the lowest index for determinism in search?
            # For speed, let's pick a vertex that minimizes the remaining candidates.
            # i.e., pick v in candidates such that (candidates & ~adj_mask[v]) is smallest.
            # This is expensive to compute for all v.
            # Let's just pick the first available vertex.
            lsb = candidates_mask & -candidates_mask
            v_idx = lsb.bit_length() - 1
            candidates_mask ^= lsb
            
            # Option 1: Include v
            # New candidates: those not adjacent to v
            new_cand = candidates_mask & ~adj_mask[v_idx]
            if find_independent_6(new_cand, size - 1):
                return True
            
            # Option 2: Exclude v
            if find_independent_6(candidates_mask, size):
                return True
            
            return False
        
        if find_independent_6((1 << n) - 1, 6):
            return False
            
        return True

    def find_violation(adj_mask, n):
        # Returns ('K4', [u,v,w,x]) or ('IND', [u,v,w,x,y,z]) or None
        # Find K4
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    common = adj_mask[u] & adj_mask[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    c = common
                    found_w = -1
                    found_x = -1
                    while c:
                        lsb = c & -c
                        w_idx = lsb.bit_length() - 1
                        c ^= lsb
                        # Check if w connected to any other in common
                        rest = common & ~(1 << w_idx)
                        if (adj_mask[w_idx] & rest) != 0:
                            # Find x
                            x_lsb = (adj_mask[w_idx] & rest) & -((adj_mask[w_idx] & rest))
                            x_idx = x_lsb.bit_length() - 1
                            return ('K4', [u, v, w_idx, x_idx])
        # Find Independent Set 6
        # Use the DFS logic but return the set
        def find_independent_6_set(candidates_mask, size, current):
            if size == 0:
                return current
            if candidates_mask == 0:
                return None
            if bin(candidates_mask).count('1') < size:
                return None
            
            lsb = candidates_mask & -candidates_mask
            v_idx = lsb.bit_length() - 1
            candidates_mask ^= lsb
            
            # Include v
            new_cand = candidates_mask & ~adj_mask[v_idx]
            res = find_independent_6_set(new_cand, size - 1, current + [v_idx])
            if res is not None:
                return res
            
            # Exclude v
            res = find_independent_6_set(candidates_mask, size, current)
            if res is not None:
                return res
            
            return None
        
        ind_set = find_independent_6_set((1 << n) - 1, 6, [])
        if ind_set is not None:
            return ('IND', ind_set)
            
        return None

    def repair(n, max_iters=5000):
        # Initialize random graph
        random.seed(42) # Deterministic for reproducibility, but we can vary seed in outer loop
        p = 0.5
        adj_mask = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            
            viol = find_violation(adj_mask, n)
            if viol is None:
                # Valid!
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj_mask[i] >> j) & 1:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            if viol[0] == 'K4':
                # Remove a random edge from the K4
                k4 = viol[1]
                # Pick random pair from the 6 pairs in K4
                pairs = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                         (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
                u, v = random.choice(pairs)
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                
            elif viol[0] == 'IND':
                # Add a random edge from the independent set
                ind = viol[1]
                # Pick random pair
                i, j = random.sample(ind, 2)
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
        return None

    # Try different n values
    # Start with 35, then 30, 25, 20
    target_ns = [35, 32, 30, 25, 20]
    
    for n in target_ns:
        if time.time() > deadline - 5:
            break
            
        # Try a few random restarts for this n
        # Allocate remaining time / remaining targets to this n
        # Simple heuristic: give 20s to n=35, 10s to n=30, etc.
        # Or just loop until time runs out?
        # Let's do 5 restarts per n, each with a small time limit.
        
        best_graph = None
        for restart in range(5):
            if time.time() > deadline - 2:
                break
            
            # Change seed for each restart
            random.seed(1000 + n * 10 + restart)
            
            # Run repair with a time check inside
            # We modify repair to check time more frequently or just rely on the outer check?
            # The repair function checks time at each iteration.
            
            res = repair(n, max_iters=2000)
            if res is not None:
                best_graph = res
                break # Found a valid graph for this n, move to next (smaller) n? 
                # No, we want MAX n. So if we found n=35, we are done.
                # If we found n=30, we should try to find n=35? 
                # The loop goes 35 -> 32 -> 30. 
                # So if we fail 35, we try 32. If we succeed at 32, we return it? 
                # But maybe we could find 35 with more time?
                # Given the tight budget, if we fail 35 in 5 restarts, it's unlikely we'll find it later.
                # So we accept the first valid graph found in the sequence of decreasing n.
                return res
                
    # Fallback to seed
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

# EVOLVE-BLOCK-END
