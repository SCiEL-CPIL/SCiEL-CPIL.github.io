import time
import random
import math
from collections import deque

def construct():
    start_time = time.time()
    DEADLINE = 58.0

    N = 256
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency as bitsets
    adj = [0] * N
    forbidden = [False] * N

    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            forbidden[i] = True
        else:
            for sq in squares:
                u = sq - v
                if 1 <= u <= N:
                    j = u - 1
                    if not forbidden[j]:
                        adj[i] |= (1 << j)

    candidates = [i for i in range(N) if not forbidden[i]]
    cand_mask = 0
    for i in candidates:
        cand_mask |= (1 << i)

    def greedy_init(seed=None):
        if seed is not None:
            random.seed(seed)
        available_mask = cand_mask
        selected_mask = 0
        while available_mask:
            # Pick vertex with minimum degree in available set
            # Use random tie-breaking for diversity
            best_v = -1
            min_deg = N + 1
            m = available_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                deg = (adj[v] & available_mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
                elif deg == min_deg:
                    # Random tie-break
                    if random.random() < 0.5:
                        best_v = v
            if best_v == -1:
                break
            selected_mask |= (1 << best_v)
            available_mask &= ~((1 << best_v) | (adj[best_v] & available_mask))
        return selected_mask

    def local_improve(mask):
        improved = True
        while improved:
            improved = False
            not_in_mask = cand_mask & ~mask
            m = not_in_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if (adj[v] & mask) == 0:
                    mask |= lsb
                    improved = True
                    break
        return mask

    def greedy_color_bound(P):
        """Compute an upper bound on independent set size using greedy coloring."""
        # Color the conflict graph induced by P
        # Each color class is a set of mutually non-adjacent vertices (independent set in complement)
        # Wait, that's not right. Let me think again.
        # 
        # For maximum independent set in G, we can use:
        # alpha(G) <= chi(G) where chi is chromatic number? No, that's wrong.
        # 
        # Actually, a better approach: use the fact that if we can color the graph with c colors,
        # then the independence number is at most n - (minimum vertex cover) ... this is getting complex.
        # 
        # Simpler bound: use a greedy algorithm to find a maximal independent set in the complement,
        # but that's the same as finding a clique in G.
        # 
        # Let me use a different approach: recursive bound with memoization on small subgraphs.
        # For now, use the simple bound: |P|
        return P.bit_count()

    def branch_and_bound_mis(P, current_set, best_size, best_set, deadline):
        if time.time() > deadline:
            return best_size, best_set
        
        # Bound: |current_set| + |P|
        if current_set.bit_count() + P.bit_count() <= best_size:
            return best_size, best_set
        
        if P == 0:
            if current_set.bit_count() > best_size:
                best_size = current_set.bit_count()
                best_set = current_set
            return best_size, best_set
        
        # Pick a vertex to branch on
        # Choose vertex with maximum degree in P for better pruning
        m = P
        best_v = -1
        max_deg = -1
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            deg = (adj[v] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
        
        if best_v == -1:
            return best_size, best_set
        
        # Branch 1: Include best_v
        new_P1 = P & ~((1 << best_v) | adj[best_v])
        new_set1 = current_set | (1 << best_v)
        best_size, best_set = branch_and_bound_mis(new_P1, new_set1, best_size, best_set, deadline)
        
        # Branch 2: Exclude best_v
        new_P2 = P & ~(1 << best_v)
        best_size, best_set = branch_and_bound_mis(new_P2, current_set, best_size, best_set, deadline)
        
        return best_size, best_set

    # Main: Use multiple restarts with B&B
    best_size = 0
    best_set = 0
    
    # Initial greedy solution
    mask = greedy_init()
    mask = local_improve(mask)
    best_size = mask.bit_count()
    best_set = mask
    
    # Try B&B with time limit
    bb_deadline = start_time + 15.0
    bb_size, bb_set = branch_and_bound_mis(cand_mask, 0, best_size, best_set, bb_deadline)
    if bb_size > best_size:
        best_size = bb_size
        best_set = bb_set
    
    # Use remaining time for more restarts and B&B
    remaining_deadline = start_time + DEADLINE - 2.0
    
    # Try multiple random restarts
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() >= remaining_deadline:
            break
        
        # Random greedy initialization
        mask = greedy_init(seed=random.randint(0, 2**31))
        mask = local_improve(mask)
        
        if mask.bit_count() > best_size:
            best_size = mask.bit_count()
            best_set = mask
        
        # Try B&B from this starting point with short time limit
        restart_deadline = min(time.time() + 2.0, remaining_deadline)
        if time.time() < restart_deadline:
            bb_size, bb_set = branch_and_bound_mis(cand_mask, 0, best_size, best_set, restart_deadline)
            if bb_size > best_size:
                best_size = bb_size
                best_set = bb_set
    
    # Final B&B with remaining time
    if time.time() < remaining_deadline:
        bb_size, bb_set = branch_and_bound_mis(cand_mask, 0, best_size, best_set, remaining_deadline)
        if bb_size > best_size:
            best_size = bb_size
            best_set = bb_set
    
    # Final local search polish
    if time.time() < remaining_deadline:
        final_mask = local_improve(best_set)
        if final_mask.bit_count() > best_size:
            best_size = final_mask.bit_count()
            best_set = final_mask
    
    # Convert best_set to list of values (1-based)
    result = []
    m = best_set
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v + 1)
        m ^= lsb
        
    return sorted(result)

# EVOLVE-BLOCK-END
