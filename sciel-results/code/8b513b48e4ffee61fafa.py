import time
import random
import numpy as np
from itertools import combinations

def _edges_from_adj_masks(adj_masks, n):
    edges = []
    for i in range(n):
        m = adj_masks[i]
        for j in range(i + 1, n):
            if (m >> j) & 1:
                edges.append([i, j])
    return edges

def _is_valid(adj_masks, n):
    """
    Returns True if no induced subgraph of size >= 6 is regular.
    Uses bitmasks for adjacency.
    """
    # Precompute list of vertices for each mask? No, too slow/memory heavy for 2^n.
    # Instead, iterate masks.
    
    # Optimization: 
    # A regular subgraph of size k has sum of degrees = k * d.
    # Sum of degrees is even, so k*d must be even.
    # If k is odd, d must be even.
    
    # We can check subsets of size 6 first, then 7, etc.
    # For n=21, C(21, 6) = 54264. C(21, 7) = 116280. Total is manageable.
    # For n=22, C(22, 6) = 74613.
    
    # To speed up checking a specific subset:
    # 1. Compute degrees of all vertices in the subset.
    # 2. Check if all equal.
    
    # We can iterate by size k from 6 to n.
    for k in range(6, n + 1):
        for comb in combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            
            # Check regularity
            # Get first vertex
            lsb = mask & -mask
            v0 = lsb.bit_length() - 1
            d0 = (adj_masks[v0] & mask).bit_count()
            
            # Parity check
            if (k * d0) % 2 != 0:
                continue
                
            # Check all others
            is_reg = True
            m = mask
            m ^= lsb
            while m:
                lsb_v = m & -m
                v = lsb_v.bit_length() - 1
                if (adj_masks[v] & mask).bit_count() != d0:
                    is_reg = False
                    break
                m ^= lsb_v
            
            if is_reg:
                return False
    return True

def _find_violation_mask(adj_masks, n):
    """
    Returns the mask of the first regular induced subgraph of size >= 6 found, or 0.
    """
    for k in range(6, n + 1):
        for comb in combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            
            lsb = mask & -mask
            v0 = lsb.bit_length() - 1
            d0 = (adj_masks[v0] & mask).bit_count()
            
            if (k * d0) % 2 != 0:
                continue
                
            is_reg = True
            m = mask
            m ^= lsb
            while m:
                lsb_v = m & -m
                v = lsb_v.bit_length() - 1
                if (adj_masks[v] & mask).bit_count() != d0:
                    is_reg = False
                    break
                m ^= lsb_v
            
            if is_reg:
                return mask
    return 0

def _count_violations(adj_masks, n, limit=10000):
    """
    Counts the number of regular induced subgraphs of size >= 6.
    Stops early if count exceeds limit.
    """
    count = 0
    for k in range(6, n + 1):
        for comb in combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            
            lsb = mask & -mask
            v0 = lsb.bit_length() - 1
            d0 = (adj_masks[v0] & mask).bit_count()
            
            if (k * d0) % 2 != 0:
                continue
                
            is_reg = True
            m = mask
            m ^= lsb
            while m:
                lsb_v = m & -m
                v = lsb_v.bit_length() - 1
                if (adj_masks[v] & mask).bit_count() != d0:
                    is_reg = False
                    break
                m ^= lsb_v
            
            if is_reg:
                count += 1
                if count > limit:
                    return count
    return count

def _random_graph(n, p=0.5):
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj

def _local_search(n, time_limit, start_adj=None):
    """
    Local search: 
    1. Start with a graph (or random).
    2. Find a violation.
    3. If no violation, return graph.
    4. If violation, try flipping edges to break it. 
       Specifically, pick an edge inside the violation or incident to it.
       Actually, flipping an edge (u,v) changes the degree of u and v in any subset containing them.
       To break regularity of a set S, we can flip an edge (u,v) where u,v in S, or one in, one out.
       If we flip (u,v) with u,v in S: deg(u) and deg(v) change by +/-1. If they were equal, they become unequal (unless we flip both ways? No, single edge flip).
       Wait, if we flip (u,v) inside S: 
       If edge existed: remove it. deg(u)-=1, deg(v)-=1. Both decrease. Regularity might be preserved if all degrees decreased? No, only u and v change. If d(u)=d(v)=d, then d(u)=d-1, d(v)=d-1. Others are d. So now we have two vertices with d-1 and k-2 with d. Not regular (unless d-1=d which is impossible).
       So flipping any edge inside a regular induced subgraph of size k >= 3 will break its regularity.
       
       Strategy:
       1. Find violation S.
       2. Pick a random edge (u,v) with u,v in S.
       3. Flip it.
       4. Repeat.
       
       This is a greedy hill-climbing on "number of violations" or just "first violation".
       Since flipping an edge inside S breaks S, we might create new violations.
       We should track the count or just restart if stuck.
    """
    start_time = time.time()
    
    if start_adj is None:
        adj = _random_graph(n)
    else:
        adj = start_adj[:]
        
    best_adj = adj[:]
    best_count = _count_violations(adj, n, limit=100)
    
    # If already valid, return
    if best_count == 0:
        return adj
    
    # Local search loop
    # We want to reduce the number of violations.
    # Since checking count is expensive, we can use a "first violation" heuristic with restarts.
    # Or, we can try to fix the first violation found, and hope it doesn't create many new ones.
    
    # Let's do a simple random walk with acceptance if it reduces violations (sampled).
    # But sampling violations is expensive.
    
    # Alternative: 
    # 1. Find first violation S.
    # 2. Generate all edges inside S.
    # 3. Pick one at random, flip it.
    # 4. Check if the graph is valid (expensive).
    #    If yes, return.
    #    If no, repeat.
    # This is "First Fit" style.
    
    # To avoid getting stuck in a cycle, we can keep a history or just random restart.
    
    for i in range(10000):
        if time.time() > time_limit:
            break
            
        viol_mask = _find_violation_mask(adj, n)
        if viol_mask == 0:
            return adj
            
        # Get vertices in violation
        verts = []
        m = viol_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
            
        # Pick a random edge inside this set
        if len(verts) < 2:
            # Should not happen for k>=6
            break
            
        u, v = random.sample(verts, 2)
        
        # Flip edge (u, v)
        if (adj[u] >> v) & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Optional: Check if we improved?
        # Counting is too slow. Let's just continue.
        # If we get stuck, the loop will just wander.
        # To help, we can occasionally restart with a new random graph if no progress in X steps?
        # But we don't know progress without counting.
        
        # Let's add a simple restart mechanism:
        # If we've done 100 steps without finding a valid graph, and we haven't found one yet,
        # maybe try a new random start?
        # But we don't know if we are "making progress".
        
        # Let's stick to the simple heuristic. It works well for SAT-like problems sometimes.
        
    return best_adj # Return the last state, even if invalid? 
                     # The caller will check validity.

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_n = 0
    best_adj = []
    
    # Try n=21 first, then 22, then 20
    # n=21 is the target for improvement (score 2.1 vs 2.0)
    # n=22 is the moonshot.
    
    for target_n in [21, 22, 20]:
        if time.time() > deadline - 10:
            break
            
        n = target_n
        found = False
        
        # Multiple restarts of local search
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize random graph
            # Bias towards p=0.5? Or maybe slightly different?
            # Random regular graphs? No, induced subgraphs are the issue.
            # Just random p=0.5.
            
            adj = _random_graph(n)
            
            # Quick check if valid?
            # _is_valid is expensive.
            # Let's just run local search.
            
            res_adj = _local_search(n, deadline - 5, start_adj=adj)
            
            if _is_valid(res_adj, n):
                best_n = n
                best_adj = res_adj
                found = True
                break
                
        if found:
            break
            
    # Fallback to smaller n if 21/22 failed
    if best_n < 21:
        for target_n in [20, 19, 18]:
            if time.time() > deadline - 2:
                break
            n = target_n
            found = False
            for r in range(20):
                if time.time() > deadline - 2:
                    break
                adj = _random_graph(n)
                if _is_valid(adj, n):
                    best_n = n
                    best_adj = adj
                    found = True
                    break
            if found:
                break
                
    if best_n == 0:
        # Fallback to a small valid graph
        # 5 vertices is always valid (no subset of size 6)
        best_n = 5
        best_adj = [0] * 5
        
    edges = _edges_from_adj_masks(best_adj, best_n)
    return {'n': best_n, 'edges': edges}

# For testing
if __name__ == "__main__":
    res = construct()
    print(f"Found graph with n={res['n']}")
    # Verify
    n = res['n']
    adj = [0] * n
    for u, v in res['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    print(f"Valid: {_is_valid(adj, n)}")
