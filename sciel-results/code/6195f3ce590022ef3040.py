"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_regular_6(adj_masks, verts, mask):
    """Check if 6 vertices induce a regular subgraph."""
    d0 = bin(adj_masks[verts[0]] & mask).count('1')
    for i in range(1, 6):
        if bin(adj_masks[verts[i]] & mask).count('1') != d0:
            return False
    return True

def _is_regular_k(adj_masks, verts, mask, k):
    """Check if k vertices induce a regular subgraph."""
    d0 = bin(adj_masks[verts[0]] & mask).count('1')
    for i in range(1, k):
        if bin(adj_masks[verts[i]] & mask).count('1') != d0:
            return False
    return True

def _count_bad_6(adj_masks, subsets_6):
    """Count number of 6-subsets that induce a regular graph."""
    bad = 0
    for verts, mask in subsets_6:
        if _is_regular_6(adj_masks, verts, mask):
            bad += 1
    return bad

def _count_bad_k(adj_masks, subsets_k, k):
    """Count number of k-subsets that induce a regular graph."""
    bad = 0
    for verts, mask in subsets_k:
        if _is_regular_k(adj_masks, verts, mask, k):
            bad += 1
    return bad

def _full_check(adj_masks, n):
    """Check all subsets of size >= 6 for regularity. Returns True if valid."""
    for k in range(6, n + 1):
        for combo in combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            if _is_regular_k(adj_masks, list(combo), mask, k):
                return False
    return True

def _construct_graph(n, deadline):
    """Construct a graph on n vertices with no regular induced subgraph on 6+ vertices."""
    # Precompute 6-subsets
    subsets_6 = []
    for combo in combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        subsets_6.append((list(combo), mask))
    
    # Precompute which 6-subsets contain a given pair (u,v)
    # subsets_6_containing[u][v] = list of indices into subsets_6
    # For n up to 20, this is manageable
    subsets_6_containing = {}
    for i, (verts, mask) in enumerate(subsets_6):
        for a in range(n):
            for b in range(a + 1, n):
                if mask & (1 << a) and mask & (1 << b):
                    key = (a, b)
                    if key not in subsets_6_containing:
                        subsets_6_containing[key] = []
                    subsets_6_containing[key].append(i)
    
    best_adj_masks = None
    best_score = float('inf')
    
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize random graph
        adj_masks = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
        
        current_bad_6 = _count_bad_6(adj_masks, subsets_6)
        
        # Local search to minimize 6-regular subgraphs
        stuck_counter = 0
        max_steps = 2000
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if current_bad_6 == 0:
                break
            
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Only subsets containing both u and v are affected
            key = (min(u, v), max(u, v))
            affected_indices = subsets_6_containing.get(key, [])
            
            # Compute change in bad count
            delta = 0
            for idx in affected_indices:
                verts, mask = subsets_6[idx]
                was_bad = _is_regular_6(adj_masks, verts, mask)
                
                # Flip edge
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
                
                now_bad = _is_regular_6(adj_masks, verts, mask)
                
                # Revert for next iteration
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
                
                if was_bad and not now_bad:
                    delta -= 1
                elif not was_bad and now_bad:
                    delta += 1
            
            new_bad = current_bad_6 + delta
            
            if new_bad < current_bad_6:
                # Accept: actually flip the edge
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
                current_bad_6 = new_bad
                stuck_counter = 0
            else:
                stuck_counter += 1
                if stuck_counter > 100:
                    break
        
        if current_bad_6 == 0:
            # Check all subsets >= 7
            valid = _full_check(adj_masks, n)
            if valid:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj_masks[u] & (1 << v):
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
        else:
            if current_bad_6 < best_score:
                best_score = current_bad_6
                best_adj_masks = adj_masks[:]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Try n from 20 down to 6
    for n in range(20, 5, -1):
        if time.time() > deadline - 5:
            break
        
        result = _construct_graph(n, deadline)
        if result is not None:
            return result
    
    # Fallback: return a small valid graph
    # Quick search for n=6
    for _ in range(5000):
        if time.time() > deadline - 1:
            break
        adj_masks = [0] * 6
        for u in range(6):
            for v in range(u + 1, 6):
                if random.random() < 0.5:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
        
        # Check if the whole graph is regular
        degs = [bin(adj_masks[i]).count('1') for i in range(6)]
        if len(set(degs)) != 1:
            edges = []
            for u in range(6):
                for v in range(u + 1, 6):
                    if adj_masks[u] & (1 << v):
                        edges.append([u, v])
            return {'n': 6, 'edges': edges}
    
    return {'n': 5, 'edges': []}
# EVOLVE-BLOCK-END
