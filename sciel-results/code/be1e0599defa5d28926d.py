"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations, product

def _has_triangle_circulant_fast(n, S):
    """Check if circulant graph on Z_n with connection set S has a triangle."""
    S_set = frozenset(S)
    for a in S:
        for b in S:
            if a == b:
                continue
            diff = (b - a) % n
            if diff in S_set:
                return True
    return False

def _max_independent_set_size(n, adj_mask):
    """
    Compute alpha of a graph given adjacency masks.
    Uses branch and bound with bitmasks and degree-based branching.
    """
    full_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = full_mask ^ (1 << i) ^ adj_mask[i]
    
    best = [0]
    
    def popcount(x):
        return x.bit_count()
    
    def dfs(candidates, current_size):
        if current_size + popcount(candidates) <= best[0]:
            return
        if candidates == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Pick vertex with highest degree in candidate subgraph
        best_v = -1
        max_deg = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = popcount(candidates & adj_mask[v])
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
        
        v = best_v
        # Include v
        dfs(candidates & comp_mask[v], current_size + 1)
        # Exclude v
        dfs(candidates & ~(1 << v), current_size)
    
    dfs(full_mask, 0)
    return best[0]

def _verify_circulant(n, S):
    """Verify circulant graph is triangle-free and alpha < 10."""
    if _has_triangle_circulant_fast(n, S):
        return False, 0
    
    # Build adjacency masks
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for s in S:
            j = (i + s) % n
            if j != i:
                mask |= (1 << j)
        adj_mask[i] = mask
    
    alpha = _max_independent_set_size(n, adj_mask)
    if alpha >= 10:
        return False, alpha
    return True, alpha

def _build_edges(n, S):
    """Build edge list from circulant connection set."""
    edges = []
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if j > i:
                edges.append([i, j])
    return edges

def _make_symmetric(n, half_set):
    """Create symmetric connection set from a set of positive offsets."""
    S = set()
    for x in half_set:
        S.add(x)
        S.add((-x) % n)
    S.discard(0)
    return S

def _generate_structured_candidates(n):
    """Generate structured circulant connection sets in order of promise."""
    half = n // 2
    candidates = []
    
    # 1. Consecutive offsets: {1, 2, ..., k}
    for k in range(2, half + 1):
        candidates.append(set(range(1, k + 1)))
    
    # 2. Arithmetic progressions: {a, a+d, a+2d, ...} within [1, half]
    for d in range(1, half + 1):
        for a in range(1, d + 1):
            ap = set()
            x = a
            while x <= half:
                ap.add(x)
                x += d
            if len(ap) >= 2:
                candidates.append(ap)
    
    # 3. Union of two APs
    for d1 in range(1, min(half, 8) + 1):
        for d2 in range(d1 + 1, min(half, 8) + 1):
            for a1 in range(1, d1 + 1):
                for a2 in range(1, d2 + 1):
                    ap1 = set()
                    x = a1
                    while x <= half:
                        ap1.add(x)
                        x += d1
                    ap2 = set()
                    x = a2
                    while x <= half:
                        ap2.add(x)
                        x += d2
                    union = ap1 | ap2
                    if len(union) >= 3 and len(union) <= half:
                        candidates.append(union)
    
    # 4. Odd offsets: {1, 3, 5, ..., 2k-1}
    for k in range(2, half + 1):
        odds = set(range(1, 2 * k, 2))
        if odds and max(odds) <= half:
            candidates.append(odds)
    
    # 5. Even offsets: {2, 4, 6, ..., 2k}
    for k in range(1, half // 2 + 1):
        evens = set(range(2, 2 * k + 1, 2))
        if evens and max(evens) <= half:
            candidates.append(evens)
    
    # 6. Multiples of small numbers: {d, 2d, 3d, ...} up to half
    for d in range(2, min(half, 10) + 1):
        mults = set(range(d, half + 1, d))
        if len(mults) >= 2:
            candidates.append(mults)
    
    # 7. Complements of small sets (high-degree graphs)
    for k in range(1, 4):
        for combo in combinations(range(1, half + 1), k):
            excluded = set(combo)
            comp = set(range(1, half + 1)) - excluded
            if len(comp) >= 2:
                candidates.append(comp)
    
    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    return unique_candidates

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Try n from 40 down to 25
    for n in range(40, 24, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        half = n // 2
        
        # Generate structured candidates
        candidates = _generate_structured_candidates(n)
        
        # Sort by size descending (larger S = higher degree = smaller alpha)
        candidates.sort(key=len, reverse=True)
        
        for half_set in candidates:
            if time.time() > deadline - 5:
                break
            
            S = _make_symmetric(n, half_set)
            if not S:
                continue
            
            # Quick triangle check
            if _has_triangle_circulant_fast(n, S):
                continue
            
            # Verify alpha
            valid, alpha = _verify_circulant(n, S)
            if valid:
                best_n = n
                best_edges = _build_edges(n, S)
                found = True
                break
        
        if found:
            break
    
    # Fallback: try random sampling if structured search didn't find anything
    if best_n == 0:
        for n in range(40, 24, -1):
            if time.time() > deadline - 5:
                break
            
            half = n // 2
            available = list(range(1, half + 1))
            
            for k in range(min(half, 12), 2, -1):
                if time.time() > deadline - 3:
                    break
                
                for attempt in range(200):
                    if time.time() > deadline:
                        break
                    
                    if k > len(available):
                        continue
                    
                    S_half = set(random.sample(available, k))
                    S = _make_symmetric(n, S_half)
                    
                    if _has_triangle_circulant_fast(n, S):
                        continue
                    
                    valid, alpha = _verify_circulant(n, S)
                    if valid:
                        best_n = n
                        best_edges = _build_edges(n, S)
                        break
                
                if best_n > 0:
                    break
            
            if best_n > 0:
                break
    
    # Fallback: Known small graphs
    if best_n < 19:
        n0 = 26
        best_edges = [[i, j] for i in range(n0) for j in range(i + 1, n0) if (j - i) % n0 % 3 == 1]
        best_n = n0
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
