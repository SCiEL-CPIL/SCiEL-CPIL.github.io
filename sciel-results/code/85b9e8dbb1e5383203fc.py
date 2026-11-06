import time
import math
import random
from itertools import combinations

_N = 128
_DEADLINE = 85.0  # Safety margin for 90s kill
_CAND = list(range(1, _N + 1))
_SEED = 42

# Global precomputed data
_GCD = None
_BAD_MASK = None  # BAD_MASK[u][v] is a bitmask of w such that {u,v,w} is a bad triple
_ADJ = None  # Adjacency masks for the conflict graph

def _ensure_pre():
    global _GCD, _BAD_MASK, _ADJ
    if _GCD is not None:
        return
    
    # 1. GCD Matrix
    _GCD = [[0] * (_N + 1) for _ in range(_N + 1)]
    for i in range(1, _N + 1):
        for j in range(i, _N + 1):
            g = math.gcd(i, j)
            _GCD[i][j] = g
            _GCD[j][i] = g

    # 2. Bad Triple Masks
    # BAD_MASK[u][v] has bit w set if gcd(u,v) == gcd(u,w) == gcd(v,w) and w != u, w != v
    _BAD_MASK = [[0] * (_N + 1) for _ in range(_N + 1)]
    
    for u in range(1, _N + 1):
        for v in range(u + 1, _N + 1):
            g_uv = _GCD[u][v]
            mask = 0
            for w in range(1, _N + 1):
                if w == u or w == v:
                    continue
                if _GCD[u][w] == g_uv and _GCD[v][w] == g_uv:
                    mask |= (1 << (w - 1))
            _BAD_MASK[u][v] = mask
            _BAD_MASK[v][u] = mask

    # 3. Conflict graph adjacency
    # Two vertices u,v are adjacent if there exists w such that {u,v,w} is a bad triple
    # i.e., BAD_MASK[u][v] is non-empty
    _ADJ = [0] * (_N + 1)
    for u in range(1, _N + 1):
        for v in range(u + 1, _N + 1):
            if _BAD_MASK[u][v] != 0:
                _ADJ[u] |= (1 << (v - 1))
                _ADJ[v] |= (1 << (u - 1))

def _popcount(x):
    return x.bit_count()

def _max_independent_set_bnb(vertices_mask, deadline, best_size_ref):
    """
    Branch and bound for Maximum Independent Set.
    vertices_mask: bitmask of available vertices
    Returns (best_size, best_set_mask)
    """
    best_size = [best_size_ref[0]]
    best_set = [best_size_ref[1]]
    
    def dfs(candidates, current_set, current_size):
        if time.time() > deadline:
            return
        
        if candidates == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = current_set
            return
        
        # Upper bound: current_size + number of remaining candidates
        upper_bound = current_size + _popcount(candidates)
        if upper_bound <= best_size[0]:
            return
        
        # Greedy coloring upper bound (simplified: just use candidate count)
        # Pick a vertex with maximum degree in the induced subgraph
        # For efficiency, pick the first available vertex with highest degree
        best_v = -1
        best_deg = -1
        temp = candidates
        while temp:
            # Get lowest set bit
            lsb = temp & (-temp)
            v = lsb.bit_length()  # 1-indexed
            # Degree of v in candidates (excluding v itself)
            deg = _popcount(_ADJ[v] & candidates & ~(1 << (v - 1)))
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp &= temp - 1
        
        if best_v == -1:
            return
        
        # Branch 1: include best_v
        new_candidates = candidates & ~_ADJ[best_v] & ~(1 << (best_v - 1))
        new_set = current_set | (1 << (best_v - 1))
        dfs(new_candidates, new_set, current_size + 1)
        
        # Branch 2: exclude best_v
        new_candidates = candidates & ~(1 << (best_v - 1))
        dfs(new_candidates, current_set, current_size)
    
    dfs(vertices_mask, 0, 0)
    return best_size[0], best_set[0]

def _greedy_coloring_upper_bound(candidates):
    """
    Compute a greedy coloring upper bound for the independence number.
    This gives a tighter bound than just counting vertices.
    """
    # Simple greedy: repeatedly pick a vertex, color it, remove it and its neighbors
    # The number of colors needed is an upper bound on the independence number
    # Actually, for MIS, we want a lower bound on the chromatic number of the complement
    # Let's use a simpler approach: just return the count for now
    return _popcount(candidates)

def _improved_bnb(vertices_mask, deadline, initial_size):
    """
    Improved branch and bound with better heuristics.
    """
    best_size = [initial_size]
    best_set = [0]
    
    # First, find a good initial solution using greedy
    def greedy_mis(candidates):
        result = 0
        size = 0
        temp = candidates
        while temp:
            # Pick vertex with minimum degree in remaining candidates
            best_v = -1
            best_deg = float('inf')
            t = temp
            while t:
                lsb = t & (-t)
                v = lsb.bit_length()
                deg = _popcount(_ADJ[v] & temp & ~(1 << (v - 1)))
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                t &= t - 1
            if best_v == -1:
                break
            result |= (1 << (best_v - 1))
            size += 1
            temp = temp & ~_ADJ[best_v] & ~(1 << (best_v - 1))
        return result, size
    
    greedy_result, greedy_size = greedy_mis(vertices_mask)
    if greedy_size > best_size[0]:
        best_size[0] = greedy_size
        best_set[0] = greedy_result
    
    def dfs(candidates, current_set, current_size):
        if time.time() > deadline:
            return
        
        if candidates == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = current_set
            return
        
        # Upper bound
        upper_bound = current_size + _popcount(candidates)
        if upper_bound <= best_size[0]:
            return
        
        # Pick vertex with maximum degree
        best_v = -1
        best_deg = -1
        temp = candidates
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length()
            deg = _popcount(_ADJ[v] & candidates & ~(1 << (v - 1)))
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp &= temp - 1
        
        if best_v == -1:
            return
        
        # Branch 1: include best_v
        new_candidates = candidates & ~_ADJ[best_v] & ~(1 << (best_v - 1))
        new_set = current_set | (1 << (best_v - 1))
        dfs(new_candidates, new_set, current_size + 1)
        
        # Branch 2: exclude best_v
        new_candidates = candidates & ~(1 << (best_v - 1))
        dfs(new_candidates, current_set, current_size)
    
    dfs(vertices_mask, 0, 0)
    return best_size[0], best_set[0]

def _local_search_polish(mask, deadline):
    """
    Try to add any element that doesn't create a conflict.
    """
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, _N + 1):
            if not ((mask >> (x - 1)) & 1):
                # Check if x conflicts with any element in mask
                conflicts = _ADJ[x] & mask
                if conflicts == 0:
                    mask |= (1 << (x - 1))
                    changed = True
    return mask

def construct():
    _ensure_pre()
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    all_mask = (1 << _N) - 1
    
    # Get initial size from a greedy approach
    def greedy_initial():
        result = 0
        temp = all_mask
        while temp:
            # Pick vertex with minimum degree in remaining candidates
            best_v = -1
            best_deg = float('inf')
            t = temp
            while t:
                lsb = t & (-t)
                v = lsb.bit_length()
                deg = _popcount(_ADJ[v] & temp & ~(1 << (v - 1)))
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                t &= t - 1
            if best_v == -1:
                break
            result |= (1 << (best_v - 1))
            temp = temp & ~_ADJ[best_v] & ~(1 << (best_v - 1))
        return result
    
    initial_mask = greedy_initial()
    initial_size = _popcount(initial_mask)
    
    # Run branch and bound
    bnb_size, bnb_mask = _improved_bnb(all_mask, deadline - 10, initial_size)
    
    # Polish the result
    polished_mask = _local_search_polish(bnb_mask, deadline - 2)
    polished_size = _popcount(polished_mask)
    
    # Use the best result
    if polished_size >= bnb_size:
        best_mask = polished_mask
    else:
        best_mask = bnb_mask
    
    # Convert mask to list
    result = [i for i in range(1, _N + 1) if (best_mask >> (i - 1)) & 1]
    
    # Fallback if empty (should not happen)
    if not result:
        result = [2]
        
    return result

if __name__ == "__main__":
    # Self-test
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    _ensure_pre()
    P = _GCD
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            for k in range(j+1, len(res)):
                a, b, c = res[i], res[j], res[k]
                if P[a][b] == P[a][c] == P[b][c]:
                    valid = False
    print(f"Valid: {valid}")
