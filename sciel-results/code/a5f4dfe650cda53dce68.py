import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 4 and g == 10:
        # Try to find a circulant graph C_n({s1, s2}) with girth >= 10
        n_target = 320
        result = _search_circulant_optimized(n_target, k, g, deadline)
        if result is not None:
            return result
        
        # Fallback to LUW construction (n=8192)
        return _luw_construct(k, g)

    # General fallback
    return _luw_construct(k, g)

def _search_circulant_optimized(n, k, g, deadline):
    """
    Systematically search for a circulant graph C_n({s1, s2}) with girth >= g.
    Uses a fast vectorized girth check.
    """
    # For k=4, steps are s1, s2.
    # We need s1 != s2, s1 != n-s2, 2*s1 != 0, 2*s2 != 0.
    # We can assume 1 <= s1 < s2 <= n//2.
    # Also s1 != n//2 and s2 != n//2.
    
    max_s = n // 2
    found = False
    best_edges = None
    
    # Precompute direction offsets
    # dirs: 0:+s1, 1:-s1, 2:+s2, 3:-s2
    
    # Iterate over possible s1
    # Try s1 values in order; smaller s1 often works better
    for s1 in range(1, max_s + 1):
        if 2 * s1 % n == 0:
            continue
        for s2 in range(s1 + 1, max_s + 1):
            if 2 * s2 % n == 0:
                continue
            if s1 + s2 == n: # s1 = -s2 mod n
                continue
            
            # Quick 4-cycle pre-check: a 4-cycle exists if there are a,b in {±s1,±s2}
            # with a+b=0 mod n where a and b are from different step pairs
            # i.e., s1+s2=0 (already checked), s1-s2=0 (s1=s2, impossible), 
            # -s1+s2=0 (s1=s2, impossible), -s1-s2=0 (s1+s2=0, checked)
            # Actually 4-cycles in C_n({s1,s2}): walk +s1, +s2, -s1, -s2 returns to start.
            # That's always a 4-cycle! Wait no, that's backtracking.
            # Non-backtracking 4-cycle: +s1, +s2, +s1, -s2? No...
            # A 4-cycle is v, v+s1, v+s1+s2, v+s2, v. The edges are (v,v+s1), (v+s1,v+s1+s2), (v+s1+s2,v+s2), (v+s2,v).
            # These are valid if s1, s2, s1, s2 are the steps used. The backtracking check:
            # step 1: +s1, step 2: +s2 (ok, not backtracking), step 3: -s2 (backtracks step 2!) 
            # So that's not a valid non-backtracking walk.
            # Non-backtracking 4-cycles: need 4 steps a,b,c,d from {±s1,±s2} with a+b+c+d=0,
            # and no consecutive pair is (x,-x). 
            # Possible: +s1, +s2, +s1, -s2? Sum = s1+s2+s1-s2 = 2s1. Need 2s1=0 mod n, already excluded.
            # +s1, +s2, -s1, +s2? Sum = s1+s2-s1+s2 = 2s2. Need 2s2=0, excluded.
            # +s1, -s2, +s1, +s2? Sum = s1-s2+s1+s2 = 2s1. Excluded.
            # So no 4-cycles if 2*s1 != 0 and 2*s2 != 0. Good, already handled.
            
            # Check girth
            if _check_girth_circulant_fast(n, s1, s2, g):
                # Found a valid circulant
                edges = _get_circulant_edges(n, [s1, s2])
                return n, edges
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return None

def _check_girth_circulant_fast(n, s1, s2, g):
    """
    Fast vectorized check if C_n({s1, s2}) has girth >= g.
    Uses non-backtracking walks with numpy arrays.
    """
    # Directions: 0:+s1, 1:-s1, 2:+s2, 3:-s2
    offsets = np.array([s1, -s1, s2, -s2], dtype=np.int64)
    opp = np.array([1, 0, 3, 2], dtype=np.int32)
    
    # visited[pos, dir] = min depth reached, -1 if unvisited
    visited = np.full((n, 4), -1, dtype=np.int32)
    
    max_depth = g - 1
    
    # Initial frontier: depth 1
    init_pos = offsets % n
    init_dir = np.array([0, 1, 2, 3], dtype=np.int32)
    visited[init_pos, init_dir] = 1
    
    frontier_pos = init_pos.copy()
    frontier_dir = init_dir.copy()
    depth = 1
    
    while depth < max_depth:
        if len(frontier_pos) == 0:
            return True
            
        new_depth = depth + 1
        
        # Expand frontier: for each (pos, prev_dir), try all next_dir except backtracking
        new_pos_list = []
        new_dir_list = []
        
        for nd in range(4):
            mask = (opp[frontier_dir] != nd)
            if not np.any(mask):
                continue
            fp = frontier_pos[mask]
            np_ = (fp + offsets[nd]) % n
            new_pos_list.append(np_)
            new_dir_list.append(np.full(len(np_), nd, dtype=np.int32))
            
        if not new_pos_list:
            return True
            
        all_new_pos = np.concatenate(new_pos_list)
        all_new_dir = np.concatenate(new_dir_list)
        
        # Check for cycles: any new_pos == 0
        zero_mask = (all_new_pos == 0)
        if np.any(zero_mask):
            if new_depth < g:
                return False
            all_new_pos = all_new_pos[~zero_mask]
            all_new_dir = all_new_dir[~zero_mask]
            
        if len(all_new_pos) == 0:
            return True
            
        # Filter by visited
        cur_visits = visited[all_new_pos, all_new_dir]
        valid_mask = (cur_visits == -1) | (cur_visits > new_depth)
        
        if not np.any(valid_mask):
            return True
            
        vp = all_new_pos[valid_mask]
        vd = all_new_dir[valid_mask]
        
        visited[vp, vd] = new_depth
        frontier_pos = vp
        frontier_dir = vd
        depth = new_depth
        
    return True

def _get_circulant_edges(n, steps):
    edges = []
    for i in range(n):
        for s in steps:
            j = (i + s) % n
            if i < j:
                edges.append((i, j))
    return edges

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for k-regular girth-g graphs.
    """
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
