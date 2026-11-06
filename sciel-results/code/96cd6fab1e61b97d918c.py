"""Search for max n with 4-colouring of K_n having no monochromatic C6."""
import time
import random
import itertools

_DEADLINE = 115.0  # safety margin under 120s
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _edge_index(n, i, j):
    if i > j: i, j = j, i
    return i * n - i * (i + 1) // 2 + (j - i - 1)

# Precompute C6 patterns in K6 as 15-bit masks
# Vertices 0..5. Edges (0,1)..(4,5) mapped to bits 0..14.
# Edge (i,j) index in K6: i*6 - i*(i+1)//2 + (j-i-1)
def _get_c6_patterns():
    patterns = []
    # Generate all 60 unique cycles in K6
    # Fix 0. Permute 1..5. Divide by 2 for reversal.
    for perm in itertools.permutations([1, 2, 3, 4, 5]):
        cyc = (0, perm[0], perm[1], perm[2], perm[3], perm[4])
        # Break reversal: require perm[0] < perm[4]
        if perm[0] > perm[4]:
            continue
        
        mask = 0
        for k in range(6):
            u, v = cyc[k], cyc[(k+1)%6]
            if u > v: u, v = v, u
            idx = u * 6 - u * (u + 1) // 2 + (v - u - 1)
            mask |= (1 << idx)
        patterns.append(mask)
    return patterns

_C6_PATTERNS = _get_c6_patterns()

def _is_valid_fast(colors, n):
    """Check if coloring is valid (no monochromatic C6) using bitmask check on 6-subsets."""
    if n < 6:
        return True
    
    # Build edge lookup: (i,j) -> color
    # For speed, we can just use the list and index formula
    # But accessing list is slow. Let's try to optimize.
    
    # Iterate all 6-subsets
    for subset in itertools.combinations(range(n), 6):
        # Get colors of the 15 edges in this subset
        # Map local indices 0..5 to global indices in subset
        # local edge (i,j) in K6 -> global edge (subset[i], subset[j])
        
        # We need to check if any of the 4 colors forms a C6 in this subgraph.
        # For each color c, build a 15-bit mask of edges with color c.
        masks = [0, 0, 0, 0]
        
        for i in range(6):
            for j in range(i+1, 6):
                u, v = subset[i], subset[j]
                idx_global = _edge_index(n, u, v)
                c = colors[idx_global]
                # local edge index in K6
                idx_local = i * 6 - i * (i + 1) // 2 + (j - i - 1)
                masks[c] |= (1 << idx_local)
        
        # Check if any mask contains a C6 pattern
        for c in range(4):
            m = masks[c]
            for p in _C6_PATTERNS:
                if (m & p) == p:
                    return False
    return True

def _min_conflicts_fast(colors, n, max_iter=50000):
    """Local search using fast conflict counting."""
    m = n * (n - 1) // 2
    cur = colors[:]
    
    def count_conflicts(cols):
        # Reuse logic from _is_valid_fast but return count
        cnt = 0
        for subset in itertools.combinations(range(n), 6):
            masks = [0, 0, 0, 0]
            for i in range(6):
                for j in range(i+1, 6):
                    u, v = subset[i], subset[j]
                    idx_global = _edge_index(n, u, v)
                    c = cols[idx_global]
                    idx_local = i * 6 - i * (i + 1) // 2 + (j - i - 1)
                    masks[c] |= (1 << idx_local)
            for c in range(4):
                m = masks[c]
                for p in _C6_PATTERNS:
                    if (m & p) == p:
                        cnt += 1
        return cnt

    cur_conf = count_conflicts(cur)
    if cur_conf == 0:
        return cur, 0
        
    best = cur[:]
    best_conf = cur_conf
    
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
            
        # Pick a random edge and change color
        e = random.randrange(m)
        old_c = cur[e]
        new_c = (old_c + 1 + random.randrange(3)) % 4
        cur[e] = new_c
        
        new_conf = count_conflicts(cur)
        
        if new_conf < best_conf:
            best = cur[:]
            best_conf = new_conf
            if best_conf == 0:
                return best, 0
        else:
            # Revert if not better (greedy hill climb)
            cur[e] = old_c
            
    return best, best_conf

def construct():
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    random.seed(12345)
    
    # Try n from 17 down to 10
    # 18, 19, 20 are very hard, skip or try with low budget?
    # Let's try 17 first as it's known to exist.
    for n in range(17, 9, -1):
        if _remaining() < 10.0:
            break
            
        m = n * (n - 1) // 2
        found = False
        
        # Try multiple restarts of Min-Conflicts
        for restart in range(10):
            if _remaining() < 8.0:
                break
            rand_colors = [random.randrange(4) for _ in range(m)]
            res, conf = _min_conflicts_fast(rand_colors, n, max_iter=20000)
            if conf == 0:
                # Verify with fast validator
                if _is_valid_fast(res, n):
                    best_n = n
                    best_colors = res
                    found = True
                    break
        
        if found:
            break
            
    # Ensure we at least have the seed n=9
    if best_n < 9:
        best_n = 9
        best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
        
    return {'n': best_n, 'colors': best_colors}
