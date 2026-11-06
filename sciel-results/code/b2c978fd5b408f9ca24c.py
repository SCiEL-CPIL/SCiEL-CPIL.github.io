"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _get_all_progressions(n=128):
    """Precompute all nonconstant 4-term arithmetic progressions."""
    progs = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            e = a + 3 * d
            if e < n:
                progs.append((a, a + d, a + 2 * d, e))
    return progs

def _build_indices(ap_list, n=128):
    """Build position-to-AP index and precompute AP arrays."""
    pos_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(i)
    # Precompute as numpy arrays for speed
    ap_array = np.array(ap_list, dtype=np.int32)  # shape (num_aps, 4)
    return pos_to_aps, ap_array

def _count_violations(colors, ap_array):
    """Count violations. Returns (violation_count, first_violation_idx)."""
    c = np.array(colors, dtype=np.int32)
    # For each AP, get the 4 colors
    c4 = c[ap_array]  # shape (num_aps, 4)
    # A violation occurs if < 3 distinct colors
    # This means: all 4 same, or exactly 2 distinct
    # Check: sort each row and check patterns
    # Faster: check if (c4[:,0]==c4[:,1]) & (c4[:,1]==c4[:,2]) & (c4[:,2]==c4[:,3]) -> all same
    # Or exactly 2 distinct: check various patterns
    # Simplest vectorized: count distinct per row
    # For 4 elements, < 3 distinct means: all same OR exactly 2
    # all same: c4[:,0]==c4[:,1] & c4[:,1]==c4[:,2] & c4[:,2]==c4[:,3]
    # exactly 2: not all same, but the set has size 2
    # Let's use: min != max but the number of unique is 2
    # Alternative: a row has <3 distinct iff it can be partitioned into 2 groups
    # c4 has <3 distinct iff: (c4[:,0]==c4[:,1])&(c4[:,0]==c4[:,2])&(c4[:,0]==c4[:,3])  [all same]
    #   OR (c4[:,0]==c4[:,1])&(c4[:,2]==c4[:,3])  [AABB]
    #   OR (c4[:,0]==c4[:,2])&(c4[:,1]==c4[:,3])  [ABAB]
    #   OR (c4[:,0]==c4[:,3])&(c4[:,1]==c4[:,2])  [ABBA]
    #   OR (c4[:,0]==c4[:,1])&(c4[:,0]==c4[:,2])&(c4[:,1]!=c4[:,3])  [AAAB]
    #   OR (c4[:,0]==c4[:,1])&(c4[:,0]==c4[:,3])&(c4[:,1]!=c4[:,2])  [AABA]
    #   OR (c4[:,0]==c4[:,2])&(c4[:,0]==c4[:,3])&(c4[:,1]!=c4[:,2])  [ABAA]
    #   OR (c4[:,1]==c4[:,2])&(c4[:,1]==c4[:,3])&(c4[:,0]!=c4[:,1])  [ABBB]
    #   OR (c4[:,0]==c4[:,3])&(c4[:,1]==c4[:,2])  [ABBA already covered]
    # Actually simpler: <3 distinct means the 4 values can be matched to at most 2 values.
    # This is equivalent to: NOT (at least 3 are pairwise distinct in some sense)
    # Let me just do: for each row, check if len(set) < 3
    # Vectorized approach:
    # all_same = np.all(c4 == c4[:, 0:1], axis=1)
    # For exactly 2: we need to check if the 4 elements use exactly 2 values
    # This is: not all_same AND (there exist i,j,k,l such that c[i]==c[j] and c[k]==c[l] and c[i]!=c[k])
    # Simpler: a row has exactly 2 distinct iff:
    #   (c4[:,0]==c4[:,1]) & (c4[:,2]==c4[:,3]) & (c4[:,0]!=c4[:,2])
    #   OR (c4[:,0]==c4[:,2]) & (c4[:,1]==c4[:,3]) & (c4[:,0]!=c4[:,1])
    #   OR (c4[:,0]==c4[:,3]) & (c4[:,1]==c4[:,2]) & (c4[:,0]!=c4[:,1])
    #   OR (c4[:,0]==c4[:,1]) & (c4[:,0]==c4[:,2]) & (c4[:,1]!=c4[:,3])  [3 same, 1 diff]
    #   OR (c4[:,0]==c4[:,1]) & (c4[:,0]==c4[:,3]) & (c4[:,1]!=c4[:,2])
    #   OR (c4[:,0]==c4[:,2]) & (c4[:,0]==c4[:,3]) & (c4[:,1]!=c4[:,2])
    #   OR (c4[:,1]==c4[:,2]) & (c4[:,1]==c4[:,3]) & (c4[:,0]!=c4[:,1])
    
    all_same = np.all(c4 == c4[:, 0:1], axis=1)
    
    # Exactly 2 distinct: check all pair-equality patterns that give exactly 2 groups
    p01 = c4[:, 0] == c4[:, 1]
    p02 = c4[:, 0] == c4[:, 2]
    p03 = c4[:, 0] == c4[:, 3]
    p12 = c4[:, 1] == c4[:, 2]
    p13 = c4[:, 1] == c4[:, 3]
    p23 = c4[:, 2] == c4[:, 3]
    
    # Patterns for exactly 2 distinct:
    # AABB: p01 & p23 & ~p02 (c0!=c2)
    # ABAB: p02 & p13 & ~p01
    # ABBA: p03 & p12 & ~p01
    # AAAB: p01 & p02 & ~p13  (c0==c1==c2, c3 different)
    # AABA: p01 & p03 & ~p12
    # ABAA: p02 & p03 & ~p12
    # ABBB: p12 & p13 & ~p01
    # Also: c0==c1, c2==c3 but c0!=c2 is AABB
    # Let me be more careful. Exactly 2 distinct means the multiset has exactly 2 unique values.
    # Cases: 2+2, 3+1, 1+3, 4+0(all same - excluded), 0+4(excluded)
    # 2+2: two pairs. The pairings are: (0,1)&(2,3), (0,2)&(1,3), (0,3)&(1,2)
    # 3+1: three of one color, one of another. 4 sub-cases.
    # 1+3: same as 3+1.
    
    exactly2 = (
        (p01 & p23 & (c4[:, 0] != c4[:, 2])) |  # AABB
        (p02 & p13 & (c4[:, 0] != c4[:, 1])) |  # ABAB
        (p03 & p12 & (c4[:, 0] != c4[:, 1])) |  # ABBA
        (p01 & p02 & (c4[:, 1] != c4[:, 3])) |  # AAAB
        (p01 & p03 & (c4[:, 1] != c4[:, 2])) |  # AABA
        (p02 & p03 & (c4[:, 1] != c4[:, 2])) |  # ABAA
        (p12 & p13 & (c4[:, 0] != c4[:, 1]))     # ABBB
    )
    
    violation = all_same | exactly2
    v_count = int(np.sum(violation))
    if v_count > 0:
        first_idx = int(np.argmax(violation))
    else:
        first_idx = -1
    return v_count, first_idx

def _local_search_structural(k, deadline, ap_array, pos_to_aps, n=128, max_restarts=100):
    """
    Local search: minimize violations, then minimize distinct colors used.
    """
    best_colors = None
    best_score = (float('inf'), float('inf'))  # (violations, distinct_colors)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize
        if restart == 0:
            # Structured init: use modular pattern
            colors = [i % k for i in range(n)]
        else:
            colors = [random.randint(0, k - 1) for _ in range(n)]
        
        stuck_count = 0
        max_stuck = 200
        
        while time.time() < deadline:
            v_count, v_idx = _count_violations(colors, ap_array)
            distinct = len(set(colors))
            score = (v_count, distinct)
            
            if v_count == 0:
                return colors
            
            if score < best_score:
                best_score = score
                best_colors = colors[:]
                stuck_count = 0
            else:
                stuck_count += 1
            
            if stuck_count > max_stuck:
                if random.random() < 0.5 and best_colors is not None:
                    colors = best_colors[:]
                else:
                    colors = [random.randint(0, k - 1) for _ in range(n)]
                stuck_count = 0
                continue
            
            # Pick a position to change
            if v_idx >= 0:
                # Pick a position from the violating AP
                ap = tuple(ap_array[v_idx])
                pos = ap[random.randint(0, 3)]
            else:
                pos = random.randint(0, n - 1)
            
            old_color = colors[pos]
            # Try a new color
            new_color = random.randint(0, k - 1)
            if new_color == old_color:
                new_color = (new_color + 1) % k
            colors[pos] = new_color
            
            # Check improvement
            new_v, _ = _count_violations(colors, ap_array)
            new_distinct = len(set(colors))
            new_score = (new_v, new_distinct)
            
            if new_score >= score:
                colors[pos] = old_color
    
    return best_colors

def _check_valid(colors, ap_array):
    """Check if all nonconstant 4-term progressions use at least 3 distinct colors."""
    c = np.array(colors, dtype=np.int32)
    c4 = c[ap_array]
    all_same = np.all(c4 == c4[:, 0:1], axis=1)
    p01 = c4[:, 0] == c4[:, 1]
    p02 = c4[:, 0] == c4[:, 2]
    p03 = c4[:, 0] == c4[:, 3]
    p12 = c4[:, 1] == c4[:, 2]
    p13 = c4[:, 1] == c4[:, 3]
    p23 = c4[:, 2] == c4[:, 3]
    exactly2 = (
        (p01 & p23 & (c4[:, 0] != c4[:, 2])) |
        (p02 & p13 & (c4[:, 0] != c4[:, 1])) |
        (p03 & p12 & (c4[:, 0] != c4[:, 1])) |
        (p01 & p02 & (c4[:, 1] != c4[:, 3])) |
        (p01 & p03 & (c4[:, 1] != c4[:, 2])) |
        (p02 & p03 & (c4[:, 1] != c4[:, 2])) |
        (p12 & p13 & (c4[:, 0] != c4[:, 1]))
    )
    return not np.any(all_same | exactly2)

def construct():
    start_time = time.time()
    deadline = start_time + 40  # Safety margin
    
    n = 128
    ap_list = _get_all_progressions(n)
    pos_to_aps, ap_array = _build_indices(ap_list, n)
    
    # Try increasing k from 3 upward
    for k in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 10:
            break
        
        k_deadline = deadline - 10
        result = _local_search_structural(k, k_deadline, ap_array, pos_to_aps, n, max_restarts=50)
        
        if result is not None:
            if _check_valid(result, ap_array):
                return result
    
    # Fallback with higher k
    for k in [9, 10, 12, 16]:
        if time.time() > deadline - 3:
            break
        k_deadline = deadline - 3
        result = _local_search_structural(k, k_deadline, ap_array, pos_to_aps, n, max_restarts=20)
        if result is not None:
            if _check_valid(result, ap_array):
                return result
    
    # Absolute fallback
    return list(range(n))

# EVOLVE-BLOCK-END
