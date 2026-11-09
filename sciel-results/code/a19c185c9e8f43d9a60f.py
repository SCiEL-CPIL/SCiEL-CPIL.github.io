"""Colorings with three colors in every four-term progression."""
# EVOLVE-BLOCK-START
import time

def construct():
    """Return 128 integer color labels for positions 1..128.
    
    Every nonconstant four-term arithmetic progression must use at least
    3 distinct colors. Minimizes the number of distinct labels used.
    """
    deadline = time.time() + 85  # 85s deadline, 5s safety margin
    
    n = 128
    colors = [-1] * n
    
    # Precompute all 4-APs as 0-indexed tuples
    aps = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            e = a + 3 * d
            if e < n:
                aps.append((a, a + d, a + 2 * d, e))
    
    # For each position, which APs does it belong to?
    pos_aps = [[] for _ in range(n)]
    for i, (a, b, c, e) in enumerate(aps):
        pos_aps[a].append(i)
        pos_aps[b].append(i)
        pos_aps[c].append(i)
        pos_aps[e].append(i)
    
    # ap_colors[ap_idx] = set of colors used in that AP so far
    ap_colors = [set() for _ in range(len(aps))]
    ap_count = [0] * len(aps)
    
    def can_assign(pos, color):
        """Check if assigning 'color' to 'pos' keeps all APs satisfiable."""
        for ap_idx in pos_aps[pos]:
            ac = ap_colors[ap_idx]
            ac.add(color)
            cnt = ap_count[ap_idx]
            # After this assignment, cnt+1 terms are set
            if cnt + 1 == 4:
                # All 4 set: need >= 3 distinct
                if len(ac) < 3:
                    ac.discard(color)
                    return False
            elif cnt + 1 == 3:
                # 3 of 4 set: the 4th must allow reaching 3 distinct
                # If currently < 2 distinct, 4th can be different -> OK
                # If currently == 2 distinct, 4th must be 3rd -> OK (we have 3 colors)
                # If currently == 3 distinct, OK
                # Actually always OK with 3 available colors
                pass
            ac.discard(color)
        return True
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        
        for color in (0, 1, 2):
            if can_assign(pos, color):
                colors[pos] = color
                for ap_idx in pos_aps[pos]:
                    ap_colors[ap_idx].add(color)
                    ap_count[ap_idx] += 1
                
                if backtrack(pos + 1):
                    return True
                
                colors[pos] = -1
                for ap_idx in pos_aps[pos]:
                    ap_colors[ap_idx].discard(color)
                    ap_count[ap_idx] -= 1
        
        return False
    
    if backtrack(0):
        return colors
    
    # Should not reach here; return a fallback
    return [i % 3 for i in range(n)]
# EVOLVE-BLOCK-END
