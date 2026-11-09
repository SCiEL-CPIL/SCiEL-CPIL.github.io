"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 128
    colors = [-1] * n  # 0-indexed: colors[i] is color for position i+1
    
    def is_valid_progression(pos, d):
        """Check if positions pos, pos+d, pos+2d, pos+3d are all assigned and use >=3 colors."""
        if pos + 3*d >= n:
            return True
        vals = [colors[pos], colors[pos+d], colors[pos+2*d], colors[pos+3*d]]
        if any(v == -1 for v in vals):
            return True  # not fully assigned yet
        return len(set(vals)) >= 3
    
    def check_conflicts(pos, color):
        """Check if assigning color to position pos (0-indexed) creates any violation
        with already-assigned positions forming 4-term progressions."""
        # Check all 4-term progressions where pos could be any of the 4 terms
        # pos = a + j*d for j in 0,1,2,3
        # Case j=0: pos=a, need pos+d, pos+2d, pos+3d
        for d in range(1, (n - pos) // 3 + 1):
            if pos + 3*d < n:
                vals = [color, colors[pos+d], colors[pos+2*d], colors[pos+3*d]]
                if all(v != -1 for v in vals):
                    if len(set(vals)) < 3:
                        return False
        # Case j=1: pos-d=a, pos=a+d, need pos-2d, pos-3d
        for d in range(1, pos // 2 + 1):
            if pos - 3*d >= 0:
                vals = [colors[pos-3*d], colors[pos-2*d], color, colors[pos+d]]
                if all(v != -1 for v in vals):
                    if len(set(vals)) < 3:
                        return False
        # Case j=2: pos-2d=a, need pos-3d, pos+d
        for d in range(1, pos // 2 + 1):
            if pos - 3*d >= 0 and pos + d < n:
                vals = [colors[pos-3*d], color, colors[pos+d], colors[pos+2*d]]
                if all(v != -1 for v in vals):
                    if len(set(vals)) < 3:
                        return False
        # Case j=3: pos-3d=a, need pos-2d, pos-d
        for d in range(1, pos // 3 + 1):
            if pos - 3*d >= 0:
                vals = [colors[pos-3*d], colors[pos-2*d], colors[pos-d], color]
                if all(v != -1 for v in vals):
                    if len(set(vals)) < 3:
                        return False
        return True
    
    def greedy_assign_next(start_pos, lookahead_k):
        """Try to greedily assign colors from start_pos to start_pos+lookahead_k-1.
        Returns True if all can be assigned without immediate conflicts."""
        saved = {}
        success = True
        for p in range(start_pos, min(start_pos + lookahead_k, n)):
            assigned = False
            for c in range(128):
                if check_conflicts(p, c):
                    colors[p] = c
                    saved[p] = c
                    assigned = True
                    break
            if not assigned:
                success = False
                break
        # Restore
        for p, c in saved.items():
            colors[p] = -1
        return success
    
    # Assign position 0 (x=1)
    colors[0] = 0
    
    # Assign remaining positions
    for x in range(1, n):
        found = False
        for c in range(128):
            if not check_conflicts(x, c):
                continue
            # Look-ahead: try to assign next 4 positions greedily
            colors[x] = c
            if greedy_assign_next(x + 1, 4):
                found = True
                break
            colors[x] = -1
        if not found:
            # Fallback: just assign smallest valid color without lookahead
            for c in range(128):
                if check_conflicts(x, c):
                    colors[x] = c
                    found = True
                    break
        # If still not found, something is wrong
        if not found:
            colors[x] = 0
    
    return colors
# EVOLVE-BLOCK-END
