import numpy as np
import time

def construct():
    n = 128
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute: for each position i (0-indexed), list of (a, d) such that i = a+3d
    # i.e., i is the 4th term of the AP
    # Also need to handle cases where i is the 1st, 2nd, or 3rd term
    # But in greedy left-to-right, when we assign position i, all previous positions are known.
    # So we only need to check APs where i is the last (4th) term, since those are the only
    # APs that become fully determined when we assign i.
    
    # Wait, that's not quite right. An AP (a, a+d, a+2d, a+3d) where a+3d = i becomes
    # fully determined when we assign i. But we also need to ensure that when we assigned
    # earlier positions, we didn't create a violation. However, in a proper greedy, we
    # check all constraints that involve the newly assigned position.
    
    # Actually, for a left-to-right greedy, when assigning position i, the only new
    # constraints that become "fully assigned" are those where i is the 4th term.
    # But we should also verify that no previously-fully-assigned AP is violated - 
    # but those were already checked. So we only need to check APs where i is the 4th term.
    
    # For position i (0-indexed), the APs where i is the 4th term:
    # i = a + 3d, so a = i - 3d, and we need a >= 0, so d <= i//3
    # The AP is (i-3d, i-2d, i-d, i)
    
    # Greedy: for each position i from 0 to n-1, try colors 0, 1, 2, ... in order
    # and pick the first one that doesn't violate any AP where i is the 4th term.
    
    colors = [-1] * n
    
    for i in range(n):
        if time.time() > deadline:
            break
        for c in range(128):
            valid = True
            # Check all APs where i is the 4th term
            for d in range(1, i // 3 + 1):
                a = i - 3 * d
                if a < 0:
                    break
                # AP is (a, a+d, a+2d, i) = (i-3d, i-2d, i-d, i)
                vals = [colors[i - 3*d], colors[i - 2*d], colors[i - d], c]
                if len(set(vals)) < 3:
                    valid = False
                    break
            if valid:
                colors[i] = c
                break
        else:
            # No color found, which shouldn't happen with enough colors
            colors[i] = 0
    
    # Verify and count distinct colors
    used_colors = set(colors)
    num_colors = len(used_colors)
    
    return colors
