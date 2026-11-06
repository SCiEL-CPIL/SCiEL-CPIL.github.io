"""Rewritten program: Backtracking CSP solver for 3-color 4-AP avoidance with bitmask lookup."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    deadline = time.time() + 80  # Safety margin
    N = 128
    
    # Precompute all nonconstant 4-term progressions
    aps = []
    for a in range(N):
        for d in range(1, (N - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps.append((a, b, c, e))
            
    # Map index to list of (ap_index, role)
    # role 0: a, 1: b, 2: c, 3: e
    idx_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        idx_to_aps[ap[0]].append((idx, 0))
        idx_to_aps[ap[1]].append((idx, 1))
        idx_to_aps[ap[2]].append((idx, 2))
        idx_to_aps[ap[3]].append((idx, 3))

    # Precompute forbidden color masks for K=3 and K=4
    # Key: tuple of 3 colors (sorted? no, order matters for distinct count? no, just set)
    # Actually, the constraint depends on the SET of colors of the other 3.
    # If other 3 are (0,0,0), set is {0}. Forbidden: all.
    # If other 3 are (0,0,1), set is {0,1}. Forbidden: {0,1}.
    # If other 3 are (0,1,2), set is {0,1,2}. Forbidden: {}.
    
    def get_forbidden_mask_lookup(k):
        lookup = {}
        # Possible sets of 3 colors from 0..k-1
        # We can just compute on the fly or precompute for all 3-tuples
        # There are k^3 tuples. For k=4, 64. Very small.
        for c1 in range(k):
            for c2 in range(k):
                for c3 in range(k):
                    distinct = {c1, c2, c3}
                    mask = 0
                    if len(distinct) == 1:
                        # All same. Any color added results in size 1 or 2.
                        # We need size >= 3.
                        # If we add c1, set is {c1} (size 1).
                        # If we add x != c1, set is {c1, x} (size 2).
                        # So ALL colors are forbidden.
                        mask = (1 << k) - 1
                    elif len(distinct) == 2:
                        # e.g. {0, 1}.
                        # If we add 0, set {0,1} (size 2). Forbidden.
                        # If we add 1, set {0,1} (size 2). Forbidden.
                        # If we add 2, set {0,1,2} (size 3). Allowed.
                        # If we add 3, set {0,1,3} (size 3). Allowed.
                        for c in distinct:
                            mask |= (1 << c)
                    else: # len(distinct) == 3
                        # e.g. {0, 1, 2}.
                        # Adding any color x:
                        # If x in {0,1,2}, set is {0,1,2} (size 3). Allowed.
                        # If x not in {0,1,2}, set is {0,1,2,x} (size 4). Allowed.
                        mask = 0
                    lookup[(c1, c2, c3)] = mask
        return lookup

    lookup_3 = get_forbidden_mask_lookup(3)
    lookup_4 = get_forbidden_mask_lookup(4)

    def solve(k, node_limit=500000):
        colors = [-1] * N
        lookup = lookup_3 if k == 3 else lookup_4
        nodes_visited = 0
        max_nodes = node_limit
        
        def backtrack(pos):
            nonlocal nodes_visited, max_nodes
            if time.time() > deadline:
                return False
            if nodes_visited > max_nodes:
                return False
            
            # Find unassigned index with Minimum Remaining Values (MRV)
            # For simplicity and speed, we can just pick the first unassigned, 
            # but MRV is much faster for CSP.
            # Given N=128, scanning for MRV is O(N * deg). 
            # Let's stick to a simple heuristic: pick the first unassigned index.
            # Or better: pick the index with the most constraints?
            # Actually, for 4-APs, the structure is regular.
            # Let's try picking the first unassigned index for simplicity first.
            # If it's too slow, we can optimize.
            
            # Optimization: Pick the index i that is unassigned and has the fewest available colors.
            # To avoid O(N^2) overhead, we can just iterate.
            best_i = -1
            best_mask = 0
            best_avail_count = k + 1
            
            # We can also just pick the first unassigned index to keep it simple.
            # Let's try first unassigned index first.
            for i in range(N):
                if colors[i] == -1:
                    best_i = i
                    break
            
            if best_i == -1:
                return True # All assigned
            
            nodes_visited += 1
            
            # Calculate forbidden mask for best_i
            forbidden = 0
            for ap_idx, role in idx_to_aps[best_i]:
                a, b, c, e = aps[ap_idx]
                # Get the other 3 indices
                if role == 0:
                    others = [colors[b], colors[c], colors[e]]
                elif role == 1:
                    others = [colors[a], colors[c], colors[e]]
                elif role == 2:
                    others = [colors[a], colors[b], colors[e]]
                else:
                    others = [colors[a], colors[b], colors[c]]
                
                # If any other is unassigned, we can't check this AP yet
                if -1 in others:
                    continue
                
                # Check constraint
                # The other 3 colors are fixed.
                # We need to ensure that for ANY color we pick for i, the total distinct count >= 3.
                # This is equivalent to: The color we pick for i must NOT be in the "forbidden" set derived from others.
                # lookup[other_tuple] gives the mask of colors that would make the AP invalid.
                other_key = (others[0], others[1], others[2])
                forbidden |= lookup[other_key]
            
            # Try available colors
            for c in range(k):
                if not (forbidden & (1 << c)):
                    colors[best_i] = c
                    if backtrack(pos + 1):
                        return True
                    colors[best_i] = -1
            
            return False

        if backtrack(0):
            return colors
        return None

    # Try 3 colors
    best_colors = solve(3)
    if best_colors:
        # Validate just in case
        for (a, b, c, e) in aps:
            if len({best_colors[a], best_colors[b], best_colors[c], best_colors[e]}) < 3:
                best_colors = None
                break
    
    if not best_colors:
        # Try 4 colors
        best_colors = solve(4)
        if best_colors:
            for (a, b, c, e) in aps:
                if len({best_colors[a], best_colors[b], best_colors[c], best_colors[e]}) < 3:
                    best_colors = None
                    break

    if not best_colors:
        # Fallback: Random 5-coloring with validation (should not happen)
        import random
        for _ in range(100):
            if time.time() > deadline:
                break
            rng = random.Random()
            colors = [rng.randint(0, 4) for _ in range(N)]
            valid = True
            for (a, b, c, e) in aps:
                if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                    valid = False
                    break
            if valid:
                best_colors = colors
                break
        
        if not best_colors:
            # Last resort: return a known invalid but non-crashing pattern?
            # No, we must return valid. 
            # If we failed, we have a problem. 
            # But 4-coloring should always work.
            # Let's force a 4-coloring with a different random seed if above failed.
            for _ in range(1000):
                if time.time() > deadline:
                    break
                rng = random.Random()
                colors = [rng.randint(0, 3) for _ in range(N)]
                valid = True
                for (a, b, c, e) in aps:
                    if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                        valid = False
                        break
                if valid:
                    best_colors = colors
                    break

    if not best_colors:
        # Absolute last resort: 128 unique colors (valid but high score)
        return list(range(N))

    # Relabel to minimize number of distinct labels used (though we already minimized k)
    # The task says "Minimize the number of labels actually used".
    # Our solver uses exactly k colors (0..k-1).
    # So the number of labels is k.
    # We can relabel to ensure they are 0..k-1 contiguous, which they are.
    return best_colors
# EVOLVE-BLOCK-END
