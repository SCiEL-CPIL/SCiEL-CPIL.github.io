import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def check_k222(adj_masks, color, n):
        """
        Returns True if the graph defined by adj_masks[color] contains K2,2,2.
        adj_masks is a list of 2 lists of ints, where adj_masks[c][i] is a bitmask of neighbors of i in color c.
        """
        for u in range(n):
            for v in range(u + 1, n):
                # Check if edge (u,v) is color 'color'
                if not (adj_masks[color][u] >> v) & 1:
                    continue
                
                # S = common neighbors of u and v in color 'color', excluding u and v
                # Note: adj_masks are symmetric
                s_mask = adj_masks[color][u] & adj_masks[color][v]
                s_mask &= ~(1 << u)
                s_mask &= ~(1 << v)
                
                # Check if the subgraph induced by s_mask contains K2,2
                # We need to find two disjoint edges (a,b) and (c,d) in s_mask
                # such that (a,c), (a,d), (b,c), (b,d) are all in adj_masks[color]
                
                # Get list of vertices in s_mask
                verts = []
                temp = s_mask
                while temp:
                    lsb = temp & -temp
                    idx = (lsb.bit_length() - 1)
                    verts.append(idx)
                    temp ^= lsb
                
                # If less than 4 vertices, no K2,2
                if len(verts) < 4:
                    continue
                
                # Iterate over pairs of edges in the induced subgraph
                # To be efficient, we can just iterate over all pairs of vertices in verts
                # and check for K2,2 structure
                # A K2,2 is defined by 4 distinct vertices a,b,c,d such that
                # (a,b) is edge, (c,d) is edge, and cross edges exist.
                
                # Optimization: Iterate over edges (a,b) in the subgraph
                edges = []
                for i in range(len(verts)):
                    a = verts[i]
                    # Neighbors of a in s_mask
                    neigh_a = adj_masks[color][a] & s_mask
                    # Only consider neighbors b > a to avoid duplicates? No, just check all
                    # Actually, we just need to find ANY K2,2
                    # Let's iterate over all pairs (a,b) in verts
                    for j in range(i + 1, len(verts)):
                        b = verts[j]
                        if (adj_masks[color][a] >> b) & 1:
                            # (a,b) is an edge in color 'color'
                            # Now we need two other vertices c,d in s_mask \ {a,b}
                            # such that (c,d) is an edge, and cross edges exist.
                            # Let's iterate over pairs (c,d) from verts excluding a,b
                            # This is O(k^4) in worst case, but k is small.
                            remaining = [x for x in verts if x != a and x != b]
                            for k_idx in range(len(remaining)):
                                c = remaining[k_idx]
                                # Check cross edges from a,b to c
                                if not ((adj_masks[color][a] >> c) & 1 and (adj_masks[color][b] >> c) & 1):
                                    continue
                                for l_idx in range(k_idx + 1, len(remaining)):
                                    d = remaining[l_idx]
                                    # Check cross edges from a,b to d
                                    if not ((adj_masks[color][a] >> d) & 1 and (adj_masks[color][b] >> d) & 1):
                                        continue
                                    # Check if (c,d) is an edge
                                    if (adj_masks[color][c] >> d) & 1:
                                        return True
        return False

    def get_conflict(adj_masks, color, n):
        """
        Returns a list of edges (u,v) forming a K2,2,2 in the given color, or None.
        """
        # Similar to check_k222 but returns the subgraph
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_masks[color][u] >> v) & 1:
                    continue
                s_mask = adj_masks[color][u] & adj_masks[color][v]
                s_mask &= ~(1 << u)
                s_mask &= ~(1 << v)
                
                verts = []
                temp = s_mask
                while temp:
                    lsb = temp & -temp
                    idx = (lsb.bit_length() - 1)
                    verts.append(idx)
                    temp ^= lsb
                
                if len(verts) < 4:
                    continue
                
                for i in range(len(verts)):
                    a = verts[i]
                    for j in range(i + 1, len(verts)):
                        b = verts[j]
                        if (adj_masks[color][a] >> b) & 1:
                            remaining = [x for x in verts if x != a and x != b]
                            for k_idx in range(len(remaining)):
                                c = remaining[k_idx]
                                if not ((adj_masks[color][a] >> c) & 1 and (adj_masks[color][b] >> c) & 1):
                                    continue
                                for l_idx in range(k_idx + 1, len(remaining)):
                                    d = remaining[l_idx]
                                    if not ((adj_masks[color][a] >> d) & 1 and (adj_masks[color][b] >> d) & 1):
                                        continue
                                    if (adj_masks[color][c] >> d) & 1:
                                        # Found K2,2,2: u,v,a,b,c,d
                                        # The edges involved are all pairs from {u,v,a,b,c,d}
                                        # except possibly edges within {u,v}, {a,b}, {c,d}
                                        # Actually, the K2,2,2 edges are the cross edges.
                                        # We can return the set of vertices to pick an edge from
                                        return [u, v, a, b, c, d]
        return None

    def solve(n):
        # Initialize random
        # adj[0] and adj[1] are lists of bitmasks
        adj = [[0] * n for _ in range(2)]
        
        # Symmetry breaking: Fix first few edges
        # e(0,1) = 0
        adj[0][0] |= 1 << 1
        adj[0][1] |= 1 << 0
        # e(0,2) = 0
        adj[0][0] |= 1 << 2
        adj[0][2] |= 1 << 0
        # e(1,2) = 0
        adj[0][1] |= 1 << 2
        adj[0][2] |= 1 << 1
        
        # Initialize rest randomly
        for i in range(n):
            for j in range(i + 1, n):
                # Skip if already fixed
                if (adj[0][i] >> j) & 1 or (adj[1][i] >> j) & 1:
                    continue
                if random.random() < 0.5:
                    adj[0][i] |= 1 << j
                    adj[0][j] |= 1 << i
                else:
                    adj[1][i] |= 1 << j
                    adj[1][j] |= 1 << i
        
        # Hill Climbing
        max_iterations = 5000
        no_improve_count = 0
        last_score = -1
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                return None
            
            # Check validity
            conflict0 = get_conflict(adj, 0, n)
            conflict1 = get_conflict(adj, 1, n)
            
            if conflict0 is None and conflict1 is None:
                # Valid!
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[0][i] >> j) & 1:
                            colors.append(0)
                        else:
                            colors.append(1)
                return {'n': n, 'colors': colors}
            
            # Pick a conflict
            if conflict0 is not None and (conflict1 is None or random.random() < 0.5):
                verts = conflict0
                color_to_flip = 0
            else:
                verts = conflict1
                color_to_flip = 1
            
            # Pick a random edge from the conflict vertices to flip
            # We must flip an edge that is currently color_to_flip
            # And changing it might break the K2,2,2
            # Let's pick a random pair from verts
            random.shuffle(verts)
            flipped = False
            for idx in range(len(verts)):
                for jdx in range(idx + 1, len(verts)):
                    u = verts[idx]
                    v = verts[jdx]
                    # Check if edge (u,v) is color_to_flip
                    if (adj[color_to_flip][u] >> v) & 1:
                        # Flip it
                        # Remove from color_to_flip
                        adj[color_to_flip][u] &= ~(1 << v)
                        adj[color_to_flip][v] &= ~(1 << u)
                        # Add to other color
                        other = 1 - color_to_flip
                        adj[other][u] |= 1 << v
                        adj[other][v] |= 1 << u
                        flipped = True
                        break
                if flipped:
                    break
            
            if not flipped:
                # Should not happen if conflict is valid
                break

        return None

    # Try n=29, then 28, etc.
    for n in range(29, 16, -1):
        # Run multiple attempts for each n
        attempts = 10 if n == 29 else 5
        for _ in range(attempts):
            if time.time() > deadline:
                break
            result = solve(n)
            if result is not None:
                return result
        if time.time() > deadline:
            break

    # Fallback: Return the known n=17 Paley coloring
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}

# EVOLVE-BLOCK-START
# The code above is the implementation.
# EVOLVE-BLOCK-END
