import time
import random

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0

    def verify(n, colors):
        """Verify that no color class contains a C6."""
        # Build adjacency per color
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        # For each color, check for C6 using DFS from each edge
        # A C6 is a cycle of length 6.
        # We check if there is a simple path of length 5 between any two adjacent nodes u, v in the graph.
        # If such a path exists, u-v + path forms a C6.
        for c in range(4):
            for i in range(n):
                # Get neighbors of i in color c
                neighbors_i = adj[c][i]
                while neighbors_i:
                    lsb_i = neighbors_i & -neighbors_i
                    j = lsb_i.bit_length() - 1
                    neighbors_i -= lsb_i
                    
                    if j <= i: continue # Only check each edge once? 
                    # Actually, checking all directed edges is fine, just redundant.
                    # Let's stick to i < j to save time, but the path check is directed.
                    # If path i->...->j exists, then C6 exists.
                    
                    # Check if there is a simple path of length 5 from i to j
                    # i - a - b - d - e - j
                    # a, b, d, e must be distinct and not i, j
                    
                    # Optimization: 
                    # We can use BFS/DFS with depth 5.
                    # State: (current_node, depth_remaining, visited_mask)
                    # Start: (i, 5, 1<<i)
                    # Goal: reach j with depth 0? No, reach a neighbor of j with depth 1?
                    # Path length 5 means 5 edges.
                    # i -> v1 -> v2 -> v3 -> v4 -> j
                    # So we need a path of length 5 from i to j.
                    
                    found = False
                    stack = [(i, 5, 1 << i)]
                    while stack and not found:
                        cur, depth, visited = stack.pop()
                        if depth == 0:
                            if cur == j:
                                found = True
                            continue
                        if depth == 1:
                            # Check if j is a neighbor of cur
                            if (adj[c][cur] >> j) & 1:
                                found = True
                            continue
                        
                        mask = adj[c][cur] & ~visited
                        while mask:
                            lsb = mask & -mask
                            nxt = lsb.bit_length() - 1
                            stack.append((nxt, depth - 1, visited | lsb))
                            mask -= lsb
                            
                    if found:
                        return False
        return True

    def get_cyclic_coloring(n):
        """
        Generates a 4-coloring of K_n using cyclic differences.
        For n=17, differences 1..8 are paired:
        {1,2} -> 0
        {3,4} -> 1
        {5,6} -> 2
        {7,8} -> 3
        """
        if n != 17:
            return None
            
        colors = []
        # Mapping from difference d (1..8) to color
        # d is min(j-i, n-(j-i))
        # Pairs:
        # 1, 2 -> 0
        # 3, 4 -> 1
        # 5, 6 -> 2
        # 7, 8 -> 3
        
        diff_to_color = {
            1: 0, 2: 0,
            3: 1, 4: 1,
            5: 2, 6: 2,
            7: 3, 8: 3
        }
        
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                # In Z_n, the distance is min(d, n-d)
                # But for undirected graph on Z_n, edge (i, i+d) is same as (i, i-d).
                # So we use d_eff = min(d, n-d)
                d_eff = min(d, n - d)
                colors.append(diff_to_color[d_eff])
        return colors

    # Attempt 1: Deterministic cyclic construction for n=17
    n = 17
    colors = get_cyclic_coloring(n)
    if colors is not None:
        if verify(n, colors):
            return {'n': n, 'colors': colors}
        else:
            # If the simple cyclic one fails, we might need a more complex one.
            # But theory suggests it should work.
            # Let's try a different pairing just in case.
            # Pairs: {1,3}, {2,4}, {5,7}, {6,8} ?
            # Or random search for a valid permutation of differences?
            pass

    # Attempt 2: Backtracking search for n=17 if cyclic failed (unlikely)
    # Or try n=16 with cyclic construction?
    # For n=16, differences 1..8.
    # {1,2}, {3,4}, {5,6}, {7,8}
    # Max sum of 6 steps from {1,2} is 12. 16 > 12. So no C6 in color 0.
    # This should also work.
    
    # Let's try to generate a valid coloring for n=17 using a simple backtracking 
    # if the deterministic one fails. But the deterministic one is O(N^2) to verify.
    
    # If n=17 fails verification (shouldn't), try n=16.
    if time.time() < hard_deadline - 5:
        n = 16
        colors = get_cyclic_coloring(n) # Reuse logic, but map for n=16
        # For n=16, d_eff is 1..8.
        # Same mapping works.
        if verify(n, colors):
            return {'n': n, 'colors': colors}

    # Fallback to n=15
    if time.time() < hard_deadline - 5:
        n = 15
        colors = get_cyclic_coloring(n) # Mapping might need adjustment for n=15
        # d_eff 1..7.
        # {1,2}, {3,4}, {5,6}, {7} ? 
        # We have 4 colors. 
        # 1,2->0; 3,4->1; 5,6->2; 7->3.
        # Max sum 6*2=12 < 15. Safe.
        diff_to_color_15 = {1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 7:3}
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                d_eff = min(d, n - d)
                colors.append(diff_to_color_15[d_eff])
        if verify(n, colors):
            return {'n': n, 'colors': colors}

    # Absolute fallback
    n = 12
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append((i + j) % 4)
    if verify(n, colors):
        return {'n': n, 'colors': colors}
        
    return {'n': 9, 'colors': [0]*36} # Dummy, should not happen

