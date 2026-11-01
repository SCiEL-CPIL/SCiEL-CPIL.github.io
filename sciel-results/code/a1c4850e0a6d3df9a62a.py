import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    # Moore bound: lower bound on n
    # For k-regular girth g:
    # if g is even: n >= 2 * (k-1)^(g/2 - 1) + k * (k-1)^(g/2 - 1) = k * (k-1)^(g/2 - 1) * 2 / (k-1) ... 
    # Actually standard Moore bound: n >= 1 + k * sum_{i=0}^{g/2-2} (k-1)^i for even g
    # For odd g: n >= 1 + k * (sum_{i=0}^{(g-3)/2} (k-1)^i + sum_{i=0}^{(g-1)/2} (k-1)^i)
    # But we don't need the exact Moore bound for the algorithm, just for reference.

    # Fallback: LUW graph construction
    def luw_construct(k, g):
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
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Try to find a lift of K5 with girth g
    if k == 4 and g == 10:
        # Lift of K5 by Z2^6
        base_edges = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
        num_base_vertices = 5
        group_order = 64
        total_vertices = num_base_vertices * group_order
        
        # Precompute base adjacency with edge indices
        base_adj = [[] for _ in range(num_base_vertices)]
        for idx, (u, v) in enumerate(base_edges):
            base_adj[u].append((v, idx))
            base_adj[v].append((u, idx))
        
        # For checking girth, we need to detect cycles of length < 10
        # A cycle in the lift corresponds to a closed walk in the base graph
        # where the sum of voltages (mod 2 in each coordinate) is 0.
        
        # We can enumerate all closed walks of length L < 10 in the base graph
        # and check if any has voltage sum = 0.
        
        # For efficiency, we enumerate closed walks up to length 9.
        # A closed walk of length L in K5: start at vertex s, take L edges, return to s.
        # Number of such walks: for each s, 4^L (but with constraints of being in K5).
        # Actually K5 is complete, so from any vertex you can go to any other.
        # So number of closed walks of length L from s is 4^L (each step has 4 choices).
        # Total: 5 * 4^9 = 5 * 262144 = 1,310,720. This is manageable.
        
        # But we need to track the voltage sum. Each edge has a voltage in Z2^6.
        # We can represent the voltage sum as an integer 0..63.
        
        # For each starting vertex s and each starting edge (s, v1), 
        # we can do a DFS/BFS tracking (current_vertex, voltage_sum, length).
        # We want to find if there's a closed walk of length L < 10 with voltage_sum = 0.
        
        # Actually, a closed walk of length L starting and ending at s with voltage sum 0
        # gives a cycle of length L in the lift (if the walk is simple in the lift, 
        # but even if not, it contains a cycle of length <= L).
        
        # Wait, a closed walk in the base graph with voltage sum 0 corresponds to a closed walk
        # in the lift. This closed walk in the lift may not be a simple cycle, but it contains
        # a simple cycle of length <= L. So if we find any closed walk of length L < 10 with 
        # voltage sum 0, then there is a cycle of length < 10.
        
        # Conversely, if there is a cycle of length L < 10 in the lift, it projects to a closed 
        # walk of length L in the base graph with voltage sum 0.
        
        # So the condition "girth >= 10" is equivalent to "no closed walk of length L < 10 
        # in the base graph has voltage sum 0".
        
        # Let's implement this check.
        
        def check_girth(voltages):
            # voltages: list of 10 integers 0..63
            # Check if any closed walk of length 1..9 has voltage sum 0
            
            # For each start vertex s:
            #   For each first edge (s, v1):
            #     Do DFS/BFS tracking (current_vertex, voltage_sum, length)
            #     If we return to s with voltage_sum = 0 and length < 10, return False
            
            # To avoid exponential blowup, note that we only need to check up to length 9.
            # The number of walks is bounded.
            
            # Actually, let's be smarter. We can use dynamic programming.
            # dp[step][vertex][voltage] = True if there's a walk of length 'step' from s 
            #   to 'vertex' with voltage sum 'voltage'.
            # But we want closed walks, so we fix s.
            
            # For each s in 0..4:
            #   dp[0][s][0] = True
            #   For step in 1..9:
            #     For each vertex v:
            #       For each voltage vol:
            #         If dp[step-1][v][vol]:
            #           For each neighbor w of v:
            #             new_vol = (vol + voltages[edge_idx(v,w)]) % 64
            #             dp[step][w][new_vol] = True
            #   If dp[L][s][0] for any L in 1..9, return False
            
            # This is 5 * 10 * 5 * 64 * 4 = 5 * 10 * 1280 = 64,000 operations per s.
            # Total: 5 * 64,000 = 320,000. Very fast.
            
            for s in range(5):
                # dp[step][vertex][voltage]
                # Use a 2D array: dp[vertex][voltage] for current step
                dp_prev = [[False] * 64 for _ in range(5)]
                dp_prev[s][0] = True
                
                for step in range(1, 10):
                    dp_curr = [[False] * 64 for _ in range(5)]
                    for v in range(5):
                        for vol in range(64):
                            if dp_prev[v][vol]:
                                for w, eidx in base_adj[v]:
                                    new_vol = (vol + voltages[eidx]) % 64
                                    dp_curr[w][new_vol] = True
                    dp_prev = dp_curr
                    
                    if dp_prev[s][0]:
                        return False
            
            return True
        
        # Search for valid voltages
        best_voltages = None
        random.seed(42)
        
        # Try random search
        max_attempts = 5000
        for i in range(max_attempts):
            if time.time() > deadline:
                break
            voltages = [random.randint(0, 63) for _ in range(10)]
            if check_girth(voltages):
                best_voltages = voltages
                break
        
        # If not found, try local search
        if best_voltages is None:
            # Start from a random point and do hill climbing
            current = [random.randint(0, 63) for _ in range(10)]
            
            # We need a way to measure "closeness" to valid.
            # We can count the number of closed walks of length < 10 with voltage sum 0.
            # Minimize this count.
            
            def count_bad_walks(voltages):
                count = 0
                for s in range(5):
                    dp_prev = [[0] * 64 for _ in range(5)]
                    dp_prev[s][0] = 1
                    for step in range(1, 10):
                        dp_curr = [[0] * 64 for _ in range(5)]
                        for v in range(5):
                            for vol in range(64):
                                if dp_prev[v][vol]:
                                    for w, eidx in base_adj[v]:
                                        new_vol = (vol + voltages[eidx]) % 64
                                        dp_curr[w][new_vol] += dp_prev[v][vol]
                        dp_prev = dp_curr
                        count += dp_prev[s][0]
                return count
            
            best_count = count_bad_walks(current)
            
            for i in range(10000):
                if time.time() > deadline:
                    break
                if best_count == 0:
                    best_voltages = current
                    break
                    
                # Try flipping one voltage
                best_neighbor = None
                best_neighbor_count = best_count
                for eidx in range(10):
                    old_val = current[eidx]
                    # Try all 64 values
                    for new_val in range(64):
                        if new_val == old_val:
                            continue
                        current[eidx] = new_val
                        cnt = count_bad_walks(current)
                        if cnt < best_neighbor_count:
                            best_neighbor_count = cnt
                            best_neighbor = (eidx, new_val)
                    current[eidx] = old_val
                
                if best_neighbor is None:
                    # Stuck, restart
                    current = [random.randint(0, 63) for _ in range(10)]
                    best_count = count_bad_walks(current)
                else:
                    current[best_neighbor[0]] = best_neighbor[1]
                    best_count = best_neighbor_count
            
            if best_count == 0:
                best_voltages = current
        
        if best_voltages is not None:
            edges = []
            for idx, (u, v) in enumerate(base_edges):
                w = best_voltages[idx]
                for g_val in range(group_order):
                    src = u * group_order + g_val
                    dst = v * group_order + ((g_val + w) % group_order)
                    if src < dst:
                        edges.append((src, dst))
            return total_vertices, edges

    # Fallback to LUW
    return luw_construct(k, g)
