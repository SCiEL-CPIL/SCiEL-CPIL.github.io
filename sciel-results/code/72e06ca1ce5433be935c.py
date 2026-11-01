import time
import random
import sys
import numpy as np

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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

    if k == 4 and g == 10:
        # Optimized Voltage Lift of Petersen Graph
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        
        # Precompute constraints: List of (edge_idx, coeff) for each closed walk of length 5-9
        # We need sum(coeff * w[edge_idx]) % 32 != 0
        constraints = []
        
        # Helper to generate walks
        def gen_walks():
            # Petersen graph adjacency
            adj = [[] for _ in range(10)]
            for i, (u, v) in enumerate(pet_edges):
                adj[u].append((v, i))
                adj[v].append((u, i))
                
            for start in range(10):
                # Stack: (current_node, path_edges, path_signs)
                # We only care about closed walks of length 5 to 9
                # To avoid duplicates and backtracking, we can use a simple DFS
                # But we need to count edge traversals with signs.
                # A walk is a sequence of edges.
                # We can just enumerate all walks of length L and check if closed.
                for L in range(5, 10):
                    # DFS
                    def dfs(curr, length, edge_path, sign_path):
                        if length == L:
                            if curr == start:
                                # Aggregate coefficients
                                coeffs = {}
                                for e_idx, sign in zip(edge_path, sign_path):
                                    coeffs[e_idx] = coeffs.get(e_idx, 0) + sign
                                if coeffs:
                                    constraints.append(tuple(sorted(coeffs.items())))
                            return
                        for next_node, e_idx in adj[curr]:
                            # Avoid immediate backtracking? 
                            # The task says "non-backtracking walks".
                            # But for girth, we just need NO cycle shorter than g.
                            # A cycle is a closed walk with no repeated vertices.
                            # However, if a closed walk has a repeated vertex, it contains a shorter cycle.
                            # So checking ALL closed walks of length < g is sufficient and necessary.
                            # Wait, if a closed walk of length L < g exists, it contains a cycle of length <= L < g.
                            # So we must ensure NO closed walk of length < g has voltage 0.
                            # This includes walks that backtrack.
                            # But a walk that backtracks (u-v-u) has length 2. Voltage sum is w - w = 0.
                            # This is always 0. So we must NOT include backtracking walks in our constraints?
                            # No, the condition for a lift to have girth g is that the voltage of EVERY
                            # closed walk of length < g in the base graph is non-zero.
                            # BUT, backtracking walks (u-v-u) have voltage 0 always.
                            # Does that mean the lift always has cycles of length 2?
                            # No. The lift graph is simple. The edge (u,v) in base lifts to edges between fibers.
                            # A "walk" u-v-u in base corresponds to u0 -> v_w -> u_w.
                            # This is a path of length 2 in the lift.
                            # It is a cycle only if u0 == u_w, i.e., w=0.
                            # If w != 0, u0 != u_w, so it's not a cycle.
                            # So we only care about **closed walks in the base** where the net voltage is 0.
                            # Backtracking walks have net voltage 0.
                            # So if we include them, we would require 0 != 0, which is impossible.
                            # Therefore, we must EXCLUDE walks that are "trivially" zero?
                            # No, the standard condition is: The lift has girth >= g iff the voltage of every
                            # **non-backtracking** closed walk of length < g in the base is non-zero?
                            # Actually, any closed walk can be decomposed.
                            # The correct condition is: The voltage of every **closed walk** of length < g
                            # that **does not contain a backtracking step** (i.e., non-backtracking) is non-zero.
                            # Wait, if a walk backtracks, it's not a simple cycle.
                            # A cycle in the lift projects to a closed walk in the base.
                            # If the projected walk has a backtracking step, the cycle in the lift would have a "zig-zag" u-v-u.
                            # In a simple graph, you can't have u-v-u as a cycle (length 2).
                            # So cycles in the lift project to **non-backtracking** closed walks in the base.
                            # So we only need to check **non-backtracking** closed walks.
                            
                            # So in DFS, we should avoid going back to the previous node.
                            pass
                            
                    # Implement DFS with non-backtracking constraint
                    def dfs_nb(curr, prev, length, edge_path, sign_path):
                        if length == L:
                            if curr == start:
                                coeffs = {}
                                for e_idx, sign in zip(edge_path, sign_path):
                                    coeffs[e_idx] = coeffs.get(e_idx, 0) + sign
                                if coeffs:
                                    constraints.append(tuple(sorted(coeffs.items())))
                            return
                        for next_node, e_idx in adj[curr]:
                            if next_node == prev:
                                continue
                            dfs_nb(next_node, curr, length + 1, edge_path + [e_idx], sign_path + [1 if curr == pet_edges[e_idx][0] else -1])
                    
                    dfs_nb(start, -1, 0, [], [])
            
        gen_walks()
        
        # Deduplicate constraints
        constraints = list(set(constraints))
        
        # Backtracking Solver
        n_edges = 15
        voltages = [0] * n_edges
        deadline = time.time() + 60
        
        def solve(idx):
            if time.time() > deadline:
                return False
            if idx == n_edges:
                for con in constraints:
                    total = 0
                    for e_idx, coeff in con:
                        total += voltages[e_idx] * coeff
                    if total % 32 == 0:
                        return False
                return True
            
            # Symmetry breaking: voltages[0] = 1
            if idx == 0:
                voltages[0] = 1
                if solve(1):
                    return True
                return False
            
            for val in range(32):
                voltages[idx] = val
                # Pruning: Check constraints that only involve variables <= idx
                # This is complex to implement efficiently without pre-processing.
                # Given small n, simple backtracking might be fine.
                if solve(idx + 1):
                    return True
            return False

        if solve(0):
            n = 320
            edges = []
            for u, v in pet_edges:
                idx = pet_edges.index((u, v)) if (u, v) in pet_edges else pet_edges.index((v, u))
                # Need correct index
                # Let's map properly
                pass
            # Re-map edges correctly
            edge_to_idx = {}
            for i, (u, v) in enumerate(pet_edges):
                edge_to_idx[(u, v)] = i
                edge_to_idx[(v, u)] = i
                
            for u, v in pet_edges:
                idx = edge_to_idx[(u, v)]
                w = voltages[idx]
                for g_val in range(32):
                    u_idx = u * 32 + g_val
                    v_idx = v * 32 + (g_val + w) % 32
                    edges.append((u_idx, v_idx))
            
            return n, edges

    return luw_construct(k, g)
