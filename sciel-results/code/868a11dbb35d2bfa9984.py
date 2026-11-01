import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill

    # Fallback: LUW graph
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

    # Specific optimization for k=4, g=10
    if k == 4 and g == 10:
        # Base graph: K5
        # Vertices 0..4
        base_edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        num_base_vertices = 5
        num_edges = len(base_edges)
        group_order = 64  # Z2^6
        
        # Map edge (u, v) to index
        edge_idx = {}
        for i, (u, v) in enumerate(base_edges):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i
        
        # Adjacency list for K5
        base_adj = [[] for _ in range(num_base_vertices)]
        for u, v in base_edges:
            base_adj[u].append(v)
            base_adj[v].append(u)

        # Precompute all non-backtracking closed walks of length L < 10 in K5
        # A walk is a sequence of vertices v0, v1, ..., vL=v0
        # Non-backtracking: v_{i+1} != v_{i-1} for all i
        # We represent a walk by the sequence of edge indices
        constraints = [] # List of lists of edge indices
        
        def find_walks(start_v, length):
            # DFS to find non-backtracking closed walks of given length
            # State: (current_v, prev_v, path_edges)
            # Start: (start_v, None, [])
            # We only care about walks that return to start_v at exactly 'length'
            # To avoid duplicates, we can restrict start_v < end_v? No, it's closed.
            # We just generate all and add if valid.
            # For efficiency, we can prune if remaining steps < 1.
            
            stack = [(start_v, -1, [])]
            while stack:
                v, prev, path = stack.pop()
                if len(path) == length:
                    if v == start_v:
                        # Check non-backtracking property is maintained in generation
                        constraints.append(path[:])
                    continue
                
                for u in base_adj[v]:
                    if u == prev:
                        continue
                    # Pruning: if we are far from start, we might not make it back?
                    # In K5, diameter is 1. So any vertex is adjacent to start.
                    # So we can always return in 1 step.
                    # If remaining steps is 1, we must go to start_v.
                    # If remaining steps > 1, we can go anywhere.
                    stack.append((u, v, path + [edge_idx[(v, u)]]))

        # Generate constraints for L = 3 to 9
        # Note: Walks of length 2 are back-and-forth, excluded by non-backtracking.
        for L in range(3, 10):
            for sv in range(num_base_vertices):
                find_walks(sv, L)
        
        # Deduplicate constraints? 
        # A walk and its reverse have the same xor sum. 
        # Also, cyclic shifts are the same sum.
        # We don't strictly need to deduplicate for the check, but it reduces work.
        # Let's just keep them all. There aren't that many.
        
        # Local Search
        # Variables: voltages[0..9]
        # Objective: Minimize number of violated constraints
        # Constraint: xor(v[e] for e in walk) != 0
        
        best_voltages = None
        
        def check_voltages(voltages):
            for walk in constraints:
                s = 0
                for e in walk:
                    s ^= voltages[e]
                if s == 0:
                    return False
            return True

        def count_violations(voltages):
            count = 0
            for walk in constraints:
                s = 0
                for e in walk:
                    s ^= voltages[e]
                if s == 0:
                    count += 1
            return count

        # Random restarts with local search
        for restart in range(10000):
            if time.time() > deadline:
                break
            
            # Initialize randomly
            voltages = [random.randint(0, 63) for _ in range(num_edges)]
            
            # Local Search: Simulated Annealing or just Hill Climbing
            # Since the landscape is likely rugged, Hill Climbing might get stuck.
            # Let's try a few steps of random perturbation + best neighbor.
            
            current_v = voltages
            current_cost = count_violations(current_v)
            
            if current_cost == 0:
                best_voltages = current_v
                break
            
            # Try to improve
            improved = True
            while improved:
                improved = False
                # Try flipping one bit in one voltage? Or changing one voltage?
                # Changing one voltage to a random new value is a larger step.
                # Let's try changing one voltage to a random value.
                for e in range(num_edges):
                    if time.time() > deadline:
                        break
                    old_val = current_v[e]
                    # Try a few random values
                    for _ in range(5):
                        new_val = random.randint(0, 63)
                        if new_val == old_val:
                            continue
                        current_v[e] = new_val
                        new_cost = count_violations(current_v)
                        if new_cost < current_cost:
                            current_cost = new_cost
                            improved = True
                            if new_cost == 0:
                                break
                    current_v[e] = old_val
                    if current_cost == 0:
                        break
                
                if current_cost == 0:
                    break
            
            if current_cost == 0:
                best_voltages = current_v
                break

        if best_voltages:
            total_vertices = num_base_vertices * group_order
            edges = []
            for idx, (u, v) in enumerate(base_edges):
                w = best_voltages[idx]
                for g_val in range(group_order):
                    src = u * group_order + g_val
                    dst = v * group_order + ((g_val + w) % group_order)
                    if src < dst:
                        edges.append((src, dst))
            
            # Final verification (optional, but good for safety)
            # The search guarantees no short cycles in the base lift sense.
            # Return the graph
            return total_vertices, edges

    # Fallback to LUW
    return luw_construct(k, g)
