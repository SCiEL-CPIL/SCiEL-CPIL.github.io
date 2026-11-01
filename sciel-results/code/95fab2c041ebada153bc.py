import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # If k != 3 or g != 16, fallback to a simple valid construction if possible
    # But the task is specific.
    if k != 3 or g != 16:
        # Fallback to seed logic or simple expansion
        # For now, just return a trivial graph or use the seed logic if we had it.
        # Since we must handle the specific case, we assume k=3, g=16.
        pass

    # Generate non-backtracking closed walks in K3,3 of lengths 4, 6, 8, 10, 12, 14
    # Vertices: 0,1,2 (U), 3,4,5 (V)
    # Edges: (i, j+3) for i in 0..2, j in 0..2
    
    def get_walks(length):
        walks = []
        # Start at 0
        for first_v in [1, 2, 3, 4, 5]:
            if first_v == 0: continue
            # Check edge 0-first_v exists
            # 0 is in U (0-2), first_v must be in V (3-5)
            if 0 <= 0 <= 2 and 3 <= first_v <= 5:
                walk = [0, first_v]
                # DFS
                def dfs(current, prev, path):
                    if len(path) == length + 1:
                        if path[-1] == 0:
                            walks.append(path[:])
                    else:
                        # Neighbors of current
                        if 0 <= current <= 2:
                            neighbors = [3, 4, 5]
                        else:
                            neighbors = [0, 1, 2]
                        
                        for nb in neighbors:
                            if nb == prev:
                                continue
                            path.append(nb)
                            dfs(nb, current, path)
                            path.pop()
                dfs(first_v, 0, walk)
        return walks

    walks_by_len = {}
    for L in range(2, 8):
        length = 2 * L
        walks_by_len[L] = get_walks(length)

    # Search for m
    best_n = 10**9
    best_edges = None
    
    # Try m from 100 to 150
    # We can randomize the order or go sequentially
    for m in range(100, 160):
        if time.time() > deadline:
            break
            
        # Try to find a matrix A
        # A is 3x3
        # We use a simple local search
        # Initialize A
        A = [[random.randint(0, m-1) for _ in range(3)] for _ in range(3)]
        
        # Objective: count violations
        def count_violations(A):
            viol = 0
            for L in range(2, 8):
                for walk in walks_by_len[L]:
                    # Calculate sum
                    s = 0
                    for i in range(0, len(walk)-1, 2):
                        u_idx = walk[i] # 0..2
                        v_idx = walk[i+1] # 3..5
                        # Edge u -> v
                        s += A[u_idx][v_idx - 3]
                        # Next edge is v -> u_next
                        # u_next is walk[i+2]
                        if i + 2 < len(walk):
                            u_next = walk[i+2]
                            # Edge v -> u_next has voltage -A[u_next][v_idx-3]
                            s -= A[u_next][v_idx - 3]
                        else:
                            # Last step back to start (0)
                            # walk[-1] is 0
                            s -= A[0][v_idx - 3]
                    s %= m
                    if s == 0:
                        viol += 1
            return viol

        # Local search
        best_A = A
        best_viol = count_violations(A)
        
        # Iterate
        for _ in range(500):
            if time.time() > deadline:
                break
            if best_viol == 0:
                break
            
            # Pick a random cell to change
            r = random.randint(0, 2)
            c = random.randint(0, 2)
            old_val = A[r][c]
            new_val = random.randint(0, m-1)
            A[r][c] = new_val
            viol = count_violations(A)
            
            if viol < best_viol:
                best_viol = viol
                best_A = [row[:] for row in A]
            else:
                A[r][c] = old_val
                
        if best_viol == 0:
            # Found a valid m
            n = 6 * m
            # Construct edges
            edges = []
            # Vertices: (u, i) -> i, (v, j) -> 3+m+j?
            # Let's map:
            # U vertices: 0..2 * m  -> u_i, x  => i*m + x
            # V vertices: 3*m .. 6*m-1 -> v_j, x => 3*m + j*m + x
            # Edge (u_i, x) to (v_j, x + A[i][j])
            for i in range(3):
                for j in range(3):
                    volt = best_A[i][j]
                    for x in range(m):
                        u_node = i * m + x
                        v_node = 3 * m + j * m + (x + volt) % m
                        edges.append((u_node, v_node))
                        
            best_n = n
            best_edges = edges
            break # Found the smallest m? No, we iterate m increasing, so first valid is smallest.
            
    if best_edges is None:
        # Fallback: Use a known valid graph or a larger m
        # If we didn't find one, we can try a larger m or return a trivial graph
        # But we must return a valid graph.
        # Let's try m=200 with a quick check? Or just return the seed's structure?
        # The seed's structure is complex. 
        # We can construct a graph using the Lazebnik-Ustimenko-Woldar method from the prompt if needed,
        # but it's long. 
        # Alternatively, we can just return a graph that is definitely valid but large.
        # A cycle of length 2*G is girth 2*G.
        # For k=3, we can't just use a cycle.
        # We can use a bipartite double cover of a cycle?
        # Let's just try to find a valid m up to 300 if we have time.
        for m in range(160, 300):
            if time.time() > deadline:
                break
            # Quick random search
            for _ in range(50):
                A = [[random.randint(0, m-1) for _ in range(3)] for _ in range(3)]
                if count_violations(A) == 0:
                    n = 6 * m
                    edges = []
                    for i in range(3):
                        for j in range(3):
                            volt = A[i][j]
                            for x in range(m):
                                u_node = i * m + x
                                v_node = 3 * m + j * m + (x + volt) % m
                                edges.append((u_node, v_node))
                    best_n = n
                    best_edges = edges
                    break
            if best_edges:
                break

    if best_edges is None:
        # Last resort: return a valid graph from a known construction
        # The seed program's logic is available in the prompt.
        # We can copy it here as a fallback.
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

    return best_n, best_edges
