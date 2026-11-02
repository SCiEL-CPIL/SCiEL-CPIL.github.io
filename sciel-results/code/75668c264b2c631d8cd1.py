import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 115  # safety margin before 120s kill
    
    # Known cages for small parameters
    known_cages = {
        (3, 5): 10,  # Petersen
        (3, 6): 14,  # Heawood
        (3, 7): 24,  # Tutte-Coxeter
        (3, 8): 24,  # McGee
        (3, 9): 30,  # (3,9) cage
        (4, 5): 21,  # (4,5) cage
        (5, 5): 12,  # Icosahedron? No, Icosahedron is (5,5) with 12 vertices. Girth 5.
    }
    
    # If we know the exact graph, we could return it, but we only know n.
    # The task asks for (n, edges).
    # I need to construct the edges for these known cages if I want to use them.
    # For now, I will only use the LUW construction for the main case, 
    # but I will try to beat LUW for (5,9) if possible.
    
    # Helper to generate a random k-regular graph on n vertices
    def random_regular_graph(n, k):
        if n * k % 2 != 0:
            return None
        # Configuration model
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        random.shuffle(half_edges)
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                return None
            if v in adj[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
        # Check for multi-edges
        for u in range(n):
            if len(adj[u]) != k:
                return None
        # Convert to edge list
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return n, edges

    # Helper to calculate girth
    def get_girth(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_girth = float('inf')
        for start in range(n):
            # BFS/DFS with depth limit
            # State: (node, parent, depth)
            # We want to find the shortest cycle.
            # This is equivalent to finding the shortest path from a neighbor of start back to start.
            # Or simpler: BFS from start, track parent. If we hit a visited node not parent, cycle found.
            # But we need the *shortest* cycle in the whole graph.
            # For small girth (9), we can do non-backtracking walks.
            
            # Non-backtracking walk from start
            # Depth up to g//2
            max_depth = g // 2
            visited = {start: -1}
            queue = deque([(start, -1, 0)])
            while queue:
                node, parent, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if neighbor == start and depth + 1 >= 3:
                        # Found a cycle
                        cycle_len = depth + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                    if neighbor not in visited or visited[neighbor] > depth + 1:
                        # We only care about shortest paths for girth?
                        # Actually, we just need to detect if a cycle of length < g exists.
                        # If we find any cycle of length < g, girth is < g.
                        pass
            # This BFS is not efficient for finding the exact girth if it's large.
            # But we only care if girth < g.
            # Let's just check if there is ANY cycle of length < g.
            
        return min_girth

    # Efficient Girth Check: Returns True if girth >= g
    def has_girth_at_least(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each vertex, do a DFS/BFS to depth g//2
        # If we find a cycle of length < g, return False.
        # Cycle length L < g.
        # Non-backtracking walks of length L.
        
        for start in range(n):
            # State: (current_node, parent_node, depth)
            # We track visited nodes in the current walk to detect cycles?
            # No, non-backtracking prevents immediate backtracking.
            # A cycle of length L corresponds to a non-backtracking walk of length L from start to start.
            # We can just do a BFS on the state space (node, parent).
            
            visited = set()
            queue = deque()
            for neighbor in adj[start]:
                queue.append((neighbor, start, 1))
            
            while queue:
                node, parent, depth = queue.popleft()
                if depth == g - 1:
                    # If we can reach start from here in 1 step, cycle length is g.
                    # But we are looking for < g.
                    # So if depth < g-1 and we see start, we have a cycle of length depth+1.
                    if node == start:
                        # This would be a cycle of length depth+1.
                        # But we started from neighbors of start.
                        # If we return to start at depth d, cycle length is d+1.
                        # We want to know if any d+1 < g.
                        # So if we ever see start at depth < g-1, we have a short cycle.
                        return False
                    continue
                
                if (node, parent) in visited:
                    continue
                visited.add((node, parent))
                
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if neighbor == start:
                        # Cycle found of length depth + 1
                        if depth + 1 < g:
                            return False
                    else:
                        queue.append((neighbor, node, depth + 1))
        return True

    # LUW Construction
    def luw_construct(k, g):
        q = k
        d = g - 5
        if d % 2 == 0:
            d += 1
        d = max(3, d)

        if q == 4:
            add_table = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
            mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            sub_table = add_table
        else:
            add_table = [[(i+j) % q for j in range(q)] for i in range(q)]
            mul_table = [[(i*j) % q for j in range(q)] for i in range(q)]
            sub_table = [[(i-j) % q for j in range(q)] for i in range(q)]

        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        rule_a[2], rule_b[2] = 1, 1
        rule_a[3], rule_b[3] = 2, 1
        rule_a[4], rule_b[4] = 1, 2
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            pairs = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
            rule_a[c], rule_b[c] = pairs[r]

        q_pow = [1] * (d + 1)
        for i in range(1, d + 1):
            q_pow[i] = q_pow[i-1] * q
        base = q_pow[d]  # q^d

        trans = [[None] * base for _ in range(2)]

        for side in range(2):
            for x_idx in range(base):
                x = [0] * (d + 1)
                rem = x_idx
                for i in range(d, 0, -1):
                    x[i] = rem % q
                    rem //= q

                trans[side][x_idx] = []
                for first in range(q):
                    y = [0] * (d + 1)
                    y[1] = first
                    for c in range(2, d + 1):
                        a = rule_a[c]
                        b = rule_b[c]
                        if side == 0:
                            y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
                        else:
                            y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]

                    y_idx = 0
                    for i in range(1, d + 1):
                        y_idx = y_idx * q + y[i]
                    trans[side][x_idx].append(y_idx)

        start_id = 0
        index = {start_id: 0}
        order = [start_id]
        edges = []

        queue = deque([0])

        while queue:
            if time.time() > deadline - 3:
                break

            vi = queue.popleft()
            v = order[vi]
            side = v // base
            x_idx = v % base

            neighbors_idx = trans[side][x_idx]

            for first in range(q):
                y_idx = neighbors_idx[first]
                other = (1 - side) * base + y_idx

                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                    queue.append(len(order) - 1)

                if side == 0:
                    edges.append((vi, index[other]))

        return len(order), edges

    # Main Logic
    if k == 2:
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges

    # Try to find a better graph than LUW for (5,9) or other cases
    # LUW gives n_luw
    n_luw, edges_luw = luw_construct(k, g)
    
    # If we have time, try to search for n < n_luw
    # This is only viable for small n.
    if time.time() < deadline - 10:
        # Search for n in [n_luw - 50, n_luw - 1]
        # This is a long shot for (5,9) as 1250 -> 1200 is a big gap.
        # But we can try a few random restarts.
        found = False
        for n in range(n_luw - 10, n_luw):
            if time.time() > deadline - 5:
                break
            for _ in range(5): # 5 restarts
                res = random_regular_graph(n, k)
                if res:
                    n_try, edges_try = res
                    if has_girth_at_least(n_try, edges_try, g):
                        return n_try, edges_try
        # If not found, return LUW
        return n_luw, edges_luw

    return n_luw, edges_luw
