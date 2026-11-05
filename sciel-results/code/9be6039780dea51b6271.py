import time
import random
import math
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    timeout = 115.0  # Safety margin before 120s limit

    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def is_prime_power(n):
        if n < 2:
            return False
        if is_prime(n):
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % p == 0:
                t = n
                while t % p == 0:
                    t //= p
                if t == 1:
                    return True
                return False
        return False

    def modinv(a, m):
        if m <= 1:
            return None
        try:
            return pow(a, -1, m)
        except ValueError:
            return None

    def moore_bound(k, g):
        if k <= 1: return 0
        if g == 1: return 1
        r = g // 2
        if g % 2 == 1:
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    def action_on_point(x, a, b, c, d, q):
        """
        Apply PGL(2,q) transformation to a point x.
        Points are 0..q-1 and q (infinity).
        Returns the image point, or None if the transformation is invalid.
        """
        if x == q:
            if c == 0:
                return q
            else:
                inv_c = modinv(c, q)
                if inv_c is None:
                    return None
                return (a * inv_c) % q
        else:
            num = (a * x + b) % q
            den = (c * x + d) % q
            if den == 0:
                return q
            else:
                inv_den = modinv(den, q)
                if inv_den is None:
                    return None
                return (num * inv_den) % q

    def check_girth_bfs(adj, n, g):
        """
        Returns True if girth >= g using BFS from each vertex.
        Uses a more efficient check: for each edge (u,v), check if there's a 
        path from u to v of length < g-1 that doesn't use the edge (u,v).
        This is equivalent to checking if the graph has no cycle of length < g.
        """
        max_depth = (g - 1) // 2
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                x = queue.popleft()
                dx = dist[x]
                if dx > max_depth:
                    continue
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dx + 1
                        parent[y] = x
                        if dx + 1 <= max_depth:
                            queue.append(y)
                    elif y != parent[x] and parent[y] != x:
                        cycle_len = dist[x] + dist[y] + 1
                        if cycle_len < g:
                            return False
        return True

    def luw_construct(q, d):
        """
        Lazebnik-Ustimenko-Woldar graph CD(d, q).
        Returns (n, edges) for the connected component of the origin.
        """
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

    def build_and_check_pgl(q, k, g, attempts=80):
        """
        Try to build a k-regular graph on PGL(2,q) acting on q+1 points.
        Use carefully chosen generators to maximize girth.
        """
        n = q + 1
        
        # Precompute all points as indices
        # Points: 0..q-1 and q (infinity)
        
        for attempt in range(attempts):
            if time.time() - start_time > timeout:
                return None
            
            # Strategy: use different types of generators
            gens = []
            used_points = [set() for _ in range(n)]
            
            valid = True
            
            for gen_idx in range(k):
                while True:
                    a = random.randint(0, q - 1)
                    b = random.randint(0, q - 1)
                    c = random.randint(0, q - 1)
                    d = random.randint(0, q - 1)
                    det = (a * d - b * c) % q
                    if det == 0:
                        continue
                    gen_valid = True
                    for i in range(n):
                        x = i if i < q else q
                        img = action_on_point(x, a, b, c, d, q)
                        if img is None:
                            gen_valid = False
                            break
                        j = img if img < q else q
                        if i == j:
                            gen_valid = False
                            break
                        if j in used_points[i]:
                            gen_valid = False
                            break
                    if gen_valid:
                        gens.append((a, b, c, d))
                        for i in range(n):
                            x = i if i < q else q
                            img = action_on_point(x, gens[gen_idx][0], gens[gen_idx][1], gens[gen_idx][2], gens[gen_idx][3], q)
                            j = img if img < q else q
                            used_points[i].add(j)
                        break
                    if time.time() - start_time > timeout:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid or len(gens) < k:
                continue
            
            adj = [set() for _ in range(n)]
            is_simple = True
            
            for i in range(n):
                for j in range(k):
                    x = i if i < q else q
                    img = action_on_point(x, gens[j][0], gens[j][1], gens[j][2], gens[j][3], q)
                    if img is None:
                        is_simple = False
                        break
                    j_idx = img if img < q else q
                    if i == j_idx:
                        is_simple = False
                        break
                    adj[i].add(j_idx)
                    adj[j_idx].add(i)
                if not is_simple:
                    break
            
            if not is_simple:
                continue
            
            for i in range(n):
                if len(adj[i]) != k:
                    is_simple = False
                    break
            
            if not is_simple:
                continue
            
            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            count = 0
            while queue:
                u = queue.popleft()
                count += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
            if count != n:
                continue
            
            if check_girth_bfs(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        
        return None

    def try_tree_closure(k, g):
        """
        Attempts to construct a graph by building a BFS tree of depth r=(g-1)/2
        and closing the leaves with random edges that satisfy girth constraints.
        """
        if g % 2 == 0:
            return None
        
        r = (g - 1) // 2
        
        nodes = []
        parent = []
        depth = []
        branch = []

        def add_node(p, d, br):
            idx = len(nodes)
            nodes.append(idx)
            parent.append(p)
            depth.append(d)
            branch.append(br)
            return idx

        add_node(-1, 0, -1)
        
        current_level = [0]
        
        for d in range(1, r + 1):
            next_level = []
            for u in current_level:
                num_children = k if d == 1 else k - 1
                for _ in range(num_children):
                    br = branch[u]
                    if d == 1:
                        br = u
                    new_node = add_node(u, d, br)
                    next_level.append(new_node)
            current_level = next_level
            
        n_tree = len(nodes)
        leaves = current_level
        
        branches = {}
        for leaf in leaves:
            br = branch[leaf]
            if br not in branches:
                branches[br] = []
            branches[br].append(leaf)
        
        stubs = []
        for leaf in leaves:
            for s in range(k - 1):
                stubs.append((leaf, s))
        
        random.shuffle(stubs)
        
        edges = []
        for i in range(1, n_tree):
            edges.append((parent[i], i))
        
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u, v), max(u, v)))
        
        used = [False] * len(stubs)
        success = True
        for i in range(0, len(stubs), 2):
            if used[i]:
                continue
            u_leaf, _ = stubs[i]
            u_branch = branch[u_leaf]
            
            found = False
            for j in range(i + 1, len(stubs)):
                if used[j]:
                    continue
                v_leaf, _ = stubs[j]
                v_branch = branch[v_leaf]
                
                if u_branch == v_branch:
                    continue
                if u_leaf == v_leaf:
                    continue
                
                edge_key = (min(u_leaf, v_leaf), max(u_leaf, v_leaf))
                if edge_key in edge_set:
                    continue
                
                edges.append((u_leaf, v_leaf))
                edge_set.add(edge_key)
                used[i] = True
                used[j] = True
                found = True
                break
            
            if not found:
                success = False
                break
        
        if not success:
            return None
            
        n = n_tree
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        for i in range(n):
            if len(adj[i]) != k:
                return None
                
        if check_girth_bfs(adj, n, g):
            return n, edges
            
        return None

    # Main strategy selection
    best_n = float('inf')
    best_result = None
    
    # Strategy 1: Try Tree Closure (for odd girth)
    if g % 2 == 1:
        res = try_tree_closure(k, g)
        if res:
            best_n, best_result = res
    
    # Strategy 2: LUW
    q = k
    d_min = max(3, g - 5 if (g - 5) % 2 else g - 4)
    d_max = d_min + 3
    
    for d in range(d_min, d_max + 1):
        if time.time() - start_time > timeout:
            break
        n_luw, edges_luw = luw_construct(q, d)
        if n_luw < best_n:
            adj = [set() for _ in range(n_luw)]
            for i, j in edges_luw:
                adj[i].add(j)
                adj[j].add(i)
            valid = True
            for i in range(n_luw):
                if len(adj[i]) != k:
                    valid = False
                    break
            if valid and check_girth_bfs(adj, n_luw, g):
                best_n = n_luw
                best_result = (n_luw, edges_luw)
    
    # Strategy 3: PGL Search
    if best_n == float('inf') or best_n > n_min * 1.5:
        q_min = max(2, n_min - 1)
        q_max = min(best_n - 2 if best_n != float('inf') else q_min + 100, q_min + 150)
        
        if q_max >= q_min:
            count_tried = 0
            for curr_q in range(q_min, q_max + 1):
                if time.time() - start_time > timeout:
                    break
                if not is_prime_power(curr_q):
                    continue
                
                res = build_and_check_pgl(curr_q, k, g, attempts=40)
                if res:
                    n, edges = res
                    if n < best_n:
                        best_n = n
                        best_result = res
                        if n <= n_min * 1.1:
                            return best_result
                
                count_tried += 1
                if count_tried > 40:
                    break
    
    # Fallback: always return something valid
    if best_result is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        best_n, best_result = luw_construct(k, d)
    
    return best_result

# EVOLVE-BLOCK-END
