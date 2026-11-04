import time
import random
import itertools
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def check_girth(n, edges, g, time_limit=None):
        """Check if graph has girth >= g using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        min_cycle = float('inf')
        for s in range(n):
            if time_limit and time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    def build_luw_graph(q, d, rule, time_limit=None):
        """Build LUW graph with given parameters."""
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

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
            if time_limit and time.time() > time_limit:
                return None
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))

        return len(order), edges

    def generate_rules(d, variant=0):
        base_rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        if variant == 0:
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                base_rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        elif variant == 1:
            for c in range(5, d + 1):
                if c % 2 == 0:
                    base_rule[c] = (1, c - 2)
                else:
                    base_rule[c] = (c - 1, 1)
        elif variant == 2:
            for c in range(5, d + 1):
                if c % 3 == 0:
                    base_rule[c] = (1, c - 3)
                elif c % 3 == 1:
                    base_rule[c] = (c - 1, 1)
                else:
                    base_rule[c] = (c - 2, 1)
        elif variant == 3:
            for c in range(5, d + 1):
                offset = 1 + (c % 2)
                base_rule[c] = (1, c - offset)
        elif variant == 4:
            for c in range(5, d + 1):
                if c % 4 == 0:
                    base_rule[c] = (1, c - 1)
                elif c % 4 == 1:
                    base_rule[c] = (c - 1, 1)
                elif c % 4 == 2:
                    base_rule[c] = (1, c - 2)
                else:
                    base_rule[c] = (c - 2, 1)
        else:
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                base_rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return base_rule

    def luw_construct(k, g):
        q = k
        d = max(3, (g - 2) // 2)
        best_result = None
        best_n = float('inf')
        for variant in range(5):
            if time.time() > deadline - 15:
                break
            rule = generate_rules(d, variant)
            result = build_luw_graph(q, d, rule, time_limit=deadline - 15)
            if result is None:
                continue
            n, edges = result
            if n >= best_n:
                continue
            girth_check = check_girth(n, edges, g, time_limit=deadline - 10)
            if girth_check == -1:
                continue
            if girth_check >= g:
                best_result = (n, edges)
                best_n = n
        if best_result is None:
            for d_try in range(d + 1, d + 10):
                if time.time() > deadline - 10:
                    break
                for variant in range(3):
                    rule = generate_rules(d_try, variant)
                    result = build_luw_graph(q, d_try, rule, time_limit=deadline - 10)
                    if result is None:
                        continue
                    n, edges = result
                    girth_check = check_girth(n, edges, g, time_limit=deadline - 5)
                    if girth_check == -1:
                        continue
                    if girth_check >= g:
                        best_result = (n, edges)
                        break
                if best_result is not None:
                    break
        if best_result is None:
            rule = generate_rules(d, 0)
            result = build_luw_graph(q, d, rule)
            if result is not None:
                best_result = result
        return best_result

    def voltage_lift_k33(m, voltages, time_limit=None):
        """
        Build a 3-regular graph by voltage lift of K_{3,3} over Z_m.
        
        K_{3,3} has vertices A0,A1,A2 (left) and B0,B1,B2 (right).
        Edges: Ai-Bj for all i,j in {0,1,2}.
        
        Voltage assignment: each edge (Ai, Bj) gets a voltage v_{i,j} in Z_m.
        The lift has 6m vertices: (side, base, coset) where side in {0,1},
        base in {0,1,2}, coset in Z_m.
        
        Vertex (0, i, a) connects to (1, j, a + v_{i,j} mod m) for all j.
        Vertex (1, j, b) connects to (0, i, b - v_{i,j} mod m) for all i.
        """
        n = 6 * m
        edges = []
        seen = set()
        
        for i in range(3):
            for j in range(3):
                v = voltages[i * 3 + j]
                for a in range(m):
                    # Edge from (0,i,a) to (1,j, a+v mod m)
                    u = i * m + a
                    w = 3 * m + j * m + ((a + v) % m)
                    if u < w:
                        edges.append((u, w))
                    else:
                        edges.append((w, u))
        
        return n, edges

    def girth_of_lift(m, voltages, g_target, time_limit=None):
        """
        Compute the girth of the voltage lift of K_{3,3} over Z_m.
        Uses BFS in the non-backtracking sense to find shortest cycle.
        
        For a voltage graph, cycles correspond to closed walks in the base
        whose voltage sum is 0 mod m. We can compute girth by BFS in the
        lift graph directly, but that's expensive for large m.
        
        Alternative: BFS from each vertex in the lift, tracking distance.
        For 6m vertices with degree 3, BFS from each vertex is O(6m * 6m) = O(36m^2).
        For m up to ~200, that's ~1.4M operations per BFS, 6m BFS calls = ~864M total.
        That's too slow. Let's use a smarter approach.
        
        Actually, let's just do BFS in the lift but with early termination.
        """
        n = 6 * m
        # Build adjacency
        adj = [[] for _ in range(n)]
        for i in range(3):
            for j in range(3):
                v = voltages[i * 3 + j]
                for a in range(m):
                    u = i * m + a
                    w = 3 * m + j * m + ((a + v) % m)
                    adj[u].append(w)
                    adj[w].append(u)
        
        min_cycle = float('inf')
        for s in range(n):
            if time_limit and time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g_target:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g_target:
                return min_cycle
        return min_cycle

    def fast_girth_check(m, voltages, g_target, time_limit=None):
        """
        Faster girth check for K_{3,3} voltage lift.
        Exploits the structure: the base is K_{3,3} which has girth 4.
        A cycle in the lift of length L corresponds to a closed walk of length L
        in K_{3,3} whose total voltage is 0 mod m.
        
        For girth >= g, we need no closed walk of length < g with voltage 0 mod m.
        The shortest closed walks in K_{3,3}:
        - Length 4: Ai-Bj-Ak-Bl-Ai (i!=k or j!=l... actually any 4-cycle)
          In K_{3,3}, 4-cycles are Ai-Bj-Ak-Bl-Ai where i!=k and j!=l.
          Voltage: v_{i,j} - v_{i,l} + v_{k,l} - v_{k,j} (depending on orientation)
          Actually let me think more carefully.
        
        Walk Ai -> Bj -> Ak -> Bl -> Ai:
          Ai to Bj: +v_{i,j}
          Bj to Ak: -v_{k,j}  (going from B to A, voltage is negated)
          Ak to Bl: +v_{k,l}
          Bl to Ai: -v_{i,l}
          Total: v_{i,j} - v_{k,j} + v_{k,l} - v_{i,l} = 0 mod m
        
        - Length 6: various 6-cycles in K_{3,3}
        
        This approach is complex. Let's just use direct BFS on the lift for small m,
        and for larger m, use the voltage sum approach.
        """
        # For m <= 100, direct BFS is feasible
        if m <= 120:
            return girth_of_lift(m, voltages, g_target, time_limit)
        
        # For larger m, check voltage sums for short closed walks
        # 4-cycles: for all i!=k, j!=l: v_{i,j} - v_{k,j} + v_{k,l} - v_{i,l} != 0 mod m
        for i in range(3):
            for k in range(3):
                if i == k:
                    continue
                for j in range(3):
                    for l in range(3):
                        if j == l:
                            continue
                        total = (voltages[i*3+j] - voltages[k*3+j] + voltages[k*3+l] - voltages[i*3+l]) % m
                        if total == 0:
                            return 4
        
        # 6-cycles in K_{3,3}: Ai-Bj-Ak-Bl-Am-Bn-Ai with distinct conditions
        # This is getting complex. Let's just do BFS for m up to 200.
        if m <= 200:
            return girth_of_lift(m, voltages, g_target, time_limit)
        
        # For larger m, we'll need a smarter approach
        # For now, return -1 (unknown) to indicate we can't verify
        return -1

    def voltage_lift_optimize(k, g, time_limit):
        """
        Optimize voltage assignment for K_{3,3} lift over Z_m using simulated annealing.
        Goal: find m and voltages such that the lift has girth >= g and is as small as possible.
        """
        best_result = None
        best_n = float('inf')
        
        # Try different values of m
        # The lift has 6m vertices. We want girth >= g.
        # Smaller m = fewer vertices, but harder to achieve high girth.
        # Start with small m and increase.
        
        for m in range(16, 300):
            if time.time() > time_limit - 20:
                break
            
            n = 6 * m
            if n >= best_n:
                break
            
            # Try random voltage assignments with simulated annealing
            best_voltages = None
            best_girth = 0
            current_voltages = [random.randint(0, m-1) for _ in range(9)]
            
            # Simulated annealing
            temp = 10.0
            cooling = 0.95
            max_iters = 500
            
            for it in range(max_iters):
                if time.time() > time_limit - 20:
                    break
                
                # Evaluate current
                girth_val = fast_girth_check(m, current_voltages, g, time_limit=time_limit - 10)
                if girth_val == -1:
                    # Can't verify, try next m
                    break
                if girth_val > best_girth:
                    best_girth = girth_val
                    best_voltages = current_voltages[:]
                    if best_girth >= g:
                        # Found valid assignment!
                        n_edges_result = voltage_lift_k33(m, current_voltages)
                        if n_edges_result[0] < best_n:
                            best_n = n_edges_result[0]
                            best_result = n_edges_result
                        break
                
                # Perturbation: change one or two voltages
                if random.random() < 0.5:
                    idx = random.randint(0, 8)
                    current_voltages[idx] = random.randint(0, m-1)
                else:
                    idx1 = random.randint(0, 8)
                    idx2 = random.randint(0, 8)
                    current_voltages[idx1] = random.randint(0, m-1)
                    current_voltages[idx2] = random.randint(0, m-1)
                
                temp *= cooling
            
            if best_girth >= g and best_result is None:
                n_edges_result = voltage_lift_k33(m, best_voltages)
                if n_edges_result[0] < best_n:
                    best_n = n_edges_result[0]
                    best_result = n_edges_result
            
            if best_result is not None:
                # Found a valid graph, try to improve with slightly smaller m
                # But since we're iterating m in increasing order, we already have the smallest
                break
        
        return best_result

    if k != 3:
        return luw_construct(k, g)
    
    # For k=3, try voltage lift first (potentially much smaller)
    voltage_result = voltage_lift_optimize(k, g, time_limit=deadline - 20)
    
    # Get LUW baseline as fallback
    luw_result = luw_construct(k, g)
    
    # Choose the better result
    if voltage_result is not None and luw_result is not None:
        if voltage_result[0] < luw_result[0]:
            return voltage_result
        else:
            return luw_result
    elif voltage_result is not None:
        return voltage_result
    elif luw_result is not None:
        return luw_result
    else:
        return 0, []
