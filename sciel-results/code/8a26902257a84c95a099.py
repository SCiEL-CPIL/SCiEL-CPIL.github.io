import time
import random
import math
import sys

def _luw_construct(k, g):
    # Seed logic - LUW construction
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

def _check_girth(adj, n, min_girth):
    """Check girth >= min_girth using BFS. Returns True if girth is sufficient."""
    max_depth = min_girth // 2 - 1
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        while qi < len(queue):
            node = queue[qi]
            qi += 1
            d = dist[node]
            if d >= max_depth:
                continue
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = d + 1
                    queue.append(nb)
                elif dist[nb] <= d:
                    if d + dist[nb] + 1 < min_girth:
                        return False
    return True

def _count_4cycles(adj, n):
    """Count 4-cycles in a bipartite graph."""
    half = n // 2
    count = 0
    for u in range(half):
        au = adj[u]
        for v in range(u + 1, half):
            common = len(au & adj[v])
            if common >= 2:
                count += common * (common - 1) // 2
    return count

def _count_6cycles(adj, n):
    """Count 6-cycles in a bipartite graph A(0..half-1), B(half..n-1)."""
    half = n // 2
    count = 0
    adj_lists = [list(s) for s in adj]
    for a1 in range(half):
        for a2 in range(a1 + 1, half):
            common_b = adj_lists[a1] & adj_lists[a2]
            if not common_b:
                continue
            for b1 in common_b:
                for b2 in adj_lists[a2]:
                    if b2 == b1:
                        continue
                    for a3 in adj_lists[b2]:
                        if a3 < half and a3 != a1 and a3 != a2:
                            common_b3 = adj_lists[a3] & adj_lists[a1]
                            common_b3.discard(b1)
                            count += len(common_b3)
    return count

def _short_cycle_score(adj, n, g):
    """
    Surrogate objective: weighted sum of short cycles.
    Lower is better. If girth >= g, score is 0 (or very small).
    """
    s4 = _count_4cycles(adj, n)
    if s4 > 0:
        return s4 * 10000
    s6 = _count_6cycles(adj, n)
    if s6 > 0:
        return s6 * 100
    # Check 8-cycles via BFS (expensive, only if no 4 or 6 cycles)
    if not _check_girth(adj, n, g):
        return 10
    return 0

def _voltage_lift(base_adj, base_n, voltage, group_order, k):
    """
    Lift a base graph using voltage assignment.
    base_adj: list of (neighbor, voltage) pairs for each base vertex
    voltage: dict mapping (u, v) -> voltage label in Z_m
    Returns the lifted graph's adjacency list.
    """
    m = group_order
    lifted_n = base_n * m
    adj = [set() for _ in range(lifted_n)]
    
    for u in range(base_n):
        for v, volt in base_adj[u]:
            for s in range(m):
                # Edge from (u, s) to (v, (s + volt) % m)
                u_lift = u * m + s
                v_lift = v * m + ((s + volt) % m)
                adj[u_lift].add(v_lift)
                adj[v_lift].add(u_lift)
    
    # Check for multi-edges (shouldn't happen with proper voltage assignment)
    for i in range(lifted_n):
        if len(adj[i]) != k:
            return None
    return adj

def _find_voltage_assignment(base_adj, base_n, k, g, deadline):
    """
    Search for a voltage assignment to a base graph that produces girth >= g.
    Uses simulated annealing on the voltage labels.
    """
    # For k=4, g=10, try small base graphs
    # The base graph should be 4-regular
    # A good base: 8-vertex 4-regular graph (cube-like) or similar
    
    # Try different base graph sizes
    for base_size in [8, 12, 16, 20, 24]:
        if time.time() > deadline:
            break
        
        # Generate a random 4-regular base graph
        base_graphs = _generate_base_graphs(base_size, k, 5)
        
        for base in base_graphs:
            if time.time() > deadline:
                break
            result = _optimize_voltage(base, base_size, k, g, deadline)
            if result is not None:
                return result
    
    return None

def _generate_base_graphs(size, k, num):
    """Generate random k-regular graphs of given size."""
    graphs = []
    for _ in range(num):
        adj = [set() for _ in range(size)]
        # Configuration model
        halfends = []
        for i in range(size):
            halfends.extend([i] * k)
        random.shuffle(halfends)
        
        for i in range(0, size * k, 2):
            u, v = halfends[i], halfends[i + 1]
            if u != v and v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
        
        # Check regularity
        valid = True
        for i in range(size):
            if len(adj[i]) != k:
                valid = False
                break
        if valid:
            # Convert to list of (neighbor, voltage) with initial random voltages
            base_adj = []
            for i in range(size):
                neighbors = list(adj[i])
                random.shuffle(neighbors)
                # Only keep edges where i < neighbor to avoid double counting
                base_adj.append([(v, 0) for v in neighbors if v > i])
            # Actually, let's store all directed edges
            base_adj = []
            for i in range(size):
                base_adj.append([(v, 0) for v in adj[i]])
            graphs.append(base_adj)
    
    return graphs

def _optimize_voltage(base_adj, base_n, k, g, deadline):
    """
    Optimize voltage assignment using simulated annealing.
    """
    # Determine group order: try powers of 2
    # For girth 10, we need the lift to have no cycles shorter than 10
    # The base graph has some cycles; their voltages must be "long enough"
    
    # Try different group orders
    for m in [4, 8, 16, 32, 64]:
        if time.time() > deadline:
            break
        
        lifted_n = base_n * m
        if lifted_n > 400:
            continue
        
        # Initialize random voltage assignment
        voltage = {}
        for u in range(base_n):
            for v, _ in base_adj[u]:
                if u < v:
                    voltage[(u, v)] = random.randint(0, m - 1)
        
        best_voltage = dict(voltage)
        best_score = float('inf')
        
        # Simulated annealing
        temp = 1.0
        cooling = 0.999
        
        for iteration in range(5000):
            if time.time() > deadline - 2:
                break
            
            # Pick a random edge to change voltage
            edges = [(u, v) for u in range(base_n) for v, _ in base_adj[u] if u < v]
            if not edges:
                break
            
            u, v = random.choice(edges)
            old_volt = voltage[(u, v)]
            new_volt = random.randint(0, m - 1)
            if new_volt == old_volt:
                continue
            voltage[(u, v)] = new_volt
            
            # Build lifted graph and compute score
            adj = [set() for _ in range(lifted_n)]
            for uu in range(base_n):
                for vv, _ in base_adj[uu]:
                    volt = voltage.get((min(uu, vv), max(uu, vv)), 0)
                    if uu > vv:
                        volt = (m - volt) % m  # Reverse edge
                    for s in range(m):
                        u_lift = uu * m + s
                        v_lift = vv * m + ((s + volt) % m)
                        if v_lift not in adj[u_lift]:
                            adj[u_lift].add(v_lift)
                            adj[v_lift].add(u_lift)
            
            # Quick check: is it k-regular?
            regular = all(len(adj[i]) == k for i in range(lifted_n))
            if not regular:
                voltage[(u, v)] = old_volt
                continue
            
            score = _short_cycle_score(adj, lifted_n, g)
            
            if score < best_score:
                best_score = score
                best_voltage = dict(voltage)
                if score == 0:
                    # Found a valid graph
                    edges_list = []
                    for i in range(lifted_n // 2):
                        for j in adj[i]:
                            if j > i:
                                edges_list.append((i, j))
                    return lifted_n, edges_list
            else:
                # Metropolis criterion
                delta = score - best_score
                if delta > 0 and random.random() > math.exp(-delta / temp):
                    voltage[(u, v)] = old_volt
            
            temp *= cooling
        
        if best_score == 0:
            # Reconstruct
            adj = [set() for _ in range(lifted_n)]
            for uu in range(base_n):
                for vv, _ in base_adj[uu]:
                    volt = best_voltage.get((min(uu, vv), max(uu, vv)), 0)
                    if uu > vv:
                        volt = (m - volt) % m
                    for s in range(m):
                        u_lift = uu * m + s
                        v_lift = vv * m + ((s + volt) % m)
                        adj[u_lift].add(v_lift)
                        adj[v_lift].add(u_lift)
            edges_list = []
            for i in range(lifted_n):
                for j in adj[i]:
                    if j > i:
                        edges_list.append((i, j))
            return lifted_n, edges_list
    
    return None

def _direct_search(n, k, g, deadline):
    """
    Direct local search on a random k-regular bipartite graph.
    """
    half = n // 2
    adj = [set() for _ in range(n)]
    
    # Generate random 4-regular bipartite graph
    a_half = []
    for i in range(half):
        a_half.extend([i] * k)
    b_half = []
    for i in range(half):
        b_half.extend([half + i] * k)
    
    random.shuffle(b_half)
    for a, b in zip(a_half, b_half):
        adj[a].add(b)
        adj[b].add(a)
    
    if not all(len(adj[i]) == k for i in range(n)):
        return None
    
    best_adj = [set(s) for s in adj]
    best_score = _short_cycle_score(best_adj, n, g)
    
    if best_score == 0:
        edges = []
        for i in range(half):
            for j in best_adj[i]:
                edges.append((i, j))
        return n, edges
    
    adj_lists = [list(s) for s in best_adj]
    
    def do_swap(u1, v1, u2, v2):
        best_adj[u1].discard(v1)
        best_adj[v1].discard(u1)
        best_adj[u2].discard(v2)
        best_adj[v2].discard(u2)
        best_adj[u1].add(v2)
        best_adj[v2].add(u1)
        best_adj[u2].add(v1)
        best_adj[v1].add(u2)
    
    def undo_swap(u1, v1, u2, v2):
        best_adj[u1].discard(v2)
        best_adj[v2].discard(u1)
        best_adj[u2].discard(v1)
        best_adj[v1].discard(u2)
        best_adj[u1].add(v1)
        best_adj[v1].add(u1)
        best_adj[u2].add(v2)
        best_adj[v2].add(u2)
    
    for iteration in range(30000):
        if time.time() > deadline:
            break
        
        u1 = random.randint(0, half - 1)
        u2 = random.randint(0, half - 1)
        if u1 == u2:
            continue
        if not best_adj[u1] or not best_adj[u2]:
            continue
        v1 = random.choice(list(best_adj[u1]))
        v2 = random.choice(list(best_adj[u2]))
        if v1 == v2:
            continue
        if v2 in best_adj[u1] or v1 in best_adj[u2]:
            continue
        
        do_swap(u1, v1, u2, v2)
        
        # Quick check
        new_score = _short_cycle_score(best_adj, n, g)
        
        if new_score < best_score:
            best_score = new_score
            adj_lists = [list(s) for s in best_adj]
            if new_score == 0:
                edges = []
                for i in range(half):
                    for j in best_adj[i]:
                        edges.append((i, j))
                return n, edges
        else:
            # Accept with probability based on improvement potential
            if new_score <= best_score + 50 and random.random() < 0.3:
                adj_lists = [list(s) for s in best_adj]
            else:
                undo_swap(u1, v1, u2, v2)
    
    return None

def construct(k, g):
    if k != 4 or g != 10:
        return _luw_construct(k, g)
    
    deadline = time.time() + 110
    best_n, best_edges = None, None
    
    # Strategy 1: Voltage lift approach
    result = _find_voltage_assignment(None, 0, k, g, deadline - 20)
    if result is not None:
        best_n, best_edges = result
    
    # Strategy 2: Direct local search on random bipartite graphs
    # Try sizes from Moore bound upward
    moore_bound = 2 * (1 + 3 + 9 + 27 + 81)  # 242
    start_n = moore_bound + 2 if moore_bound % 2 == 0 else moore_bound + 1
    
    # If voltage lift found something, try to beat it
    upper = best_n if best_n is not None else 500
    
    for n in range(start_n, min(upper + 1, 400), 2):
        if time.time() > deadline - 5:
            break
        
        attempts = max(2, 10 - (n - start_n) // 15)
        for _ in range(attempts):
            if time.time() > deadline - 5:
                break
            result = _direct_search(n, k, g, deadline - 5)
            if result is not None:
                best_n, best_edges = result
                break
        if best_n is not None and best_n <= n:
            break
    
    # Fallback: LUW construction
    if best_n is None or best_n > 320:
        luw_n, luw_edges = _luw_construct(k, g)
        if best_n is None or luw_n < best_n:
            best_n, best_edges = luw_n, luw_edges
    
    return best_n, best_edges
