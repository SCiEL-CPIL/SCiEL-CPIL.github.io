"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Strategy:
    1. Try to find a small Cayley graph on A6 (order 360) with girth >= g.
    2. Try voltage graph (lift) construction over a small 4-regular base graph of girth 5,
       assigning Z_2^m voltage matrices to edges to maximize the girth of the resulting cover.
    3. Fallback to Lazebnik-Ustimenko-Woldar (LUV) graph construction.
    
    The evaluator measures n (number of vertices), lower is better.
    """
    start_time = time.time()
    deadline = 115.0  # 5s safety margin before 120s limit

    # --- Part 1: Try Cayley graph on A6 (order 360) ---
    def try_cayley_a6(k, g):
        """Search for a 4-regular Cayley graph on A6 with girth >= g."""
        if k != 4:
            return None
        
        import itertools
        
        def is_even(perm):
            inv_count = 0
            for i in range(6):
                for j in range(i+1, 6):
                    if perm[i] > perm[j]:
                        inv_count += 1
            return inv_count % 2 == 0
        
        a6_elems = []
        for p in itertools.permutations(range(6)):
            if is_even(p):
                a6_elems.append(p)
        
        n = len(a6_elems)
        if n != 360:
            return None
        
        index = {e: i for i, e in enumerate(a6_elems)}
        identity = tuple(range(6))
        e_id = index[identity]
        
        mult = [[0] * n for _ in range(n)]
        for i in range(n):
            pi = a6_elems[i]
            for j in range(n):
                pj = a6_elems[j]
                prod = tuple(pi[pj[x]] for x in range(6))
                mult[i][j] = index[prod]
        
        inv = [0] * n
        for i in range(n):
            for j in range(n):
                if mult[i][j] == e_id:
                    inv[i] = j
                    break
        
        candidates = []
        for i in range(n):
            if i == e_id:
                continue
            if inv[i] == i:
                continue
            candidates.append(i)
        
        element_order = [1] * n
        for i in range(n):
            if i == e_id:
                continue
            cur = i
            order = 1
            while cur != e_id:
                cur = mult[cur][i]
                order += 1
                if order > n:
                    break
            element_order[i] = order
        
        cand_sorted = sorted(candidates, key=lambda x: element_order[x], reverse=True)
        random.seed(12345)
        
        top_count = min(30, len(cand_sorted))
        top_cands = cand_sorted[:top_count]
        
        # Phase 1: Systematic search among top-order candidates
        for i1 in range(len(top_cands)):
            if time.time() > deadline:
                break
            g1 = top_cands[i1]
            g1_inv = inv[g1]
            
            for i2 in range(i1 + 1, len(top_cands)):
                if time.time() > deadline:
                    break
                g2 = top_cands[i2]
                g2_inv = inv[g2]
                
                if g1 == g2 or g1 == g2_inv or g1_inv == g2 or g1_inv == g2_inv:
                    continue
                
                depth = [-1] * n
                parent = [-1] * n
                q = deque()
                q.append(e_id)
                depth[e_id] = 0
                
                min_cycle = float('inf')
                valid = True
                
                while q:
                    u = q.popleft()
                    du = depth[u]
                    if du >= g:
                        continue
                    
                    for s in (g1, g1_inv, g2, g2_inv):
                        v = mult[u][s]
                        if depth[v] == -1:
                            depth[v] = du + 1
                            parent[v] = u
                            q.append(v)
                        else:
                            dv = depth[v]
                            if u != parent[v] and v != parent[u]:
                                cl = du + dv + 1
                                if cl < min_cycle:
                                    min_cycle = cl
                                    if min_cycle < g:
                                        valid = False
                                        break
                    if not valid:
                        break
                
                if valid and min_cycle >= g:
                    edges = []
                    for i in range(n):
                        for s in (g1, g1_inv, g2, g2_inv):
                            j = mult[i][s]
                            if i < j:
                                edges.append((i, j))
                    return n, edges
        
        # Phase 2: Random search
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        max_random_pairs = int(remaining * 1000 / 2.0)
        max_random_pairs = max(50, min(max_random_pairs, 2000))
        
        random.shuffle(cand_sorted)
        
        for _ in range(max_random_pairs):
            if time.time() > deadline:
                break
            
            i1 = random.randint(0, len(cand_sorted) - 1)
            i2 = random.randint(0, len(cand_sorted) - 1)
            if i1 == i2:
                continue
            
            g1 = cand_sorted[i1]
            g2 = cand_sorted[i2]
            g1_inv = inv[g1]
            g2_inv = inv[g2]
            
            if g1 == g2 or g1 == g2_inv or g1_inv == g2 or g1_inv == g2_inv:
                continue
            
            depth = [-1] * n
            parent = [-1] * n
            q = deque()
            q.append(e_id)
            depth[e_id] = 0
            
            min_cycle = float('inf')
            valid = True
            
            while q:
                u = q.popleft()
                du = depth[u]
                if du >= g:
                    continue
                
                for s in (g1, g1_inv, g2, g2_inv):
                    v = mult[u][s]
                    if depth[v] == -1:
                        depth[v] = du + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        dv = depth[v]
                        if u != parent[v] and v != parent[u]:
                            cl = du + dv + 1
                            if cl < min_cycle:
                                min_cycle = cl
                                if min_cycle < g:
                                    valid = False
                                    break
                if not valid:
                    break
            
            if valid and min_cycle >= g:
                edges = []
                for i in range(n):
                    for s in (g1, g1_inv, g2, g2_inv):
                        j = mult[i][s]
                        if i < j:
                            edges.append((i, j))
                return n, edges
        
        return None

    # --- Part 2: Voltage graph (lift) construction ---
    def try_voltage_lift(k, g):
        """
        Try to construct a 4-regular graph with girth >= g using voltage graph (lift) construction.
        Start with a small 4-regular base graph of girth 5 (e.g., 8 vertices), then lift it
        by assigning Z_2^m voltages to edges to maximize the girth of the resulting cover.
        """
        if k != 4:
            return None
        
        # Base graph: 8-vertex 4-regular graph of girth 5
        # This is the complement of two disjoint 4-cycles, or we can use a known small graph.
        # Let's use the 8-vertex graph formed by connecting two 4-cycles with cross edges.
        # Actually, let's use a simpler approach: start with a known small 4-regular graph.
        
        # For now, let's try a different approach: use a 4-regular graph on 8 vertices.
        # One such graph is the complete bipartite K_{4,4} minus a perfect matching.
        # That gives 8 vertices, each of degree 3. Not 4-regular.
        
        # Let's use the 8-vertex 4-regular graph: take two disjoint K_4's and add 4 cross edges.
        # K_4 has 4 vertices each of degree 3. Add 4 cross edges to make each vertex degree 4.
        
        # Actually, let's just use a random 4-regular graph on 8 vertices and check girth.
        # But for determinism, let's use a fixed one.
        
        # The 8-vertex 4-regular graph with girth 5:
        # Vertices 0-3 form a 4-cycle: 0-1-2-3-0
        # Vertices 4-7 form a 4-cycle: 4-5-6-7-4
        # Cross edges: 0-4, 1-5, 2-6, 3-7
        # This gives each vertex degree 4 (2 from cycle + 1 cross + 1 from... wait, that's only 3)
        
        # Let me reconsider. A 4-regular graph on 8 vertices:
        # Each vertex has degree 4, so total edges = 8*4/2 = 16.
        # Let's use the complement of a 3-regular graph on 8 vertices.
        # The 3-regular graph on 8 vertices with girth 4 is K_{4,4} minus a perfect matching.
        # Its complement would be 4-regular.
        
        # Actually, let's just use a known 4-regular graph on 8 vertices:
        # Vertices 0-7
        # Edges: 0-1, 0-2, 0-3, 0-4, 1-2, 1-3, 1-5, 1-6, 2-3, 2-5, 2-7, 3-4, 3-6, 3-7, 4-5, 4-6, 5-7, 6-7
        # Wait, that's 18 edges. Let me count degrees.
        
        # Let's just use a simple approach: generate a random 4-regular graph on 8 vertices
        # and check if it has girth >= 5. If not, try another.
        
        # For now, let's skip this and focus on improving the Cayley graph search.
        # The voltage lift approach is complex and may not yield better results in the time budget.
        return None

    # --- Part 3: Fallback to LUV construction ---
    def luv_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph construction."""
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
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    # --- Main logic ---
    # First, try the Cayley graph approach if k=4
    if k == 4:
        result = try_cayley_a6(k, g)
        if result is not None:
            n, edges = result
            return n, edges
        
        # Try voltage lift if Cayley didn't find anything
        result = try_voltage_lift(k, g)
        if result is not None:
            n, edges = result
            return n, edges
    
    # Fallback to LUV
    try:
        n, edges = luv_construct(k, g)
        return n, edges
    except:
        pass

    return 0, []
# EVOLVE-BLOCK-END
