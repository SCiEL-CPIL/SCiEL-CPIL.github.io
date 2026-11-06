import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def get_edges(adj, n):
        """Extract edge list from bitmask adjacency."""
        edges = []
        for i in range(n):
            mask = adj[i]
            j = 0
            while mask:
                if mask & 1:
                    if i < j:
                        edges.append([i, j])
                mask >>= 1
                j += 1
        return edges

    def check_valid(adj, n, min_deg):
        """Check if graph is C4-free and has min degree >= min_deg using bitmasks."""
        for i in range(n):
            if bin(adj[i]).count('1') < min_deg:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if bin(common).count('1') > 1:
                    return False
        return True

    def is_c4_free(adj, n):
        """Check C4-freeness using bitmasks."""
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if bin(common).count('1') > 1:
                    return False
        return True

    def check_add_edge(adj, u, v, n):
        """Check if adding edge (u,v) maintains C4-freeness.
        A C4 is formed if u and v share >= 2 common neighbors.
        Also need to check that no pair (x,y) where x is neighbor of u and y is neighbor of v
        would now share u and v as common neighbors... but actually the only new C4s
        involve the new edge (u,v). A C4 containing (u,v) needs two more vertices x,y
        such that x is adjacent to both u and v, and y is adjacent to both u and v,
        with x != y. So we just need common_neighbors(u,v) <= 1.
        """
        common = adj[u] & adj[v]
        if bin(common).count('1') > 1:
            return False
        return True

    def greedy_priority(n, target_deg):
        """
        Priority-based greedy: build graph by adding edges that maximize
        the minimum degree while maintaining C4-freeness.
        """
        adj = [0] * n
        deg = [0] * n
        
        # Precompute all pairs and their potential
        # Use a priority queue approach: at each step, add the edge that
        # most improves the minimum degree among low-degree vertices
        
        # Start with random seed edges to break symmetry
        random.seed(42)
        for _ in range(n * 2):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and not (adj[u] & (1 << v)):
                if check_add_edge(adj, u, v, n):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1
        
        # Now greedily add edges prioritizing low-degree vertices
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            # Find the vertex with minimum degree
            min_u = min(range(n), key=lambda i: deg[i])
            
            if deg[min_u] >= target_deg:
                # Check if all vertices meet target
                if min(deg) >= target_deg:
                    break
            
            # Try to add an edge from min_u to a non-neighbor
            candidates = []
            for v in range(n):
                if v == min_u or (adj[min_u] & (1 << v)):
                    continue
                if check_add_edge(adj, min_u, v, n):
                    # Score by how much it helps: prefer connecting to other low-degree vertices
                    score = deg[v]  # lower is better (connect to low-degree vertices)
                    candidates.append((score, v))
            
            if not candidates:
                continue
            
            candidates.sort()
            # Try the best candidate (lowest degree non-neighbor)
            _, v = candidates[0]
            adj[min_u] |= (1 << v)
            adj[v] |= (1 << min_u)
            deg[min_u] += 1
            deg[v] += 1
        
        if min(deg) >= target_deg and is_c4_free(adj, n):
            return adj, deg
        return None, None

    def local_search_adj(adj, n, target_deg, max_iters=50000):
        """
        Local search: try to increase minimum degree while maintaining C4-freeness.
        Uses incremental C4 checking for speed.
        """
        deg = [bin(adj[i]).count('1') for i in range(n)]
        best_adj = adj[:]
        best_min_deg = min(deg)
        
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find vertex with minimum degree
            min_u = min(range(n), key=lambda i: deg[i])
            
            if deg[min_u] >= target_deg:
                if min(deg) >= target_deg:
                    break
                # Try to boost other low-degree vertices
                low_verts = [i for i in range(n) if deg[i] < target_deg]
                if not low_verts:
                    break
                min_u = random.choice(low_verts)
            
            # Find non-neighbors of min_u that can be connected
            candidates = []
            for v in range(n):
                if v == min_u or (adj[min_u] & (1 << v)):
                    continue
                if check_add_edge(adj, min_u, v, n):
                    candidates.append(v)
            
            if not candidates:
                no_improve += 1
                if no_improve > 100:
                    break
                continue
            
            no_improve = 0
            
            # Try adding an edge - prefer connecting to low-degree vertices
            candidates.sort(key=lambda v: deg[v])
            v = candidates[0]
            
            # Add edge
            adj[min_u] |= (1 << v)
            adj[v] |= (1 << min_u)
            deg[min_u] += 1
            deg[v] += 1
            
            new_min_deg = min(deg)
            if new_min_deg > best_min_deg:
                best_adj = adj[:]
                best_min_deg = new_min_deg
        
        return best_adj, best_min_deg

    def repair_c4s(adj, n, deg):
        """
        Repair C4 violations by removing edges.
        Returns (adj, deg, success)
        """
        for _ in range(5000):
            if time.time() > deadline:
                break
            
            # Find a C4 violation
            found = False
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[i] & adj[j]
                    if bin(common).count('1') > 1:
                        # Remove an edge to break the C4
                        common_list = []
                        mask = common
                        k = 0
                        while mask:
                            if mask & 1:
                                common_list.append(k)
                            mask >>= 1
                            k += 1
                        
                        # Remove edge from higher-degree vertex to a common neighbor
                        if deg[i] >= deg[j]:
                            w = common_list[-1]
                            adj[i] &= ~(1 << w)
                            adj[w] &= ~(1 << i)
                            deg[i] -= 1
                            deg[w] -= 1
                        else:
                            w = common_list[-1]
                            adj[j] &= ~(1 << w)
                            adj[w] &= ~(1 << j)
                            deg[j] -= 1
                            deg[w] -= 1
                        
                        found = True
                        break
                if found:
                    break
            
            if not found:
                return adj, deg, True
        
        return adj, deg, False

    def solve_for_n(n, target_deg):
        """
        Try to find a C4-free graph on n vertices with min degree >= target_deg.
        """
        # Strategy 1: Priority-based greedy
        adj, deg = greedy_priority(n, target_deg)
        if adj is not None:
            return adj
        
        # Strategy 2: Random init + repair + local search
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 7919 + 13)
            
            # Initialize with sparse random graph
            p = max(0.05, target_deg / n * 1.5)
            adj = [0] * n
            deg = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        deg[i] += 1
                        deg[j] += 1
            
            # Repair C4s
            adj, deg, success = repair_c4s(adj, n, deg)
            if not success:
                continue
            
            # Local search to increase min degree
            adj, min_deg_achieved = local_search_adj(adj, n, target_deg, max_iters=20000)
            
            if min_deg_achieved >= target_deg:
                if is_c4_free(adj, n):
                    return adj
        
        return None

    # Try n from 47 down to 43
    for n in range(47, 42, -1):
        target_deg = n - 39
        if target_deg < 1:
            continue
        
        result_adj = solve_for_n(n, target_deg)
        
        if result_adj is not None:
            if check_valid(result_adj, n, target_deg):
                edges = get_edges(result_adj, n)
                return {'n': n, 'edges': edges}

    # Fallback: Return a valid small graph
    n = 40
    target_deg = 1
    adj = [0] * n
    for i in range(n):
        adj[i] |= (1 << ((i + 1) % n))
        adj[(i + 1) % n] |= (1 << i)
    
    if check_valid(adj, n, target_deg):
        edges = get_edges(adj, n)
        return {'n': n, 'edges': edges}
    
    return {'n': 39, 'edges': []}
