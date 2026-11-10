import time
import random
import math
import sys

def _luw_construct(k, g):
    # Seed logic
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

def _check_girth_fast(adj, n, min_girth):
    """Check if graph has girth >= min_girth using BFS from each vertex. Early exit."""
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
                    cycle_len = d + dist[nb] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _count_4cycles(adj, n):
    """Count 4-cycles in a bipartite graph (A: 0..half-1, B: half..n-1)."""
    half = n // 2
    count = 0
    # For each pair of B vertices, count common A neighbors
    # A 4-cycle is a-b-c-d-a where a,c in A and b,d in B
    # Equivalently: pairs of A vertices sharing 2+ common B neighbors
    for u in range(half):
        au = adj[u]
        for v in range(u + 1, half):
            common = len(au & adj[v])
            if common >= 2:
                count += common * (common - 1) // 2
    return count

def _count_6cycles_partial(adj, n, sample_pairs=None):
    """
    Count 6-cycles. A 6-cycle: a1-b1-a2-b2-a3-b3-a1
    where a1,a2,a3 in A, b1,b2,b3 in B.
    For each ordered pair (a1,a2) of distinct A vertices sharing a B neighbor b1,
    and each B neighbor b3 of a2 that is not b1, count common A neighbors of b3 and a1
    (excluding a2).
    This is O(half^2 * deg^2) which is manageable for small half.
    """
    half = n // 2
    count = 0
    # Precompute adjacency as lists for speed
    adj_lists = [list(s) for s in adj]
    for a1 in range(half):
        for a2 in range(a1 + 1, half):
            # Find common B neighbors of a1 and a2
            common_b = adj_lists[a1] & adj_lists[a2]
            if not common_b:
                continue
            for b1 in common_b:
                # b1 connects a1 and a2
                # Now need a3 in A such that a3 is neighbor of b1 (no, b1 is in B)
                # 6-cycle: a1-b1-a2-b2-a3-b3-a1
                # b1 is shared by a1 and a2
                # b2 is a neighbor of a2 (not b1)
                # a3 is a neighbor of b2 (in A, not a1, not a2)
                # b3 is a neighbor of a3 (in B) that is also neighbor of a1 (not b1)
                for b2 in adj_lists[a2]:
                    if b2 == b1:
                        continue
                    # a3 must be in A, neighbor of b2, not a1 or a2
                    # b2 is in B, its A-neighbors are adj[b2]
                    for a3 in adj_lists[b2]:
                        if a3 <= half - 1 and a3 != a1 and a3 != a2:
                            # b3 must be common neighbor of a3 and a1 in B, not b1
                            common_b3 = adj_lists[a3] & adj_lists[a1]
                            common_b3.discard(b1)
                            if common_b3:
                                count += len(common_b3)
    return count

def construct(k, g):
    if k != 4 or g != 10:
        return _luw_construct(k, g)
    
    deadline = time.time() + 115
    
    # First, get the LUW construction as fallback
    luw_n, luw_edges = _luw_construct(k, g)
    best_n = luw_n
    best_edges = luw_edges
    
    # Try to find a smaller graph using local search
    # Start with random 4-regular bipartite graphs and try to maximize girth
    
    def generate_random_bipartite_4reg(n):
        """Generate a random 4-regular bipartite graph on n vertices (n even)."""
        half = n // 2
        # Create adjacency: each vertex in A (0..half-1) connects to 4 vertices in B (half..n-1)
        # Use configuration model
        adj = [set() for _ in range(n)]
        
        # Each A vertex needs degree 4, each B vertex needs degree 4
        # Create list of halfends
        a_halfends = []
        for i in range(half):
            a_halfends.extend([i] * 4)
        b_halfends = []
        for i in range(half):
            b_halfends.extend([half + i] * 4)
        
        random.shuffle(b_halfends)
        for a, b in zip(a_halfends, b_halfends):
            adj[a].add(b)
            adj[b].add(a)
        
        # Check for multi-edges
        for i in range(n):
            if len(adj[i]) != k:
                return None
        
        return adj
    
    def local_search(adj, n, deadline):
        """Local search to maximize girth via edge swaps."""
        best_adj = [set(s) for s in adj]
        half = n // 2
        adj_lists = [list(s) for s in best_adj]
        
        def recompute_lists():
            nonlocal adj_lists
            adj_lists = [list(s) for s in best_adj]
        
        def swap(u1, v1, u2, v2):
            """Perform swap: (u1,v1)+(u2,v2) -> (u1,v2)+(u2,v1)"""
            best_adj[u1].remove(v1)
            best_adj[v1].remove(u1)
            best_adj[u2].remove(v2)
            best_adj[v2].remove(u2)
            best_adj[u1].add(v2)
            best_adj[v2].add(u1)
            best_adj[u2].add(v1)
            best_adj[v1].add(u2)
        
        def unswap(u1, v1, u2, v2):
            """Revert swap: (u1,v2)+(u2,v1) -> (u1,v1)+(u2,v2)"""
            best_adj[u1].remove(v2)
            best_adj[v2].remove(u1)
            best_adj[u2].remove(v1)
            best_adj[v1].remove(u2)
            best_adj[u1].add(v1)
            best_adj[v1].add(u1)
            best_adj[u2].add(v2)
            best_adj[v2].add(u2)
        
        # Initial girth check
        if _check_girth_fast(best_adj, n, g):
            edges = []
            for i in range(half):
                for j in best_adj[i]:
                    edges.append((i, j))
            return n, edges
        
        # Score: minimize 4-cycles primarily, then 6-cycles
        score_4 = _count_4cycles(best_adj, n)
        best_score_4 = score_4
        
        for iteration in range(20000):
            if time.time() > deadline:
                break
            
            # Try multiple swaps
            for attempt in range(30):
                if time.time() > deadline:
                    break
                u1 = random.randint(0, half - 1)
                u2 = random.randint(0, half - 1)
                if u1 == u2:
                    continue
                v1 = random.choice(adj_lists[u1])
                v2 = random.choice(adj_lists[u2])
                if v1 == v2:
                    continue
                if v2 in best_adj[u1] or v1 in best_adj[u2]:
                    continue
                
                # Perform swap
                swap(u1, v1, u2, v2)
                recompute_lists()
                
                # Quick check: count 4-cycles
                new_score_4 = _count_4cycles(best_adj, n)
                
                if new_score_4 < best_score_4:
                    # Improvement - check girth
                    if _check_girth_fast(best_adj, n, g):
                        edges = []
                        for i in range(half):
                            for j in best_adj[i]:
                                edges.append((i, j))
                        return n, edges
                    best_score_4 = new_score_4
                elif new_score_4 == best_score_4:
                    # Check girth anyway
                    if _check_girth_fast(best_adj, n, g):
                        edges = []
                        for i in range(half):
                            for j in best_adj[i]:
                                edges.append((i, j))
                        return n, edges
                    # Revert if no improvement and no girth win
                    unswap(u1, v1, u2, v2)
                    recompute_lists()
                else:
                    # Worse, revert
                    unswap(u1, v1, u2, v2)
                    recompute_lists()
        
        return None
    
    # Try different sizes starting from smaller
    # Moore bound for (4,10): n >= 2*(1 + 3 + 9 + 27 + 81) = 2*121 = 242
    # So minimum possible is 242, but must be even.
    
    # Try sizes from 244 up to luw_n, with more attempts for smaller sizes
    for n in range(244, min(luw_n + 1, 400), 2):
        if time.time() > deadline - 3:
            break
        
        # More attempts for smaller n (harder to find valid graphs)
        num_attempts = max(3, 20 - (n - 244) // 10)
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 3:
                break
            
            adj = generate_random_bipartite_4reg(n)
            if adj is None:
                continue
            
            # Quick girth check before expensive local search
            if _check_girth_fast(adj, n, g):
                edges = []
                half = n // 2
                for i in range(half):
                    for j in adj[i]:
                        edges.append((i, j))
                best_n, best_edges = n, edges
                return best_n, best_edges
            
            result = local_search(adj, n, deadline - 3)
            if result is not None:
                best_n, best_edges = result
                return best_n, best_edges
    
    return best_n, best_edges
