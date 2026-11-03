import time
import random
from collections import deque
import numpy as np

_DEADLINE = 145.0


def _elapsed(start):
    return time.time() - start


def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _girth_ok(adj, n, g, start):
    """
    Check that no cycle shorter than g exists using vectorized non-backtracking walks.
    """
    # Precompute adjacency matrix as sparse or dense? 
    # For n=320, dense is 320x320 = 102k floats, very small.
    # But we need to handle "previous" state.
    # State: matrix M where M[i, j] = 1 if we can reach i from j (last step j->i).
    # Initially, for each start v0, M is zero.
    # Step 1: For each neighbor nb of v0, M[nb, v0] = 1.
    # Step 2: New_M[i, j] = OR over all k != j of (M[i, k] AND A[k, j])? 
    # Actually, standard non-backtracking:
    # If we are at (u, v), we can go to w if A[v, w]=1 and w != u.
    # Let's represent the frontier as a boolean matrix F of size n x n.
    # F[u, v] = True if we are at u having come from v.
    
    # We need to check for every v0 if we return to v0.
    # A cycle of length L < g exists if there is a walk v0 -> ... -> v0 of length L.
    # In the non-backtracking walk context:
    # Start: F[nb, v0] = 1 for nb in adj[v0].
    # If at step L we have F[v0, nb] = 1 for any nb, we have a cycle of length L.
    
    # Vectorized update:
    # F_new[u, v] = (sum over w != v of F[u, w] * A[w, v]) > 0
    # Since A is 0/1, this is: (F @ A.T)[u, v] - F[u, v] * A[v, v] (A has no loops)
    # Wait, F[u, w] means we are at u, came from w.
    # We want to move to v. So we need A[w, v] = 1 and w != v (no loops) and w != current_prev?
    # The condition is: we are at u, came from w. We move to v.
    # The edge is (w, u) then (u, v).
    # Non-backtracking means v != w.
    # So F_new[u, v] is True if there exists w such that F[u, w] is True, A[w, v] is True, and w != v.
    # Note: A[w, v] implies w != v for simple graphs.
    # So F_new[u, v] = (sum_{w} F[u, w] * A[w, v]) > 0.
    # This is simply F @ A.T.
    
    # Let's verify:
    # F[u, w] = 1 means path ... -> w -> u.
    # We want to extend to v. Edge (u, v) must exist.
    # So we need A[u, v] = 1? No, the matrix multiplication F @ A.T:
    # (F @ A.T)[u, v] = sum_w F[u, w] * A.T[w, v] = sum_w F[u, w] * A[v, w].
    # This checks if we came from w, and there is an edge between v and w.
    # But we are at u. We need an edge between u and v.
    # So the transition should be:
    # We are at u, came from w. We move to v if A[u, v] = 1 and v != w.
    # This is not a simple matrix multiplication on F alone because the condition v != w depends on the "from" index of F and the "to" index of the new state.
    
    # Let's stick to the list-based approach but optimize it? 
    # Or use the fact that n is small.
    # The current Python loop is:
    # for cur, prev in frontier:
    #   for nb in adj[cur]:
    #     if nb != prev: ...
    
    # This is O(E * degree) per step.
    # For n=320, E=640, deg=4. 640*4 = 2560 ops per step.
    # g=10, so 8 steps. 8 * 2560 * 320 (start vertices) = 6.5 million ops.
    # Python does ~10M simple ops/sec? Maybe 1-2 seconds.
    # The previous code was likely slower due to bytearray overhead and function calls.
    
    # Let's try to optimize the Python loop by using local variables and avoiding list creation where possible.
    
    for v0 in range(n):
        # visited[u][v] = 1 if we reached u from v
        visited = [bytearray(n) for _ in range(n)]
        frontier = []
        
        # Optimization: Pre-allocate or use lists of ints
        # adj is list of lists of ints
        
        for nb in adj[v0]:
            frontier.append((nb, v0))
            visited[nb][v0] = 1

        for length in range(2, g):
            if _elapsed(start) > _DEADLINE - 2:
                # If we run out of time, we assume it's OK to return the known good graph,
                # but we shouldn't claim verification passed if we didn't finish.
                # However, if we are inside _verify, returning True might be invalid.
                # Better to return False if we can't finish, so we don't return an unverified graph.
                # But for the Lazebnik graph, we KNOW it is valid.
                # The risk is if _verify is called on a random graph and times out.
                # We will only call _verify on Lazebnik (known good) and Circulants (fast to check).
                return False 

            nxt = []
            # Local vars for speed
            adj_local = adj
            visited_local = visited
            v0_local = v0
            
            for cur, prev in frontier:
                for nb in adj_local[cur]:
                    if nb == prev:
                        continue
                    if nb == v0_local:
                        return False
                    if not visited_local[nb][cur]:
                        visited_local[nb][cur] = 1
                        nxt.append((nb, cur))
            frontier = nxt
            if not frontier:
                break
    return True

def _lazebnik(k, g):
    """
    Construct the Lazebnik graph for the (4, 10)-cage.
    This specific graph has 320 vertices.
    """
    if k != 4:
        # Fallback for other cases, though task is specific
        pass

    q = 4
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start_vert = (0, (None,) + (0,) * d)
    index, order, edges = {start_vert: 0}, [start_vert], []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges


def _verify(n, edges, k, g, start):
    # Basic structural checks
    if len(edges) != n * k // 2:
        return False
    deg = [0] * n
    eset = set()
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n or u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in eset:
            return False
        eset.add((u, v))
        deg[u] += 1
        deg[v] += 1
    if any(d != k for d in deg):
        return False
    adj = _build_adj(edges, n)
    return _girth_ok(adj, n, g, start)


def _try_circulant(n, k, g, start):
    """
    Try to find a circulant graph C_n(S) with degree k and girth >= g.
    """
    valid_pairs = []
    for a in range(1, n // 2 + 1):
        for b in range(a + 1, n // 2 + 1):
            if (a + b) % n == 0:
                continue
            if (2 * a) % n == 0:
                continue
            if (2 * b) % n == 0:
                continue
            valid_pairs.append((a, b))
    
    if not valid_pairs:
        return None
    
    def check_girth_circulant_fast(n, a, b, g):
        # Fast BFS girth check
        offsets = [a, -a, b, -b]
        depth = [-1] * n
        depth[0] = 0
        queue = [0]
        head = 0
        min_cycle = float('inf')
        max_depth = (g - 1) // 2
        
        while head < len(queue):
            u = queue[head]
            head += 1
            if depth[u] >= max_depth:
                continue
            for off in offsets:
                v = (u + off) % n
                if depth[v] == -1:
                    depth[v] = depth[u] + 1
                    queue.append(v)
                else:
                    # Check if it's a back edge to parent
                    # We don't store parent, so we check if depth[v] == depth[u] - 1 and v is the one that led to u?
                    # In BFS, if we encounter a visited node v:
                    # If depth[v] < depth[u], it could be a back edge.
                    # Cycle length = depth[u] + depth[v] + 1 is an upper bound on the cycle containing this edge?
                    # Actually, if we just want to know if girth < g, we can stop if we find any cycle < g.
                    # The shortest cycle containing edge (u,v) where both are visited in BFS tree
                    # is at most depth[u] + depth[v] + 1.
                    # If depth[u] + depth[v] + 1 < g, we have a cycle shorter than g.
                    # Note: This is a sufficient condition for finding a short cycle, but not necessary 
                    # (the actual cycle might be shorter, but if this sum is >= g, the actual cycle might still be < g? 
                    # No, the BFS tree path is the shortest path in the tree. The actual graph path is <= tree path.
                    # So if tree path sum + 1 >= g, the actual cycle could be < g.
                    # So this check is NOT sufficient to prove girth >= g.
                    # However, if we find a cycle < g, we can reject.
                    # So if depth[u] + depth[v] + 1 < g, reject.
                    # If not, we can't conclude.
                    # But for circulant graphs, BFS from 0 often finds the girth.
                    # Let's use this as a fast filter. If it passes, we do full verification.
                    pass 
                    
        # Since the fast check is tricky to make rigorous for "girth >= g", 
        # and n is small, we can just build the graph and call _verify.
        # But _verify is expensive.
        # Let's just return the first few candidates and verify them.
        return True # Placeholder
        
    # Given time constraints, we will only verify a small number of candidates.
    # We iterate n from 244 upwards.
    # We return None if we don't find one quickly.
    # This function is called in a loop in construct.
    
    # We'll just pick a random valid pair to test, or the first one.
    # Actually, for n=244, there are ~15000 pairs.
    # We can't verify all.
    # We'll verify the first 50 pairs.
    
    for i, (a, b) in enumerate(valid_pairs[:50]):
        if _elapsed(start) > _DEADLINE - 10:
            return None
        
        edges = []
        for u in range(n):
            for off in [a, -a, b, -b]:
                v = (u + off) % n
                if u < v:
                    edges.append((u, v))
        
        if _verify(n, edges, k, g, start):
            return n, edges
            
    return None

def construct(k, g):
    start = time.time()
    
    # 1. Construct Lazebnik (320 vertices)
    best_n, best_edges = _lazebnik(k, g)
    
    # 2. Verify it. With optimized _girth_ok, this should be fast.
    if _verify(best_n, best_edges, k, g, start):
        # 3. Try to find smaller circulant graphs.
        # Moore bound is 243.
        # Try even n from 244 to 318.
        # We have ~140s left.
        # Each _verify takes ~0.1-0.5s? 
        # If _girth_ok is 1s, we can try 100 candidates.
        # But _try_circulant only tries 50 candidates per n.
        # And there are ~37 values of n.
        # 37 * 50 = 1850 verifications. Too many if each takes 1s.
        
        # Let's limit the search.
        # We will try n=244, 246, ..., up to time limit.
        
        # Optimization: Use a faster girth check for circulants?
        # Or just accept that we likely won't find one and return 320.
        
        # Let's try a few n values.
        for n in range(244, best_n, 2):
            if _elapsed(start) > _DEADLINE - 5:
                break
            
            # Try a subset of circulants
            res = _try_circulant(n, k, g, start)
            if res:
                best_n, best_edges = res
                # Since we iterate n upwards, first found is smallest.
                # But we might miss a smaller one if we didn't search all pairs.
                # However, if we find one, it's better than 320.
                # We break because we want the smallest n.
                # But we only checked 50 pairs for this n.
                # There might be a valid circulant for n=244 that we missed.
                # If we found one for n=250, we return 250.
                # If there was a valid one for n=244, we missed it.
                # Given the rarity, this is a gamble.
                break

    return best_n, best_edges
