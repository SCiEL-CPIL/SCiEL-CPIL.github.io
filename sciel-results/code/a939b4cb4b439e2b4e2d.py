import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # The seed program used LUW(5,4) which gives n=625.
    # We try to beat 625 using voltage graphs on K5.
    # n = 5*m. We want 5*m < 625 => m < 125.
    
    if k != 5 or g != 9:
        # Fallback to seed behavior for other params, though task is specific
        # For now, just return a dummy or seed logic if applicable
        pass

    # Pre-compute Non-Backtracking walks of length < 9 in K5
    # Vertices 0..4
    # Edges are pairs (u, v) with u < v
    edges_k5 = [(i, j) for i in range(5) for j in range(i+1, 5)]
    edge_index = {e: i for i, e in enumerate(edges_k5)}
    
    # Adjacency list for K5
    adj_k5 = [[] for _ in range(5)]
    for u, v in edges_k5:
        adj_k5[u].append(v)
        adj_k5[v].append(u)

    # Generate all NB walks of length 1 to 8
    # Each walk is a sequence of vertices (v0, v1, ..., vL)
    # We need the voltage sum.
    # Voltage of step vi -> vi+1 is vol_e if vi+1 > vi? No.
    # We define voltage direction. Let's say edge (u,v) has voltage vol.
    # If we go u->v, add vol. If v->u, add -vol.
    
    walks = [] # List of coefficient vectors
    
    def gen_nb_walks(current, prev, length, coeffs):
        if length == 0:
            # If we are back at start? No, we are generating walks starting from any vertex.
            # We need to check if it's a closed walk?
            # No, we generate all NB walks. A cycle in the lift corresponds to a CLOSED walk in the base.
            # So we only care about walks where start == end.
            # But generating all walks and filtering is easier?
            # Or generate closed walks directly?
            # Let's generate all NB walks of length L, and check if start==end.
            pass
            
    # Better approach:
    # Iterate start vertex s.
    # Do DFS for NB walks of length L.
    # If end vertex is s, it's a candidate cycle.
    # Record the coefficients.
    
    all_closed_nb_walks = []
    
    for s in range(5):
        # DFS state: (curr, prev, length, coeffs)
        # coeffs is a list of 10 integers
        def dfs(curr, prev, length, coeffs):
            if length == 0:
                if curr == s:
                    # Found a closed NB walk
                    all_closed_nb_walks.append(tuple(coeffs))
                return
            
            for nb in adj_k5[curr]:
                if nb == prev:
                    continue # Non-backtracking
                # Update coeffs
                new_coeffs = coeffs[:]
                if curr < nb:
                    e_idx = edge_index[(curr, nb)]
                    new_coeffs[e_idx] += 1
                else:
                    e_idx = edge_index[(nb, curr)]
                    new_coeffs[e_idx] -= 1
                dfs(nb, curr, length - 1, new_coeffs)
        
        dfs(s, -1, 1, [0]*10)
        dfs(s, -1, 2, [0]*10)
        dfs(s, -1, 3, [0]*10)
        dfs(s, -1, 4, [0]*10)
        dfs(s, -1, 5, [0]*10)
        dfs(s, -1, 6, [0]*10)
        dfs(s, -1, 7, [0]*10)
        dfs(s, -1, 8, [0]*10)
        
    # Note: A walk of length L starting at s and ending at s is a cycle.
    # We have collected all such walks.
    # There might be duplicates (same set of edges traversed in same order)?
    # The coefficient vector captures the net usage.
    # Two different walks can have the same coefficient vector.
    # We only need to check if the sum is 0 for ANY walk.
    # If two walks have the same coeffs, checking one is enough.
    # So we can deduplicate the coefficient vectors.
    
    unique_walks = list(set(all_closed_nb_walks))
    
    # Sort by length? Not strictly necessary for validity, but good for early termination?
    # No, we just need ALL to be non-zero.
    
    # Search for m and voltages
    best_n = 625
    best_edges = None
    
    # Try m from 124 down to 1
    # We want the smallest n, so smallest m.
    # But finding a solution for small m is harder.
    # Start from m=124 (n=620) and go down.
    # If we find a solution, we can try smaller m if time permits.
    # Or just return the first one found (which is the largest m < 125 that works)?
    # No, we want MIN n. So we want MIN m.
    # So we should try m=1, 2, ... 124?
    # But small m is very likely to fail.
    # Let's try a few promising m values.
    # The density of solutions increases with m.
    # Let's try m in [100, 124] first, then [80, 99], etc.
    
    # Actually, if we find a valid m, n=5m.
    # To minimize n, we minimize m.
    
    # Let's try m from 100 to 124.
    # If found, we can try to go lower?
    # Given time, let's just find the best m in a range.
    
    # Heuristic: Start with m=124, if found, check if we can find smaller.
    # But "smallest known" is 625. If we find 620, we win.
    
    for m in range(124, 79, -1):
        if time.time() > deadline - 10:
            break
            
        # Local search for this m
        # Init random voltages
        # Volts are in 1..m-1 (0 is forbidden? No, 0 is allowed if it doesn't create short cycles?
        # If vol=0, the edge is "trivial".
        # But if vol=0, it might contribute to sums being 0?
        # Actually, if vol=0, it's just an unlifted edge.
        # Let's allow 0..m-1.
        
        # Objective: Minimize violations
        def count_violations(vols):
            viol = 0
            for w in unique_walks:
                s = 0
                for i in range(10):
                    s += w[i] * vols[i]
                if s % m == 0:
                    viol += 1
            return viol

        # Random restarts
        found = False
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            vols = [random.randint(0, m-1) for _ in range(10)]
            
            # Local search
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                v = count_violations(vols)
                if v == 0:
                    found = True
                    break
                    
                # Pick a random edge to change
                idx = random.randint(0, 9)
                old_val = vols[idx]
                # Try a few random values
                for _ in range(5):
                    new_val = random.randint(0, m-1)
                    if new_val == old_val:
                        continue
                    vols[idx] = new_val
                    new_v = count_violations(vols)
                    if new_v <= v:
                        v = new_v
                        break
                    else:
                        vols[idx] = old_val
                        # If we are stuck, maybe accept worse?
                        # For now, simple hill climbing.
                
                if v == 0:
                    break
            
            if found:
                break
        
        if found:
            n = 5 * m
            # Construct edges
            edges = []
            for u, v in edges_k5:
                vol = vols[edge_index[(u, v)]]
                for k in range(m):
                    u_k = u * m + k
                    v_k = v * m + ((k + vol) % m)
                    edges.append((u_k, v_k))
                    
            # Verify
            # Simple check: n < best_n
            if n < best_n:
                best_n = n
                best_edges = edges
                # We can continue to search for smaller m?
                # The loop goes downwards in m? No, range(124, 79, -1) is 124, 123...
                # So we are finding larger m first.
                # We want smallest m.
                # So we should iterate m from 1 to 124?
                # But small m is hard.
                # Let's stick to finding a valid one first.
                # If we find m=124, n=620.
                # Can we find m=100?
                # Let's modify the loop to try smaller m if time permits.
                
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to Seed LUW(5,4)
    # Re-implementing the seed's LUW logic for safety
    def luw_construct(q, d):
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
        
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
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    n, edges = luw_construct(5, 4)
    return n, edges
