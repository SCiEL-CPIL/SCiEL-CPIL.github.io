import time
import sys
import random
import itertools

import numpy as np


def _luw_construct(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar construction as a guaranteed-valid fallback."""
    q = k
    d_candidates = []
    for d in range(3, 20):
        if d % 2 == 1:
            if d + 5 >= g:
                d_candidates.append(d)
        else:
            if d + 4 >= g:
                d_candidates.append(d)
    if not d_candidates:
        d = max(3, g - 5)
        if d % 2 == 0:
            d += 1
    else:
        d = min(d_candidates)

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
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]

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
            yield (1 - side), tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges


def _verify_girth_np(n, edges, g, deadline):
    """Vectorised exact girth check: no non-backtracking closed walk of length < g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    max_len = g - 1
    for start_v in range(n):
        if time.time() > deadline:
            return False
        states = [(w, start_v) for w in adj[start_v]]
        for d in range(2, max_len + 1):
            new_states = []
            for (v, p) in states:
                for w in adj[v]:
                    if w != p:
                        new_states.append((w, v))
            if not new_states:
                break
            if any(v == start_v for (v, p) in new_states):
                return False
            states = new_states
    return True


def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 135.0

    if k != 4 or g != 10:
        return _luw_construct(k, g, deadline)

    MOD = 40

    def _build_lift(voltages):
        n = 320
        edges = []
        for i in range(4):
            for j in range(4):
                volt = voltages[i][j]
                for kk in range(40):
                    a_idx = i * 40 + kk
                    b_idx = 160 + j * 40 + (kk + volt) % 40
                    edges.append((a_idx, b_idx))
        return n, edges

    # Pre-compute cycle structures for fast cost evaluation
    # 4-cycles: (i, j, k, l) with i!=k, j!=l
    # Voltage: V[i,j] - V[k,j] + V[k,l] - V[i,l]
    cycles_4 = []
    for i in range(4):
        for k in range(4):
            if i == k: continue
            for j in range(4):
                for l in range(4):
                    if j == l: continue
                    # Each cycle counted twice? 
                    # i-j-k-l-i and l-k-j-i-l are the same set of edges?
                    # In voltage graph, direction matters for sign?
                    # V(i,j) is A_i -> B_j.
                    # Cycle A_i -> B_j -> A_k -> B_l -> A_i.
                    # Sum: V[i,j] + (-V[k,j]) + V[k,l] + (-V[i,l])?
                    # Wait, edge B_j -> A_k is reverse of A_k -> B_j.
                    # If V[k,j] is A_k -> B_j, then B_j -> A_k is -V[k,j].
                    # So sum = V[i,j] - V[k,j] + V[k,l] - V[i,l].
                    # We store (i,j,k,l) and the coefficients.
                    cycles_4.append((i, j, k, l, 1, -1, 1, -1))

    # 6-cycles: i,k,m distinct; j,l,n distinct
    # A_i -> B_j -> A_k -> B_l -> A_m -> B_n -> A_i
    # Sum: V[i,j] - V[k,j] + V[k,l] - V[m,l] + V[m,n] - V[i,n]
    cycles_6 = []
    for i, k, m in itertools.permutations(range(4), 3):
        for j, l, n in itertools.permutations(range(4), 3):
            cycles_6.append((i, j, k, l, m, n, 1, -1, 1, -1, 1, -1))

    # 8-cycles: i0..i3 distinct; j0..j3 distinct
    # A_i0 -> B_j0 -> A_i1 -> B_j1 -> A_i2 -> B_j2 -> A_i3 -> B_j3 -> A_i0
    # Sum: V[i0,j0] - V[i1,j0] + V[i1,j1] - V[i2,j1] + V[i2,j2] - V[i3,j2] + V[i3,j3] - V[i0,j3]
    cycles_8 = []
    for ip in itertools.permutations(range(4)):
        for jp in itertools.permutations(range(4)):
            cycles_8.append((ip[0], jp[0], ip[1], jp[1], ip[2], jp[2], ip[3], jp[3],
                             1, -1, 1, -1, 1, -1, 1, -1))

    def _cost(V):
        c = 0
        # V is a 4x4 numpy array
        for cyc in cycles_4:
            s = (V[cyc[0], cyc[1]] * cyc[4] + V[cyc[2], cyc[1]] * cyc[5] + 
                 V[cyc[2], cyc[3]] * cyc[6] + V[cyc[0], cyc[3]] * cyc[7]) % MOD
            if s == 0: c += 1
        for cyc in cycles_6:
            s = (V[cyc[0], cyc[1]] * cyc[6] + V[cyc[2], cyc[1]] * cyc[7] + 
                 V[cyc[2], cyc[3]] * cyc[8] + V[cyc[4], cyc[3]] * cyc[9] +
                 V[cyc[4], cyc[5]] * cyc[10] + V[cyc[0], cyc[5]] * cyc[11]) % MOD
            if s == 0: c += 1
        for cyc in cycles_8:
            s = (V[cyc[0], cyc[1]] * cyc[8] + V[cyc[2], cyc[1]] * cyc[9] +
                 V[cyc[2], cyc[3]] * cyc[10] + V[cyc[4], cyc[3]] * cyc[11] +
                 V[cyc[4], cyc[5]] * cyc[12] + V[cyc[6], cyc[5]] * cyc[13] +
                 V[cyc[6], cyc[7]] * cyc[14] + V[cyc[0], cyc[7]] * cyc[15]) % MOD
            if s == 0: c += 1
        return c

    # Search
    found = None
    
    # Try structured first
    for c in range(1, 40):
        if c % 10 == 0: continue
        V = np.array([[ (c * i * j) % MOD for j in range(4) ] for i in range(4)])
        if _cost(V) == 0:
            found = V
            break
            
    if found is None:
        # Hill Climbing
        V = np.random.randint(0, MOD, size=(4, 4))
        V[0, 0] = 0 # Fix gauge
        current_cost = _cost(V)
        
        for _ in range(10000): # Max iterations
            if time.time() > deadline - 10:
                break
            if current_cost == 0:
                found = V
                break
            
            # Pick random variable
            i = random.randint(0, 3)
            j = random.randint(0, 3)
            if i == 0 and j == 0:
                continue
                
            best_val = V[i, j]
            best_c = current_cost
            
            for val in range(MOD):
                if val == best_val: continue
                old = V[i, j]
                V[i, j] = val
                new_c = _cost(V)
                if new_c < best_c:
                    best_c = new_c
                    best_val = val
                V[i, j] = old
                
            V[i, j] = best_val
            current_cost = best_c
            
            # If stuck, random restart
            if random.random() < 0.01:
                V = np.random.randint(0, MOD, size=(4, 4))
                V[0, 0] = 0
                current_cost = _cost(V)

    if found is not None:
        n, edges = _build_lift(found)
        if _verify_girth_np(n, edges, g, deadline):
            return n, edges

    return _luw_construct(4, 10, deadline)
