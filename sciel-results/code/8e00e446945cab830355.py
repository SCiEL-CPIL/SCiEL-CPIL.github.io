"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 142.0  # safety margin before 150s
    
    d = 7
    n = 2 ** d  # 128 vertices
    
    # Precompute all edges and their properties
    edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    m = len(edges)  # 448
    
    # Map edge to index
    edge_to_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = idx
    
    # For each edge, precompute the "conflict" information
    # Edge e=(u,v) along dim i. For each j != i, the C4 is:
    #   u-v, v-v^j, v^j-u^j, u^j-u
    # The three other edges are:
    #   (v, v^j) along dim j
    #   (v^j, u^j) along dim i  
    #   (u^j, u) along dim j
    
    # Precompute for each edge, list of (e1_idx, e2_idx, e3_idx) conflict triples
    conflict_triples = [[] for _ in range(m)]
    
    for idx, (u, v) in enumerate(edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            e1 = (min(v, v2), max(v, v2))
            e2 = (min(v2, u2), max(v2, u2))
            e3 = (min(u2, u), max(u2, u))
            if e1 in edge_to_idx and e2 in edge_to_idx and e3 in edge_to_idx:
                conflict_triples[idx].append((edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3]))
    
    # State: boolean array of length m, True if edge is selected
    # Start with greedy
    random.seed(42)
    perm = list(range(m))
    random.shuffle(perm)
    
    selected = np.zeros(m, dtype=bool)
    for idx in perm:
        ok = True
        for (e1, e2, e3) in conflict_triples[idx]:
            if selected[e1] and selected[e2] and selected[e3]:
                ok = False
                break
        if ok:
            selected[idx] = True
    
    best_selected = selected.copy()
    best_count = selected.sum()
    
    # Local search with simulated annealing
    T = 20.0
    T_min = 0.01
    cooling = 0.9998
    
    cur = selected.copy()
    cur_count = cur.sum()
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            break
        
        # Random move
        r = random.random()
        
        if r < 0.5:
            # Try to add a random unselected edge
            unselected = np.where(~cur)[0]
            if len(unselected) == 0:
                continue
            idx = random.choice(unselected)
            # Check if can add
            ok = True
            for (e1, e2, e3) in conflict_triples[idx]:
                if cur[e1] and cur[e2] and cur[e3]:
                    ok = False
                    break
            if ok:
                cur[idx] = True
                new_count = cur_count + 1
                cur_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best_selected = cur.copy()
            else:
                # Try swap: remove one from conflict, add this
                for (e1, e2, e3) in conflict_triples[idx]:
                    if cur[e1] and cur[e2] and cur[e3]:
                        # Remove one of them
                        rem = random.choice([e1, e2, e3])
                        cur[rem] = False
                        cur[idx] = True
                        # Count unchanged
                        break
        elif r < 0.7:
            # Remove a random selected edge
            selected_edges = np.where(cur)[0]
            if len(selected_edges) == 0:
                continue
            idx = random.choice(selected_edges)
            cur[idx] = False
            new_count = cur_count - 1
            # Accept with prob exp(-(-1)/T) = exp(1/T) ~ always
            if random.random() < np.exp(1.0 / T):
                cur_count = new_count
            else:
                cur[idx] = True
        else:
            # Swap: remove one, add another
            selected_edges = np.where(cur)[0]
            unselected = np.where(~cur)[0]
            if len(selected_edges) == 0 or len(unselected) == 0:
                continue
            idx_rem = random.choice(selected_edges)
            idx_add = random.choice(unselected)
            # Check if idx_add can be added after removing idx_rem
            ok = True
            for (e1, e2, e3) in conflict_triples[idx_add]:
                if e1 == idx_rem or e2 == idx_rem or e3 == idx_rem:
                    continue
                if cur[e1] and cur[e2] and cur[e3]:
                    ok = False
                    break
            if ok:
                cur[idx_rem] = False
                cur[idx_add] = True
                # Count unchanged, accept with prob 0.3
                if random.random() < 0.3:
                    pass
                else:
                    cur[idx_add] = False
                    cur[idx_rem] = True
        
        T = max(T_min, T * cooling)
    
    # Final polish
    for idx in range(m):
        if not best_selected[idx]:
            ok = True
            for (e1, e2, e3) in conflict_triples[idx]:
                if best_selected[e1] and best_selected[e2] and best_selected[e3]:
                    ok = False
                    break
            if ok:
                best_selected[idx] = True
    
    # Convert to list of [u, v]
    result = []
    for idx in range(m):
        if best_selected[idx]:
            u, v = edges[idx]
            result.append([u, v])
    
    return result
# EVOLVE-BLOCK-END
