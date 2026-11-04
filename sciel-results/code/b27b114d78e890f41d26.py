import time
import sys
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # Petersen Graph Definition
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    max_walk_len = g - 1
    closed_walks = []
    
    def generate_closed_walks(current, prev, start, depth, path):
        if depth > max_walk_len:
            return
        if depth >= 3 and current == start:
            closed_walks.append(tuple(path))
        
        for nxt in adj[current]:
            if nxt == prev:
                continue
            generate_closed_walks(nxt, current, start, depth + 1, path + [nxt])
            
    for v in range(10):
        generate_closed_walks(v, -1, v, 1, [v])
        
    # Optional: Deduplicate closed walks to speed up check
    # Normalize: rotate so smallest vertex is first, and second vertex is smallest possible
    unique_walks = set()
    for w in closed_walks:
        # w is (v0, ..., vL) with vL == v0
        # Length L = len(w) - 1
        # Try all rotations
        min_rot = None
        for i in range(len(w) - 1):
            # Rotate so w[i] is first
            # New walk: w[i], w[i+1], ..., w[L-1], w[0], ..., w[i-1]
            # But w[L] is w[0].
            # The sequence of vertices is cyclic.
            # We just need to check if this cyclic sequence is in the set.
            # To keep it simple, just add tuple(w) to set.
            # Rotations are distinct tuples but represent same cycle.
            # Checking all is fine as count is small.
            pass
        
        unique_walks.add(tuple(w))
        
    # Actually, let's just use the list, it's small.
    
    for m in range(90, 151):
        if time.time() > deadline - 10:
            break
            
        for attempt in range(500):
            if time.time() > deadline - 5:
                break
                
            voltages = [random.randint(0, m - 1) for _ in range(len(petersen_edges))]
            
            v_map = {}
            for i, (u, v) in enumerate(petersen_edges):
                g_val = voltages[i]
                v_map[(u, v)] = g_val
                v_map[(v, u)] = -g_val
            
            valid = True
            for w in closed_walks:
                s = 0
                for i in range(len(w) - 1):
                    u, v = w[i], w[i+1]
                    s += v_map[(u, v)]
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                n = 10 * m
                edges = []
                for u, v in petersen_edges:
                    g_val = v_map[(u, v)]
                    for h in range(m):
                        u_idx = u * m + h
                        v_idx = v * m + (h + g_val) % m
                        edges.append((u_idx, v_idx))
                return n, edges

    raise Exception("Failed to construct graph")
