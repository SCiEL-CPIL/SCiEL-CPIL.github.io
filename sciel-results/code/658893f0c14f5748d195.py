import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    # Q3 adjacency
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }

    # Precompute all 8-subsets
    all_8sets = list(combinations(range(16), 8))

    # Helper to find a Q3 in a specific 8-set
    def find_q3_in_set(S, edge_set):
        # S is a tuple of 8 vertices
        idx_map = {v: i for i, v in enumerate(S)}
        
        edge_count = 0
        deg = [0] * 8
        local_adj = [set() for _ in range(8)]
        
        for i in range(8):
            for j in range(i+1, 8):
                u, v = S[i], S[j]
                if (min(u,v), max(u,v)) in edge_set:
                    edge_count += 1
                    local_adj[i].add(j)
                    local_adj[j].add(i)
                    deg[i] += 1
                    deg[j] += 1
        
        if edge_count < 12:
            return None
        
        for d in deg:
            if d < 3:
                return None
        
        assignment = [-1] * 8
        used_local = [False] * 8
        
        def backtrack(q3_v):
            if q3_v == 8:
                return True
            
            candidates = []
            for li in range(8):
                if used_local[li]:
                    continue
                ok = True
                for nb in q3_adj[q3_v]:
                    if nb < q3_v:
                        nb_li = assignment[nb]
                        if nb_li not in local_adj[li]:
                            ok = False
                            break
                if ok:
                    candidates.append(li)
            
            # Heuristic: try candidates with higher degree first
            candidates.sort(key=lambda x: -deg[x])
            
            for li in candidates:
                assignment[q3_v] = li
                used_local[li] = True
                if backtrack(q3_v + 1):
                    return True
                used_local[li] = False
                assignment[q3_v] = -1
            return False
        
        if not backtrack(0):
            return None
        
        # Extract edges
        q3_global_edges = []
        for u in q3_adj:
            for v in q3_adj[u]:
                if u < v:
                    gu = S[assignment[u]]
                    gv = S[assignment[v]]
                    q3_global_edges.append((min(gu, gv), max(gu, gv)))
        return q3_global_edges

    # Initialize K16
    edge_set = set()
    for i in range(16):
        for j in range(i+1, 16):
            edge_set.add((i, j))
            
    bad_sets = all_8sets[:]
    random.seed(42)
    random.shuffle(bad_sets)
    
    while time.time() < deadline - 2.0 and bad_sets:
        # Pick a random bad set
        idx = random.randint(0, len(bad_sets) - 1)
        S = bad_sets[idx]
        
        q3_edges = find_q3_in_set(S, edge_set)
        
        if q3_edges is None:
            # Set is now Q3-free
            bad_sets.pop(idx)
            continue
        else:
            # Found a Q3. Remove one edge from it.
            e_to_remove = random.choice(q3_edges)
            edge_set.remove(e_to_remove)
            
            # Move the set to the end to avoid immediate re-selection
            # but keep it in the list as it might still contain Q3
            bad_sets[idx], bad_sets[-1] = bad_sets[-1], bad_sets[idx]

    return [[u, v] for (u, v) in edge_set]
