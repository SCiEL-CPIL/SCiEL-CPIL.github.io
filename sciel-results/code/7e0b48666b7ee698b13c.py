import random
import time

def construct():
    deadline = time.time() + 135
    d = 7
    n = 1 << d
    total_edges = n * d // 2  # 448
    
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute conflict triples as edge indices
    conflict_triples_idx = []
    for (u, v) in all_edges:
        diff = u ^ v
        i = diff.bit_length() - 1
        triples = []
        for j in range(d):
            if j == i:
                continue
            w = u ^ (1 << j)
            z = v ^ (1 << j)
            e1 = (min(w,u), max(w,u))
            e2 = (min(v,z), max(v,z))
            e3 = (min(w,z), max(w,z))
            triples.append((edge_index[e1], edge_index[e2], edge_index[e3]))
        conflict_triples_idx.append(triples)
    
    best_count = 0
    best_selected = None
    
    def can_add_idx(idx, selected_arr):
        for triple in conflict_triples_idx[idx]:
            if selected_arr[triple[0]] and selected_arr[triple[1]] and selected_arr[triple[2]]:
                return False
        return True
    
    def local_search(selected_arr, time_budget):
        nonlocal best_count, best_selected
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            for idx in range(total_edges):
                if not selected_arr[idx] and can_add_idx(idx, selected_arr):
                    selected_arr[idx] = True
                    improved = True
            cnt = sum(selected_arr)
            if cnt > best_count:
                best_count = cnt
                best_selected = selected_arr.copy()
            
            if not improved:
                for idx in range(total_edges):
                    if selected_arr[idx]:
                        selected_arr[idx] = False
                        added = 0
                        for idx2 in range(total_edges):
                            if not selected_arr[idx2] and can_add_idx(idx2, selected_arr):
                                selected_arr[idx2] = True
                                added += 1
                        cnt = sum(selected_arr)
                        if cnt > best_count:
                            best_count = cnt
                            best_selected = selected_arr.copy()
                        if added > 0:
                            improved = True
                        else:
                            selected_arr[idx] = True
                        if time.time() > time_budget:
                            return
    
    rng = random.Random(42)
    for trial in range(1000):
        if time.time() > deadline:
            break
        order = list(range(total_edges))
        rng.shuffle(order)
        selected_arr = [False] * total_edges
        for idx in order:
            if can_add_idx(idx, selected_arr):
                selected_arr[idx] = True
        local_search(selected_arr, deadline)
    
    if best_selected is None:
        return []
    return [[all_edges[i][0], all_edges[i][1]] for i in range(total_edges) if best_selected[i]]
