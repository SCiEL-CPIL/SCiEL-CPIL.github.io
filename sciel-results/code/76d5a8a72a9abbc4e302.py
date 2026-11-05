import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    d = 7
    n = 2 ** d
    total_edges = d * 2 ** (d - 1)
    
    # Precompute all edges
    edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    
    # Precompute edge index
    edge_set_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_set_idx[(u, v)] = i
    
    # Precompute for each edge, the list of "conflict triples"
    edge_conflicts = [[] for _ in range(len(edges))]
    edge_neighbors = [[] for _ in range(len(edges))]
    
    for idx, (u, v) in enumerate(edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(u2, v2), max(u2, v2))
            e3 = (min(v2, v), max(v2, v))
            i1 = edge_set_idx.get(e1)
            i2 = edge_set_idx.get(e2)
            i3 = edge_set_idx.get(e3)
            if i1 is not None and i2 is not None and i3 is not None:
                edge_conflicts[idx].append((i1, i2, i3))
                edge_neighbors[idx].append(i1)
                edge_neighbors[idx].append(i2)
                edge_neighbors[idx].append(i3)
    
    for idx in range(len(edges)):
        edge_neighbors[idx] = list(set(edge_neighbors[idx]))
    
    # Better greedy: try multiple random orderings, keep best
    def greedy_init(order):
        sel = [False] * len(edges)
        cnt = 0
        for idx in order:
            ok = True
            for (i1, i2, i3) in edge_conflicts[idx]:
                if sel[i1] and sel[i2] and sel[i3]:
                    ok = False
                    break
            if ok:
                sel[idx] = True
                cnt += 1
        return sel, cnt
    
    best_sel = None
    best_cnt = 0
    for trial in range(5):
        order = list(range(len(edges)))
        random.shuffle(order)
        sel, cnt = greedy_init(order)
        if cnt > best_cnt:
            best_cnt = cnt
            best_sel = sel
    
    selected = best_sel[:]
    selected_count = best_cnt
    best_count = selected_count
    best_selected = selected[:]
    
    def can_add(idx):
        for (i1, i2, i3) in edge_conflicts[idx]:
            if selected[i1] and selected[i2] and selected[i3]:
                return False
        return True
    
    def local_search_pass():
        nonlocal selected, selected_count, best_count, best_selected
        
        improved = True
        max_passes = 30
        passes = 0
        
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            
            edge_order = list(range(len(edges)))
            random.shuffle(edge_order)
            
            for e_out in edge_order:
                if time.time() > deadline:
                    return
                
                if not selected[e_out]:
                    continue
                
                selected[e_out] = False
                selected_count -= 1
                
                candidates = set()
                for neighbor in edge_neighbors[e_out]:
                    if not selected[neighbor]:
                        candidates.add(neighbor)
                
                candidate_list = list(candidates)
                random.shuffle(candidate_list)
                
                added_count = 0
                for c in candidate_list:
                    if can_add(c):
                        selected[c] = True
                        selected_count += 1
                        added_count += 1
                
                if added_count >= 1:
                    improved = True
                    if selected_count > best_count:
                        best_count = selected_count
                        best_selected = selected[:]
                else:
                    selected[e_out] = True
                    selected_count += 1
    
    local_search_pass()
    
    # 2-swap local search: remove 2 edges, try to add 2 or more
    def two_swap_search():
        nonlocal selected, selected_count, best_count, best_selected
        
        improved = True
        max_passes = 20
        passes = 0
        
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            
            selected_indices = [i for i in range(len(edges)) if selected[i]]
            if len(selected_indices) < 2:
                break
            
            random.shuffle(selected_indices)
            
            for ai in range(min(len(selected_indices) - 1, 200)):
                if time.time() > deadline:
                    return
                
                a = selected_indices[ai]
                b = selected_indices[ai + 1]
                
                if a == b:
                    continue
                
                # Remove both
                selected[a] = False
                selected[b] = False
                selected_count -= 2
                
                # Collect candidates
                candidates = set()
                for neighbor in edge_neighbors[a]:
                    if not selected[neighbor]:
                        candidates.add(neighbor)
                for neighbor in edge_neighbors[b]:
                    if not selected[neighbor]:
                        candidates.add(neighbor)
                
                candidate_list = list(candidates)
                random.shuffle(candidate_list)
                
                added_count = 0
                added_edges = []
                for c in candidate_list:
                    if can_add(c):
                        selected[c] = True
                        selected_count += 1
                        added_count += 1
                        added_edges.append(c)
                
                if added_count >= 2:
                    improved = True
                    if selected_count > best_count:
                        best_count = selected_count
                        best_selected = selected[:]
                else:
                    # Revert
                    selected[a] = True
                    selected[b] = True
                    selected_count += 2
                    for c in added_edges:
                        selected[c] = False
                        selected_count -= 1
    
    two_swap_search()
    
    # Simulated Annealing with proper revert
    T = 1.0
    cooling_rate = 0.9995
    min_T = 0.001
    
    random.seed(42)
    
    sa_steps = 0
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.35:
            # Remove one selected edge, try to add neighbors
            selected_indices = [i for i in range(len(edges)) if selected[i]]
            if not selected_indices:
                continue
            idx = random.choice(selected_indices)
            
            selected[idx] = False
            selected_count -= 1
            
            candidates = [n for n in edge_neighbors[idx] if not selected[n]]
            random.shuffle(candidates)
            
            added = 0
            added_list = []
            for c in candidates:
                if can_add(c):
                    selected[c] = True
                    selected_count += 1
                    added += 1
                    added_list.append(c)
            
            delta = added - 1
            
            if delta >= 0 or (delta < 0 and random.random() < np.exp(-abs(delta) / max(T, 1e-10))):
                if selected_count > best_count:
                    best_count = selected_count
                    best_selected = selected[:]
            else:
                selected[idx] = True
                selected_count += 1
                for c in added_list:
                    selected[c] = False
                    selected_count -= 1
                    
        elif move_type < 0.65:
            # 2-swap: remove 2, add 2
            selected_indices = [i for i in range(len(edges)) if selected[i]]
            if len(selected_indices) < 2:
                continue
            
            a, b = random.sample(selected_indices, 2)
            
            selected[a] = False
            selected[b] = False
            selected_count -= 2
            
            candidates = set()
            for neighbor in edge_neighbors[a]:
                if not selected[neighbor]:
                    candidates.add(neighbor)
            for neighbor in edge_neighbors[b]:
                if not selected[neighbor]:
                    candidates.add(neighbor)
            
            candidate_list = list(candidates)
            random.shuffle(candidate_list)
            
            added = 0
            added_list = []
            for c in candidate_list:
                if can_add(c):
                    selected[c] = True
                    selected_count += 1
                    added += 1
                    added_list.append(c)
            
            delta = added - 2
            
            if delta >= 0 or (delta < 0 and random.random() < np.exp(-abs(delta) / max(T, 1e-10))):
                if selected_count > best_count:
                    best_count = selected_count
                    best_selected = selected[:]
            else:
                selected[a] = True
                selected[b] = True
                selected_count += 2
                for c in added_list:
                    selected[c] = False
                    selected_count -= 1
                    
        else:
            # Single add if possible
            unselected_indices = [i for i in range(len(edges)) if not selected[i]]
            if not unselected_indices:
                continue
            idx = random.choice(unselected_indices)
            if can_add(idx):
                selected[idx] = True
                selected_count += 1
                if selected_count > best_count:
                    best_count = selected_count
                    best_selected = selected[:]
        
        T *= cooling_rate
        if T < min_T:
            T = min_T
        
        sa_steps += 1
        if sa_steps % 200 == 0 and time.time() < deadline:
            local_search_pass()
    
    # Final polish
    if time.time() < deadline:
        local_search_pass()
    if time.time() < deadline:
        two_swap_search()
    
    final_selected = best_selected
    result = []
    for idx in range(len(edges)):
        if final_selected[idx]:
            result.append([edges[idx][0], edges[idx][1]])
    
    return result
