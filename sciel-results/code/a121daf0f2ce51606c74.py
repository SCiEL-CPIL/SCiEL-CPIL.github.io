import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_graph(edges, max_restarts=5):
        best_paths = None
        best_count = float('inf')
        
        for restart in range(max_restarts):
            # Randomize the initial order of paths to break symmetry in merging
            initial_paths = [list(e) for e in edges]
            random.shuffle(initial_paths)
            
            active_paths = initial_paths[:]
            
            changed = True
            while changed:
                changed = False
                endpoint_map = {}
                for i, p in enumerate(active_paths):
                    if not p: continue
                    u, v = p[0], p[-1]
                    endpoint_map.setdefault(u, set()).add(i)
                    endpoint_map.setdefault(v, set()).add(i)
                
                # Randomize vertex order
                vertices = list(endpoint_map.keys())
                random.shuffle(vertices)
                
                merged = False
                for v in vertices:
                    if v not in endpoint_map: continue
                    p_indices = list(endpoint_map[v])
                    if len(p_indices) < 2:
                        continue
                    random.shuffle(p_indices)
                    
                    for i in range(len(p_indices)):
                        for j in range(i+1, len(p_indices)):
                            idx1 = p_indices[i]
                            idx2 = p_indices[j]
                            
                            if idx1 >= len(active_paths) or idx2 >= len(active_paths):
                                continue
                            
                            p1 = active_paths[idx1]
                            p2 = active_paths[idx2]
                            
                            if not p1 or not p2:
                                continue
                            
                            # Check merge
                            if v in (p1[0], p1[-1]) and v in (p2[0], p2[-1]):
                                s1 = set(p1)
                                s2 = set(p2)
                                if s1.intersection(s2) == {v}:
                                    # Merge
                                    if p1[0] == v:
                                        p1 = p1[::-1]
                                    if p2[-1] == v:
                                        p2 = p2[::-1]
                                    
                                    new_path = p1 + p2[1:]
                                    
                                    new_active = []
                                    for k in range(len(active_paths)):
                                        if k == idx1 or k == idx2:
                                            continue
                                        new_active.append(active_paths[k])
                                    new_active.append(new_path)
                                    active_paths = new_active
                                    merged = True
                                    changed = True
                                    break
                        if merged:
                            break
                    if merged:
                        break
            
            if len(active_paths) < best_count:
                best_count = len(active_paths)
                best_paths = active_paths[:]
                
            if time.time() > deadline - 10:
                break
                
        if best_paths is None:
            best_paths = [list(e) for e in edges]
            
        return best_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline - 5:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        paths = solve_graph(edges)
        results.append(paths)
        
    return results
