import random
import time

def construct():
    start_time = time.time()
    deadline = 115.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_graph(edges, time_limit):
        if not edges:
            return []
        
        best_paths = [list(e) for e in edges]
        best_count = len(edges)
        
        while time.time() < time_limit:
            # State: list of (path_tuple, vertex_set)
            paths = [(tuple(e), frozenset(e)) for e in edges]
            random.shuffle(paths)
            
            changed = True
            while changed:
                changed = False
                random.shuffle(paths)
                n_paths = len(paths)
                merged = False
                
                for i in range(n_paths):
                    p1, s1 = paths[i]
                    v1_s, v1_e = p1[0], p1[-1]
                    
                    for j in range(i + 1, n_paths):
                        p2, s2 = paths[j]
                        v2_s, v2_e = p2[0], p2[-1]
                        
                        candidate_path = None
                        candidate_set = None
                        junction = None
                        
                        # Check merges
                        if v1_e == v2_s:
                            junction = v1_e
                            # Intersection should be only junction
                            if s1 & s2 == frozenset([junction]):
                                candidate_path = p1 + p2[1:]
                                candidate_set = s1 | s2
                                
                        elif v1_e == v2_e:
                            junction = v1_e
                            if s1 & s2 == frozenset([junction]):
                                candidate_path = p1 + p2[-2::-1]
                                candidate_set = s1 | s2
                                
                        elif v1_s == v2_s:
                            junction = v1_s
                            if s1 & s2 == frozenset([junction]):
                                candidate_path = p1[-2::-1] + p2
                                candidate_set = s1 | s2
                                
                        elif v1_s == v2_e:
                            junction = v1_s
                            if s1 & s2 == frozenset([junction]):
                                candidate_path = p1[-2::-1] + p2[-2::-1]
                                candidate_set = s1 | s2

                        if candidate_path:
                            new_paths = []
                            for k in range(n_paths):
                                if k != i and k != j:
                                    new_paths.append(paths[k])
                            new_paths.append((candidate_path, candidate_set))
                            paths = new_paths
                            changed = True
                            merged = True
                            break
                    if merged: break
                    
            current_count = len(paths)
            if current_count < best_count:
                best_count = current_count
                best_paths = [list(p) for p, _ in paths]
                
        return best_paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    budgets = [20.0, 30.0, 60.0]
    
    for i, (n, p, seed) in enumerate(specs):
        edges = instance(n, p, seed)
        current_time = time.time()
        remaining = deadline - current_time
        alloc = min(budgets[i], remaining)
        if alloc <= 1.0:
             paths = [list(e) for e in edges]
        else:
             paths = solve_graph(edges, current_time + alloc)
        results.append(paths)

    return results
