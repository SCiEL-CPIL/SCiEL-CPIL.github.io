import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n = 11
    all_edges = list(combinations(range(n), 5))
    
    # Fast 2-colorability check with unit propagation
    def is_2_colorable(edges):
        colors = [-1] * n
        
        def propagate():
            changed = True
            while changed:
                changed = False
                for e in edges:
                    c0 = 0
                    c1 = 0
                    unknown = -1
                    for v in e:
                        if colors[v] == 0:
                            c0 += 1
                        elif colors[v] == 1:
                            c1 += 1
                        else:
                            unknown = v
                    
                    if c0 == 4:
                        if colors[unknown] == 0:
                            return False
                        elif colors[unknown] == -1:
                            colors[unknown] = 1
                            changed = True
                    elif c1 == 4:
                        if colors[unknown] == 1:
                            return False
                        elif colors[unknown] == -1:
                            colors[unknown] = 0
                            changed = True
            return True

        def backtrack():
            if not propagate():
                return False
            
            v = -1
            for i in range(n):
                if colors[i] == -1:
                    v = i
                    break
            
            if v == -1:
                return True
            
            colors[v] = 0
            if backtrack():
                return True
            colors[v] = 1
            if backtrack():
                return True
            
            colors[v] = -1
            return False

        return backtrack()

    def get_2_coloring(edges):
        colors = [-1] * n
        
        def propagate():
            changed = True
            while changed:
                changed = False
                for e in edges:
                    c0 = 0
                    c1 = 0
                    unknown = -1
                    for v in e:
                        if colors[v] == 0:
                            c0 += 1
                        elif colors[v] == 1:
                            c1 += 1
                        else:
                            unknown = v
                    
                    if c0 == 4:
                        if colors[unknown] == 0:
                            return False
                        elif colors[unknown] == -1:
                            colors[unknown] = 1
                            changed = True
                    elif c1 == 4:
                        if colors[unknown] == 1:
                            return False
                        elif colors[unknown] == -1:
                            colors[unknown] = 0
                            changed = True
            return True

        def backtrack():
            if not propagate():
                return False
            
            v = -1
            for i in range(n):
                if colors[i] == -1:
                    v = i
                    break
            
            if v == -1:
                return True
            
            colors[v] = 0
            if backtrack():
                return True
            colors[v] = 1
            if backtrack():
                return True
            
            colors[v] = -1
            return False

        if backtrack():
            return colors
        return None

    best_edges = None
    best_count = float('inf')
    
    # Multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Start with random edges
        initial_size = 30
        current_edges = set(random.sample(all_edges, initial_size))
        
        # Phase 1: Add edges until non-2-colorable
        while time.time() < deadline:
            if not is_2_colorable(list(current_edges)):
                break
            coloring = get_2_coloring(list(current_edges))
            if coloring is None:
                break
            
            candidates = []
            for e in all_edges:
                if e not in current_edges:
                    c = coloring[e[0]]
                    if all(coloring[v] == c for v in e):
                        candidates.append(e)
            
            if not candidates:
                break
            
            # Pick edge that maximizes "impact"? 
            # For now, random
            new_edge = random.choice(candidates)
            current_edges.add(new_edge)
            
        # Phase 2: Remove edges
        if not is_2_colorable(list(current_edges)):
            improved = True
            while improved:
                improved = False
                edges_list = list(current_edges)
                random.shuffle(edges_list)
                for e in edges_list:
                    if time.time() > deadline:
                        break
                    new_edges = current_edges - {e}
                    if not is_2_colorable(list(new_edges)):
                        current_edges = new_edges
                        improved = True
                        if len(current_edges) < best_count:
                            best_count = len(current_edges)
                            best_edges = set(current_edges)
                if time.time() > deadline:
                    break

    if best_edges is None:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': n, 'edges': [list(e) for e in best_edges]}
