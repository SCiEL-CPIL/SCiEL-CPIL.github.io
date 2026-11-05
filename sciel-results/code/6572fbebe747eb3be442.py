import time
import random
from itertools import combinations

def is_not_2_colorable(edges, n):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    """
    if not edges:
        return False
        
    edges = [tuple(sorted(e)) for e in edges]
    
    adj = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            adj[v].append(i)
            
    color = [-1] * n
    
    def propagate(color):
        changed = True
        while changed:
            changed = False
            for e_idx in range(len(edges)):
                e = edges[e_idx]
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    if color[v] == 0:
                        c0 += 1
                    elif color[v] == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                if c0 == 5 or c1 == 5:
                    return False
                
                if len(unassigned) == 1:
                    v = unassigned[0]
                    if c0 == 4:
                        if color[v] == 1:
                            return False
                        if color[v] == -1:
                            color[v] = 1
                            changed = True
                    elif c1 == 4:
                        if color[v] == 0:
                            return False
                        if color[v] == -1:
                            color[v] = 0
                            changed = True
                elif len(unassigned) == 0:
                    if c0 == 5 or c1 == 5:
                        return False
        return True

    def backtrack(start_v):
        v = -1
        for i in range(start_v, n):
            if color[i] == -1:
                v = i
                break
        
        if v == -1:
            for e in edges:
                c0 = sum(1 for x in e if color[x] == 0)
                if c0 == 5:
                    return False
            return True
        
        color[v] = 0
        if propagate(color):
            if backtrack(v + 1):
                return True
        color[v] = -1
        
        color[v] = 1
        if propagate(color):
            if backtrack(v + 1):
                return True
        color[v] = -1
        
        return False

    if not propagate(color):
        return True
        
    if backtrack(0):
        return False
    else:
        return True

def minimize_edges(edges, n, time_limit):
    current_edges = list(edges)
    changed = True
    start_time = time.time()
    
    while changed and time.time() - start_time < time_limit:
        changed = False
        random.shuffle(current_edges)
        
        for i in range(len(current_edges)):
            if time.time() - start_time > time_limit:
                break
            candidate_edges = current_edges[:i] + current_edges[i+1:]
            if is_not_2_colorable(candidate_edges, n):
                current_edges = candidate_edges
                changed = True
                break
    
    return current_edges

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_edges = None
    best_n = 9
    best_count = float('inf')
    
    edges_9 = [list(e) for e in combinations(range(9), 5)]
    
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        
        n = random.choice([10, 11, 12])
        e_target = random.randint(30, 50)
        
        all_possible = list(combinations(range(n), 5))
        random.shuffle(all_possible)
        candidate_edges = [list(e) for e in all_possible[:e_target]]
        
        if is_not_2_colorable(candidate_edges, n):
            remaining = time_limit - (time.time() - start_time)
            if remaining > 5:
                min_edges = minimize_edges(candidate_edges, n, remaining)
                if len(min_edges) < best_count:
                    best_count = len(min_edges)
                    best_edges = min_edges
                    best_n = n
                    if best_count <= 35:
                        break
        
        if time.time() - start_time > time_limit - 10:
            break

    if best_edges is None:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 10:
            best_edges = minimize_edges(edges_9, 9, remaining)
            best_n = 9
        else:
            best_edges = edges_9
            best_n = 9
        best_count = len(best_edges)
        
    return {'n': best_n, 'edges': best_edges}
