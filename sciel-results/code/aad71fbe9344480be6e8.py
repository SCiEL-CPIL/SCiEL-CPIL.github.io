import time
import random
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 115.0

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable.
    """
    if not edges:
        return True
    
    m = len(edges)
    if m == 0:
        return True

    edge_verts = [list(e) for e in edges]
    
    vert_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_verts):
        for v in e:
            vert_to_edges[v].append(i)

    colors = [-1] * n
    edge_state = [[0, 0, 5] for _ in range(m)]

    def propagate():
        changed = True
        while changed:
            changed = False
            for i in range(m):
                c0, c1, un = edge_state[i]
                if un == 0:
                    if c0 == 5 or c1 == 5:
                        return False
                elif un == 1:
                    if c0 == 4:
                        for v in edge_verts[i]:
                            if colors[v] == -1:
                                colors[v] = 1
                                edge_state[i][1] += 1
                                edge_state[i][2] -= 1
                                changed = True
                                for ei in vert_to_edges[v]:
                                    if ei != i:
                                        edge_state[ei][1] += 1
                                        edge_state[ei][2] -= 1
                                break
                    elif c1 == 4:
                        for v in edge_verts[i]:
                            if colors[v] == -1:
                                colors[v] = 0
                                edge_state[i][0] += 1
                                edge_state[i][2] -= 1
                                changed = True
                                for ei in vert_to_edges[v]:
                                    if ei != i:
                                        edge_state[ei][0] += 1
                                        edge_state[ei][2] -= 1
                                break
        return True

    def backtrack():
        best_v = -1
        best_score = -1
        for v in range(n):
            if colors[v] == -1:
                critical = 0
                for ei in vert_to_edges[v]:
                    c0, c1, un = edge_state[ei]
                    if un == 1:
                        critical += 1
                score = len(vert_to_edges[v]) + critical * 10
                if score > best_score:
                    best_score = score
                    best_v = v
                    if critical > 0:
                        break

        if best_v == -1:
            for i in range(m):
                c0, c1, un = edge_state[i]
                if un == 0 and (c0 == 5 or c1 == 5):
                    return False
            return True

        for val in [0, 1]:
            saved_colors = colors[:]
            saved_state = [s[:] for s in edge_state]
            
            colors[best_v] = val
            if val == 0:
                for ei in vert_to_edges[best_v]:
                    edge_state[ei][0] += 1
                    edge_state[ei][2] -= 1
            else:
                for ei in vert_to_edges[best_v]:
                    edge_state[ei][1] += 1
                    edge_state[ei][2] -= 1
            
            if propagate():
                if backtrack():
                    return True
            
            colors[:] = saved_colors
            for i in range(m):
                edge_state[i][:] = saved_state[i]
        
        return False

    colors[0] = 0
    for ei in vert_to_edges[0]:
        edge_state[ei][0] += 1
        edge_state[ei][2] -= 1
        
    if not propagate():
        return False
        
    return backtrack()

def construct():
    n = 10
    all_edges = list(combinations(range(n), 5))
    current_edges = [list(e) for e in all_edges]
    
    if is_2_colorable(n, current_edges):
        return {'n': n, 'edges': current_edges}
        
    random.shuffle(current_edges)
    edges_to_keep = list(current_edges)
    
    improved = True
    while improved and time_left() > 5:
        improved = False
        random.shuffle(edges_to_keep)
        
        new_edges = []
        # Iterate through a copy to avoid modification during iteration issues?
        # We are building new_edges from edges_to_keep.
        for e in edges_to_keep:
            if time_left() < 5:
                # Add the rest
                # Find index of e in original list? 
                # Since we shuffled, e is in edges_to_keep.
                # We can just extend with the remainder of the current list?
                # No, we are iterating over edges_to_keep.
                # If we break, we should keep all remaining edges to be safe.
                idx = edges_to_keep.index(e)
                new_edges.extend(edges_to_keep[idx:])
                break
                
            test_edges = [x for x in edges_to_keep if x != e]
            if is_2_colorable(n, test_edges):
                new_edges.append(e)
            else:
                improved = True
                
        if new_edges:
            edges_to_keep = new_edges
        else:
            break

    return {'n': n, 'edges': edges_to_keep}
