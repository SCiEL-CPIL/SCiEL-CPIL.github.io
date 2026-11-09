import time
import random
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 115.0

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def check_2_colorable(n, edges):
    """
    Fast check for 2-colorability.
    Returns True if 2-colorable, False if not.
    """
    if not edges:
        return True

    # Normalize and deduplicate
    unique_edges = []
    seen = set()
    for e in edges:
        e_sorted = tuple(sorted(e))
        if e_sorted not in seen:
            seen.add(e_sorted)
            unique_edges.append(e_sorted)
    edges = unique_edges
    m = len(edges)
    if m == 0:
        return True

    # Build vertex to edge index map
    vert_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(i)

    colors = [-1] * n
    colors[0] = 0 # Symmetry breaking

    def propagate():
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edges):
                c0 = 0
                c1 = 0
                un = -1
                for v in e:
                    cv = colors[v]
                    if cv == 0:
                        c0 += 1
                    elif cv == 1:
                        c1 += 1
                    else:
                        un = v
                
                if un != -1:
                    if c0 == 4:
                        colors[un] = 1
                        changed = True
                    elif c1 == 4:
                        colors[un] = 0
                        changed = True
                else:
                    if c0 == 5 or c1 == 5:
                        return False
        return True

    def backtrack():
        best_v = -1
        max_threat = -1
        for v in range(n):
            if colors[v] == -1:
                threat = 0
                for ei in vert_to_edges[v]:
                    e = edges[ei]
                    c0 = 0
                    c1 = 0
                    un = 0
                    for u in e:
                        cv = colors[u]
                        if cv == 0: c0 += 1
                        elif cv == 1: c1 += 1
                        else: un += 1
                    if c0 + c1 == 4 and un == 1:
                        threat += 1
                if threat > max_threat:
                    max_threat = threat
                    best_v = v
                    if threat > 0:
                        break

        if best_v == -1:
            for e in edges:
                c = colors[e[0]]
                for u in e[1:]:
                    if colors[u] != c:
                        break
                else:
                    return False
            return True

        colors[best_v] = 0
        if propagate():
            if backtrack():
                return True
        
        colors[best_v] = 1
        if propagate():
            if backtrack():
                return True
        
        colors[best_v] = -1
        return False

    if not propagate():
        return False
    return backtrack()

def minimize_from_dense(n, edges):
    """Greedily remove edges from a known non-2-colorable hypergraph."""
    current = list(edges)
    improved = True
    # Shuffle to randomize removal order
    while improved and time_left() > 1:
        improved = False
        idxs = list(range(len(current)))
        random.shuffle(idxs)
        for i in idxs:
            if time_left() < 1:
                break
            test_edges = current[:i] + current[i+1:]
            if not check_2_colorable(n, test_edges):
                current = test_edges
                improved = True
                break
    return current

def construct():
    best_edges = None
    best_n = 9
    best_m = 126

    # Start with K_9^(5)
    initial_edges = [list(e) for e in combinations(range(9), 5)]
    
    # Minimize this
    minimized = minimize_from_dense(9, initial_edges)
    best_edges = minimized
    best_m = len(minimized)
    best_n = 9

    # Try other n if time permits
    if time_left() > 20:
        for n in [10, 11]:
            if time_left() < 10:
                break
            # Generate a dense random hypergraph
            all_edges = list(combinations(range(n), 5))
            m = int(len(all_edges) * 0.3)
            random.shuffle(all_edges)
            edges = [list(e) for e in all_edges[:m]]
            
            if not check_2_colorable(n, edges):
                minimized = minimize_from_dense(n, edges)
                if len(minimized) < best_m:
                    best_edges = minimized
                    best_m = len(minimized)
                    best_n = n

    return {'n': best_n, 'edges': best_edges}
