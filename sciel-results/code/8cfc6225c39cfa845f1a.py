import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_edges = 0
    best_edge_list = []
    
    for restart in range(5): # 5 restarts
        if time.time() > deadline - 10:
            break
            
        # Initial graph: K_{3,13}
        edges = set()
        for u in range(3):
            for v in range(3, 16):
                edges.add((u, v))
        
        def is_q3(vertices, current_edges):
            if len(vertices) != 8:
                return False
            
            sub_edges = []
            for u, v in itertools.combinations(vertices, 2):
                if (u, v) in current_edges or (v, u) in current_edges:
                    sub_edges.append((u, v))
            
            if len(sub_edges) != 12:
                return False
            
            degrees = {v: 0 for v in vertices}
            for u, v in sub_edges:
                degrees[u] += 1
                degrees[v] += 1
            for d in degrees.values():
                if d != 3:
                    return False
            
            color = {}
            for v in vertices:
                if v not in color:
                    color[v] = 0
                    stack = [v]
                    while stack:
                        curr = stack.pop()
                        for u, w in sub_edges:
                            if u == curr:
                                nxt = w
                            elif w == curr:
                                nxt = u
                            else:
                                continue
                            if nxt not in color:
                                color[nxt] = 1 - color[curr]
                                stack.append(nxt)
                            else:
                                if color[nxt] == color[curr]:
                                    return False
            
            part0 = [v for v in vertices if color[v] == 0]
            part1 = [v for v in vertices if color[v] == 1]
            
            cycle_count = 0
            for p0 in itertools.combinations(part0, 2):
                for p1 in itertools.combinations(part1, 2):
                    if ((p0[0], p1[0]) in current_edges or (p1[0], p0[0]) in current_edges) and \
                       ((p0[1], p1[0]) in current_edges or (p1[0], p0[1]) in current_edges) and \
                       ((p0[1], p1[1]) in current_edges or (p1[1], p0[1]) in current_edges) and \
                       ((p0[0], p1[1]) in current_edges or (p1[1], p0[0]) in current_edges):
                        cycle_count += 1
            
            return cycle_count == 6

        all_pairs = set(itertools.combinations(range(16), 2))
        non_edges = list(all_pairs - edges)
        random.shuffle(non_edges)
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            current_non_edges = [e for e in non_edges if e not in edges]
            for e in current_non_edges:
                if time.time() > deadline - 5:
                    break
                u, v = e
                remaining = [x for x in range(16) if x != u and x != v]
                safe = True
                for combo in itertools.combinations(remaining, 6):
                    verts = frozenset([u, v] + list(combo))
                    # Check if adding (u,v) creates Q3
                    # We simulate adding it
                    test_edges = set(edges)
                    test_edges.add((u, v))
                    if is_q3(verts, test_edges):
                        safe = False
                        break
                
                if safe:
                    edges.add((u, v))
                    changed = True
        
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_edge_list = [list(e) for e in edges]
            
    return best_edge_list
