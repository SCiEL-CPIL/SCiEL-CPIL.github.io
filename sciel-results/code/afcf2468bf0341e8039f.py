import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Seed graph: K_{3,13}
    base_edges = set()
    for a in range(3):
        for b in range(3, 16):
            base_edges.add((a, b) if a < b else (b, a))
            
    best_edges = list(base_edges)
    best_count = len(base_edges)
    
    all_possible_edges = []
    for i in range(16):
        for j in range(i+1, 16):
            if (i, j) not in base_edges:
                all_possible_edges.append((i, j))
                
    def has_q3_containing_edge(graph_edges, u, v):
        other_vertices = [x for x in range(16) if x != u and x != v]
        for S in itertools.combinations(other_vertices, 6):
            V = [u, v] + list(S)
            V_set = set(V)
            
            N_u = set()
            N_v = set()
            for x in V:
                if x == u or x == v: continue
                if (u, x) in graph_edges or (x, u) in graph_edges:
                    N_u.add(x)
                if (v, x) in graph_edges or (x, v) in graph_edges:
                    N_v.add(x)
            
            N_u_list = list(N_u)
            N_v_list = list(N_v)
            
            if len(N_u_list) < 2 or len(N_v_list) < 2:
                continue
                
            for v1, v2 in itertools.combinations(N_u_list, 2):
                for u1, u2 in itertools.combinations(N_v_list, 2):
                    if v1 == u1 or v1 == u2 or v2 == u1 or v2 == u2:
                        continue
                    
                    used = {u, v, v1, v2, u1, u2}
                    remaining = [x for x in V if x not in used]
                    if len(remaining) != 2:
                        continue
                    w1, w2 = remaining
                    
                    if not ((w1, w2) in graph_edges or (w2, w1) in graph_edges):
                        continue
                    
                    if not ((w1, v1) in graph_edges or (v1, w1) in graph_edges):
                        continue
                    if not ((w1, v2) in graph_edges or (v2, w1) in graph_edges):
                        continue
                    
                    if not ((w2, u1) in graph_edges or (u1, w2) in graph_edges):
                        continue
                    if not ((w2, u2) in graph_edges or (u2, w2) in graph_edges):
                        continue
                    
                    ok1 = ((v1, u1) in graph_edges or (u1, v1) in graph_edges) and \
                          ((v2, u2) in graph_edges or (u2, v2) in graph_edges)
                    ok2 = ((v1, u2) in graph_edges or (u2, v1) in graph_edges) and \
                          ((v2, u1) in graph_edges or (u1, v2) in graph_edges)
                    
                    if ok1 or ok2:
                        return True
                        
        return False

    # Try multiple random orders
    for _ in range(50):
        if time.time() > deadline:
            break
            
        random.shuffle(all_possible_edges)
        current_edges = set(base_edges)
        
        for edge in all_possible_edges:
            if time.time() > deadline:
                break
            u, v = edge
            test_edges = current_edges | {(u, v) if u < v else (v, u)}
            
            if not has_q3_containing_edge(test_edges, u, v):
                current_edges.add((u, v) if u < v else (v, u))
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            
    return [[u, v] for u, v in best_edges]
