import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s buffer for safety
    
    n = 9
    initial_edges = [list(e) for e in combinations(range(n), 5)]
    current_edges = list(initial_edges)
    
    best_edges = current_edges[:]
    best_len = len(current_edges)
    
    def is_2_colorable_fast(n, edges):
        if not edges:
            return True
            
        # Precompute
        edges_of_vertex = [[] for _ in range(n)]
        pos_in_edge = [[-1]*n for _ in range(len(edges))]
        
        for i, e in enumerate(edges):
            for pos, v in enumerate(e):
                edges_of_vertex[v].append(i)
                pos_in_edge[i][v] = pos

        colors = [-1] * n
        edge_states = [[-1]*5 for _ in range(len(edges))]
        
        def check_rec(colors, edge_states):
            changed = True
            while changed:
                changed = False
                for i in range(len(edges)):
                    st = edge_states[i]
                    c0 = st.count(0)
                    c1 = st.count(1)
                    cu = st.count(-1)
                    
                    if c0 == 5 or c1 == 5:
                        return False
                    
                    if cu == 1:
                        if c0 == 4:
                            idx = st.index(-1)
                            v = edges[i][idx]
                            if colors[v] == -1:
                                colors[v] = 1
                                for ei in edges_of_vertex[v]:
                                    pos = pos_in_edge[ei][v]
                                    edge_states[ei][pos] = 1
                                changed = True
                            elif colors[v] != 1:
                                return False
                        elif c1 == 4:
                            idx = st.index(-1)
                            v = edges[i][idx]
                            if colors[v] == -1:
                                colors[v] = 0
                                for ei in edges_of_vertex[v]:
                                    pos = pos_in_edge[ei][v]
                                    edge_states[ei][pos] = 0
                                changed = True
                            elif colors[v] != 0:
                                return False
            
            unassigned = [v for v in range(n) if colors[v] == -1]
            if not unassigned:
                return True
                
            # Heuristic: pick vertex with most assigned neighbors in edges?
            # For simplicity and speed with small N, pick first.
            v = unassigned[0]
            
            # Try 0
            colors[v] = 0
            for ei in edges_of_vertex[v]:
                pos = pos_in_edge[ei][v]
                edge_states[ei][pos] = 0
                
            if check_rec(colors, edge_states):
                return True
                
            # Undo
            colors[v] = -1
            for ei in edges_of_vertex[v]:
                pos = pos_in_edge[ei][v]
                edge_states[ei][pos] = -1
                
            # Try 1
            colors[v] = 1
            for ei in edges_of_vertex[v]:
                pos = pos_in_edge[ei][v]
                edge_states[ei][pos] = 1
                
            if check_rec(colors, edge_states):
                return True
                
            # Undo
            colors[v] = -1
            for ei in edges_of_vertex[v]:
                pos = pos_in_edge[ei][v]
                edge_states[ei][pos] = -1
                
            return False

        return check_rec(colors, edge_states)

    # Main Loop
    # We perform multiple passes.
    # In each pass, we try to delete edges in random order.
    
    while time.time() < deadline:
        indices = list(range(len(current_edges)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
                
            removed_edge = current_edges.pop(idx)
            
            # Check
            edges_as_tuples = [tuple(e) for e in current_edges]
            
            if is_2_colorable_fast(n, edges_as_tuples):
                current_edges.insert(idx, removed_edge)
            else:
                if len(current_edges) < best_len:
                    best_len = len(current_edges)
                    best_edges = current_edges[:]
                    
    return {'n': n, 'edges': best_edges}
