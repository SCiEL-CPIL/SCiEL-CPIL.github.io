import time
import random
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 115  # Safety margin

def build_groetzsch_adj():
    """
    Returns adjacency sets for the Grötzsch graph (11 vertices).
    Vertices 0-4 form a C5.
    Vertices 5-9 are Mycielski copies of 0-4.
    Vertex 10 is connected to 5-9.
    """
    n = 11
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        
    # C5 edges
    c5_edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    for u, v in c5_edges:
        add_edge(u, v)
        
    # Mycielski construction
    # 5 (copy of 0) connects to N(0) = {1, 4}
    add_edge(5, 1)
    add_edge(5, 4)
    
    # 6 (copy of 1) connects to N(1) = {0, 2}
    add_edge(6, 0)
    add_edge(6, 2)
    
    # 7 (copy of 2) connects to N(2) = {1, 3}
    add_edge(7, 1)
    add_edge(7, 3)
    
    # 8 (copy of 3) connects to N(3) = {2, 4}
    add_edge(8, 2)
    add_edge(8, 4)
    
    # 9 (copy of 4) connects to N(4) = {3, 0}
    add_edge(9, 3)
    add_edge(9, 0)
    
    # 10 connects to 5, 6, 7, 8, 9
    for i in range(5, 10):
        add_edge(10, i)
        
    return adj

def gen_compositions(remaining, num_parts):
    """
    Generator for compositions of 'remaining' into 'num_parts' positive integers.
    """
    if num_parts == 1:
        yield [remaining]
        return
    for first in range(1, remaining - num_parts + 2):
        for rest in gen_compositions(remaining - first, num_parts - 1):
            yield [first] + rest

def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    
    n_vertices = 16
    n_base = 11
    
    base_adj = build_groetzsch_adj()
    
    # Precompute which pairs of vertices in the base graph can be "saturated"
    can_connect = [[False] * n_base for _ in range(n_base)]
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if j in base_adj[i]:
                can_connect[i][j] = True
                can_connect[j][i] = True
            else:
                if not (base_adj[i] & base_adj[j]):
                    can_connect[i][j] = True
                    can_connect[j][i] = True

    best_edges_count = -1
    best_composition = None
    
    # Iterate over all compositions of 16 into 11 parts
    for comp in gen_compositions(n_vertices, n_base):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Calculate edges
        edge_count = 0
        for i in range(n_base):
            for j in range(i + 1, n_base):
                if can_connect[i][j]:
                    edge_count += comp[i] * comp[j]
                    
        if edge_count > best_edges_count:
            best_edges_count = edge_count
            best_composition = comp

    if best_composition is None:
        best_composition = [1]*11
        rem = 16 - 11
        for i in range(rem):
            best_composition[i] += 1

    # Local search: try moving vertices between bags to improve edge count
    # A move: take 1 vertex from bag a, put it in bag b (a != b, both >= 1 before move)
    # This is valid if comp[a] >= 2 (so bag a stays non-empty)
    # Gain = sum over j != a,b of can_connect[a][j]*comp[j] - can_connect[b][j]*comp[j]
    #       + can_connect[a][b]*comp[b] - can_connect[a][b]*comp[b]  (cancels, but comp changes)
    # Actually: removing from a: lose edges to all other bags j: sum_j can_connect[a][j]*comp[j]
    #           adding to b: gain edges to all other bags j: sum_j can_connect[b][j]*comp[j]
    #           but comp[a] decreases by 1, comp[b] increases by 1, so edges between a and b change
    #           old: can_connect[a][b]*comp[a]*comp[b], new: can_connect[a][b]*(comp[a]-1)*(comp[b]+1)
    #           delta_ab = can_connect[a][b]*((comp[a]-1)*(comp[b]+1) - comp[a]*comp[b])
    #                    = can_connect[a][b]*(comp[b] - comp[a])
    #           For j not a,b: old contribution involves comp[a]*comp[j] and comp[b]*comp[j]
    #           new: (comp[a]-1)*comp[j] for a-j, (comp[b]+1)*comp[j] for b-j
    #           delta_j = -can_connect[a][j]*comp[j] + can_connect[b][j]*comp[j]
    #           Total delta = delta_ab + sum_{j != a,b} (can_connect[b][j] - can_connect[a][j])*comp[j]
    
    improved = True
    while improved and time.time() - START_TIME < TIME_LIMIT - 5:
        improved = False
        for a in range(n_base):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            if best_composition[a] < 2:
                continue
            for b in range(n_base):
                if a == b:
                    continue
                # Compute gain
                gain = 0
                for j in range(n_base):
                    if j == a or j == b:
                        continue
                    gain += (can_connect[b][j] - can_connect[a][j]) * best_composition[j]
                # Edges between a and b
                if can_connect[a][b]:
                    gain += best_composition[b] - best_composition[a]
                
                if gain > 0:
                    best_composition[a] -= 1
                    best_composition[b] += 1
                    best_edges_count += gain
                    improved = True
                    break
            if improved:
                break

    # Construct the actual graph edges
    edges = []
    bag_map = [0] * n_vertices
    current_idx = 0
    for i in range(n_base):
        for _ in range(best_composition[i]):
            bag_map[current_idx] = i
            current_idx += 1
            
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            bi = bag_map[i]
            bj = bag_map[j]
            if bi == bj:
                continue
            if can_connect[bi][bj]:
                edges.append([i, j])
                
    return edges
