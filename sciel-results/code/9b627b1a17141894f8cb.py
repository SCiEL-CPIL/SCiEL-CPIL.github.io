import itertools
import random

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(POINTS)
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}

def dist2(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Precompute isosceles triples
triple_list = []
vertex_to_triples = {i: [] for i in range(NUM_POINTS)}

for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        p_i = POINTS[i]
        p_j = POINTS[j]
        d_ij = dist2(p_i, p_j)
        for k in range(j + 1, NUM_POINTS):
            p_k = POINTS[k]
            d_ik = dist2(p_i, p_k)
            d_jk = dist2(p_j, p_k)
            
            if d_ik == d_jk or d_ik == d_ij or d_jk == d_ij:
                t_idx = len(triple_list)
                triple_list.append((i, j, k))
                vertex_to_triples[i].append(t_idx)
                vertex_to_triples[j].append(t_idx)
                vertex_to_triples[k].append(t_idx)

def construct():
    H = set()
    available = set(range(NUM_POINTS))
    
    cover_count = [0] * NUM_POINTS
    for v in available:
        cover_count[v] = len(vertex_to_triples[v])
        
    uncovered_triples = set(range(len(triple_list)))
    
    while uncovered_triples:
        best_v = -1
        min_count = float('inf')
        
        # Prioritize degree 1 vertices (Leaves)
        for v in available:
            if cover_count[v] == 1:
                best_v = v
                break
                
        if best_v == -1:
            for v in available:
                if cover_count[v] > 0 and cover_count[v] < min_count:
                    min_count = cover_count[v]
                    best_v = v
                    
        if best_v == -1:
            break
            
        H.add(best_v)
        available.remove(best_v)
        
        for t_idx in vertex_to_triples[best_v]:
            if t_idx in uncovered_triples:
                uncovered_triples.remove(t_idx)
                i, j, k = triple_list[t_idx]
                for u in (i, j, k):
                    if u != best_v:
                        cover_count[u] -= 1
                        
    S = [POINTS[q] for q in range(NUM_POINTS) if q not in H]
    return S
