"""K(2,2,2)-free 3-graph on 16 vertices: maximize triples."""
import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
NUM_TRIPLES = len(ALL_TRIPLES)

def _precompute():
    triple_to_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}
    struct_to_triples = []
    triple_to_structs = [[] for _ in range(NUM_TRIPLES)]
    
    for points in combinations(range(V), 6):
        p0, p1, p2, p3, p4, p5 = points
        pairings = [
            ((0,1),(2,3),(4,5)), ((0,1),(2,4),(3,5)), ((0,1),(2,5),(3,4)),
            ((0,2),(1,3),(4,5)), ((0,2),(1,4),(3,5)), ((0,2),(1,5),(3,4)),
            ((0,3),(1,2),(4,5)), ((0,3),(1,4),(2,5)), ((0,3),(1,5),(2,4)),
            ((0,4),(1,2),(3,5)), ((0,4),(1,3),(2,5)), ((0,4),(1,5),(2,3)),
            ((0,5),(1,2),(3,4)), ((0,5),(1,3),(2,4)), ((0,5),(1,4),(2,3)),
        ]
        for pairing in pairings:
            p1a, p1b = points[pairing[0][0]], points[pairing[0][1]]
            p2a, p2b = points[pairing[1][0]], points[pairing[1][1]]
            p3a, p3b = points[pairing[2][0]], points[pairing[2][1]]
            
            triples_in_struct = []
            for a in (p1a, p1b):
                for b in (p2a, p2b):
                    for c in (p3a, p3b):
                        t = tuple(sorted((a, b, c)))
                        idx = triple_to_idx[t]
                        triples_in_struct.append(idx)
            
            s_id = len(struct_to_triples)
            struct_to_triples.append(triples_in_struct)
            for idx in triples_in_struct:
                triple_to_structs[idx].append(s_id)
    
    return struct_to_triples, triple_to_structs

STRUCT_TO_TRIPLES, TRIPLE_TO_STRUCTS = _precompute()
NUM_STRUCTS = len(STRUCT_TO_TRIPLES)

def _is_valid_add(struct_count, t_idx):
    for s in TRIPLE_TO_STRUCTS[t_idx]:
        if struct_count[s] == 7:
            return False
    return True

def _greedy_randomized(timeout=None):
    best_size = 0
    best_result = None
    
    if timeout is None:
        timeout = time.time() + 110
    
    for restart in range(100):
        if time.time() > timeout:
            break
        
        idx_list = list(range(NUM_TRIPLES))
        random.shuffle(idx_list)
        
        present = [False] * NUM_TRIPLES
        struct_count = [0] * NUM_STRUCTS
        count = 0
        
        for idx in idx_list:
            if _is_valid_add(struct_count, idx):
                present[idx] = True
                count += 1
                for s in TRIPLE_TO_STRUCTS[idx]:
                    struct_count[s] += 1
        
        if count > best_size:
            best_size = count
            best_result = [list(ALL_TRIPLES[i]) for i in range(NUM_TRIPLES) if present[i]]
        
        if time.time() > timeout:
            break
        
        absent = [i for i in range(NUM_TRIPLES) if not present[i]]
        random.shuffle(absent)
        
        for t_idx in absent[:30]:
            if time.time() > timeout:
                break
            blocking = [s for s in TRIPLE_TO_STRUCTS[t_idx] if struct_count[s] == 7]
            if not blocking:
                continue
            
            for s in blocking:
                candidates = [i for i in STRUCT_TO_TRIPLES[s] if present[i]]
                if not candidates:
                    continue
                for u_idx in candidates:
                    can_swap = True
                    for s2 in TRIPLE_TO_STRUCTS[t_idx]:
                        if struct_count[s2] == 7 and u_idx not in STRUCT_TO_TRIPLES[s2]:
                            can_swap = False
                            break
                    if can_swap:
                        present[u_idx] = False
                        for s3 in TRIPLE_TO_STRUCTS[u_idx]:
                            struct_count[s3] -= 1
                        present[t_idx] = True
                        for s3 in TRIPLE_TO_STRUCTS[t_idx]:
                            struct_count[s3] += 1
                        break
                break
        
        if count > best_size:
            best_size = count
            best_result = [list(ALL_TRIPLES[i]) for i in range(NUM_TRIPLES) if present[i]]
    
    return best_result if best_result is not None else []

def construct():
    deadline = time.time() + 115
    return _greedy_randomized(timeout=deadline)
