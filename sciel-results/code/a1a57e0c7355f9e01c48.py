import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    for mask in range(1 << n):
        valid = True
        for e in edges:
            c0 = (mask >> e[0]) & 1
            mono = True
            for i in range(1, 5):
                if ((mask >> e[i]) & 1) != c0:
                    mono = False
                    break
            if mono:
                valid = False
                break
        if valid:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    best_edges = None
    best_count = float('inf')
    best_n = 9
    
    for n in [10, 11]:
        if time.time() > deadline:
            break
            
        all_edges = [list(e) for e in combinations(range(n), 5)]
        
        for attempt in range(3):
            if time.time() > deadline:
                break
                
            edges = all_edges[:]
            random.seed(n * 100 + attempt)
            random.shuffle(edges)
            
            for i in range(len(edges)):
                if time.time() > deadline:
                    break
                    
                e = edges[i]
                new_edges = [x for x in edges if x != e]
                
                if is_2_colorable(n, new_edges):
                    pass
                else:
                    edges = new_edges
            
            if len(edges) < best_count:
                best_count = len(edges)
                best_edges = edges[:]
                best_n = n
                
            if best_count <= 51:
                break

    if best_edges is None:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    if is_2_colorable(best_n, best_edges):
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
