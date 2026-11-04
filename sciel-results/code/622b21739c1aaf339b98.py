import numpy as np
import itertools
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    D = 11
    subsets = list(itertools.combinations(range(D), 4))
    n = len(subsets)  # 330
    
    masks = []
    for s in subsets:
        m = 0
        for pos in s:
            m |= (1 << pos)
        masks.append(m)
        
    adj = [set() for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            inter = (masks[i] & masks[j]).bit_count()
            if inter <= 2:
                adj[i].add(j)
                adj[j].add(i)
                
    best_clique = []
    
    def color_sort(P):
        if not P:
            return []
        
        P_set = set(P)
        degs = []
        for node in P:
            deg = 0
            for nb in adj[node]:
                if nb in P_set:
                    deg += 1
            degs.append((deg, node))
            
        degs.sort(reverse=True)
        sorted_P = [node for _, node in degs]
        
        color_map = {}
        result = []
        
        for v in sorted_P:
            used_colors = set()
            for u in adj[v]:
                if u in color_map:
                    used_colors.add(color_map[u])
            c = 1
            while c in used_colors:
                c += 1
            color_map[v] = c
            result.append((v, c))
            
        result.sort(key=lambda x: x[1])
        return result

    def expand(R, P):
        global best_clique
        
        if time.time() > deadline:
            return
            
        if len(P) == 0:
            if len(R) > len(best_clique):
                best_clique = R.copy()
            return
            
        colored_P = color_sort(P)
        m = len(colored_P)
        
        for i in range(m - 1, -1, -1):
            v, c = colored_P[i]
            
            if len(R) + c <= len(best_clique):
                break
                
            if time.time() > deadline:
                return
            
            new_R = R + [v]
            
            new_P = []
            for node, _ in colored_P[:i]:
                if node in adj[v]:
                    new_P.append(node)
            
            expand(new_R, new_P)
            
    expand([], list(range(n)))
    
    sel_indices = best_clique
    
    vectors = []
    for idx in sel_indices:
        s = subsets[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)
            
    if len(vectors) == 0:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    return np.array(vectors, dtype=np.int64)
