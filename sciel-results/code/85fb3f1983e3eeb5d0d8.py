import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Fast validity checker using bitmasks
    def is_valid_fast(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Check K4
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    continue
                common = adj[u] & adj[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                
                # Check if common contains an edge
                c = common
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    c ^= lsb
                    if adj[w] & common:
                        return False
        
        # Check Ind6 (K6 in comp)
        comp = [0] * n
        for i in range(n):
            comp[i] = ((1 << n) - 1) ^ adj[i] ^ (1 << i)
            
        for u in range(n):
            for v in range(u + 1, n):
                if not (comp[u] & (1 << v)):
                    continue
                common = comp[u] & comp[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                
                # Check if common contains a K4
                c = common
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    c ^= lsb
                    
                    c2 = common & comp[w]
                    c2 &= ~(1 << w)
                    
                    while c2:
                        lsb2 = c2 & -c2
                        z = lsb2.bit_length() - 1
                        c2 ^= lsb2
                        
                        c3 = common & comp[w] & comp[z]
                        c3 &= ~(1 << w)
                        c3 &= ~(1 << z)
                        
                        c4 = c3
                        while c4:
                            lsb3 = c4 & -c4
                            y = lsb3.bit_length() - 1
                            c4 ^= lsb3
                            if comp[y] & c3:
                                return False
        return True

    # Known 35-vertex graph
    def get_35_graph():
        vertices = list(combinations(range(7), 3))
        n = len(vertices)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                u = vertices[i]
                v = vertices[j]
                inter_size = len(set(u).intersection(v))
                if inter_size != 1:
                    edges.append([i, j])
        return n, edges

    n35, edges35 = get_35_graph()
    
    # Try to find a 36-vertex graph
    # Add vertex 35, try random connections
    for _ in range(10000):
        if time.time() > deadline:
            break
            
        mask = 0
        for i in range(35):
            if random.random() < 0.5:
                mask |= (1 << i)
                
        edges36 = edges35.copy()
        for i in range(35):
            if mask & (1 << i):
                edges36.append([35, i])
                
        if is_valid_fast(36, edges36):
            return {'n': 36, 'edges': edges36}
            
    # Fallback to 35
    if is_valid_fast(35, edges35):
        return {'n': 35, 'edges': edges35}
        
    # Fallback to 1
    return {'n': 1, 'edges': []}
