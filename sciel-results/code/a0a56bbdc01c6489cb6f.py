import time
import random

N = 12

def construct():
        # Fallback: Two disjoint K_{3,3}
        # Vertices 0-5 and 6-11
        # K_{3,3} on 0-2 (A) and 3-5 (B)
        edges = []
        for u in [0, 1, 2]:
            for v in [3, 4, 5]:
                edges.append([u, v])
        
        # K_{3,3} on 6-8 (A) and 9-11 (B)
        for u in [6, 7, 8]:
            for v in [9, 10, 11]:
                edges.append([u, v])
                
        return edges
