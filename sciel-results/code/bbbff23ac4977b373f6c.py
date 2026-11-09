import time
import random

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # The theoretical minimum for distinct cycle lengths in a graph with min degree 3 and girth >= 4 is 2.
    # Achievable by K_{3,3} (cycles 4, 6).
    # Two disjoint K_{3,3} components use 12 vertices and satisfy all constraints.
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0-5
    # Partition A: 0, 1, 2
    # Partition B: 3, 4, 5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6-11
    # Partition A: 6, 7, 8
    # Partition B: 9, 10, 11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    return edges
