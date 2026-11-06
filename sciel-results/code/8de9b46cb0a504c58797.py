import time
import random
import sys
from itertools import combinations

def construct():
    # The Schläfli graph on 35 vertices
    # Vertices are the 3-subsets of {0, 1, 2, 3, 4, 5, 6}
    # Two vertices are adjacent if their intersection has size 0 or 2
    
    vertices = list(combinations(range(7), 3))
    n = len(vertices)
    
    # Map vertex to index
    index_map = {v: i for i, v in enumerate(vertices)}
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(set(vertices[i]).intersection(vertices[j]))
            if inter == 0 or inter == 2:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
