import time
import itertools
from collections import defaultdict

def construct():
    """
    Returns a list of edges for a graph on 12 vertices (0..11)
    with minimum degree 3, girth >= 4, and minimum distinct cycle lengths.
    
    Strategy: Use two disjoint K_{3,3} components.
    K_{3,3} is 3-regular, bipartite (girth 4), and has cycle lengths {4, 6}.
    Distinct cycle lengths = 2.
    This is likely the minimum possible for 12 vertices with min degree 3.
    """
    edges = []
    
    # Component 1: K_{3,3} on vertices 0..5
    # Partition A: 0, 1, 2
    # Partition B: 3, 4, 5
    for i in range(3):
        for j in range(3):
            edges.append([i, 3 + j])
            
    # Component 2: K_{3,3} on vertices 6..11
    # Partition A: 6, 7, 8
    # Partition B: 9, 10, 11
    for i in range(3):
        for j in range(3):
            edges.append([6 + i, 9 + j])
            
    # Ensure types are strictly Python int
    return [[int(u), int(v)] for u, v in edges]

if __name__ == "__main__":
    result = construct()
    print(result)
