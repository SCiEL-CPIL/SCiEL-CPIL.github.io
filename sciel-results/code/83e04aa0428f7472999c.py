import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    edges = []
    
    # Construct the graph K_20 minus a triangle on vertices 0, 1, 2
    # This graph is K_3,3-free because the only independent set of size 3 is {0, 1, 2}
    # and we cannot find a disjoint independent set of size 3.
    
    missing_edges = {(0, 1), (1, 2), (0, 2)}
    
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in missing_edges:
                edges.append([i, j])
                
    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
