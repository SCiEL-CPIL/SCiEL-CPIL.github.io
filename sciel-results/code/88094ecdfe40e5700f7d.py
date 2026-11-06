import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 150  # seconds
    
    # Vertices 0..14 form M(C_7)
    # Vertices:
    # 0..6 : C_7
    # 7..13 : C_7' (independent set, no edges between them)
    # 14 : w (apex)
    
    edges = []
    
    # 1. Edges of C_7 in V (0..6)
    for i in range(7):
        edges.append([i, (i + 1) % 7])
        
    # 2. Cross edges
    # For each edge u-v in C_7, add u-v' and v-u'
    # v' is 7+v
    for i in range(7):
        u = i
        v = (i + 1) % 7
        u_prime = 7 + u
        v_prime = 7 + v
        edges.append([u, v_prime])
        edges.append([v, u_prime])
        
    # 3. Apex edges
    # w (14) connected to all u' (7..13)
    for i in range(7):
        edges.append([14, 7 + i])
        
    # Current edges: 7 + 14 + 7 = 28
    # Vertex 15 is new.
    # Connect 15 to an independent set of M(C_7).
    # Max independent set size is 6.
    # One such set: S_V = {0, 2, 4}, S_V' = {0', 2', 4'}
    # Indices: 0, 2, 4, 7, 9, 11
    
    independent_set = [0, 2, 4, 7, 9, 11]
    
    for v in independent_set:
        edges.append([15, v])
        
    return edges

if __name__ == "__main__":
    print(construct())
