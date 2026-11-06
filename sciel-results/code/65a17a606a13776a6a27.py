"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Known valid graph with n=32: 8 disjoint K4s
    # n=33 construction:
    # 8 disjoint K4s: C1, C2, ..., C8.
    # Vertices 0-31.
    # C1: 0,1,2,3
    # C2: 4,5,6,7
    # ...
    # C8: 28,29,30,31
    # Add vertex 32.
    # Connect 32 to all vertices in C1 (0,1,2,3).
    # Connect 32 to one vertex in each of C2..C8 (e.g., the first vertex: 4, 8, 12, 16, 20, 24, 28).
    
    # This graph is C5-free:
    # - Within any K4, no C5.
    # - Cycles involving vertex 32:
    #   - If the cycle stays within C1 + 32: 32 is connected to all 4 in C1. Any cycle is a triangle (32-u-v-32) since C1 is a clique. No C5.
    #   - If the cycle goes into C_i (i>=2): 32 is connected only to u_i (the first vertex of C_i).
    #     To enter C_i, must go through u_i. To leave, must go through u_i.
    #     So any path entering C_i and leaving is u_i - w - u_i? No, simple path.
    #     Path 32 - u_i - w ... to close cycle, w must connect to 32 or previous node.
    #     w does not connect to 32. w connects to u_i and other vertices in C_i.
    #     If w connects to another vertex x in C_i, and x connects to ... eventually back to 32?
    #     Only u_i connects to 32. So the cycle must return to u_i.
    #     Path 32 - u_i - w - x - ... - u_i - 32.
    #     Since C_i is a K4, the longest simple path inside C_i starting from u_i is length 3 (u_i-a-b-c).
    #     So max cycle length involving C_i is 32-u_i-a-b-c-u_i-32? No, u_i is repeated.
    #     Simple cycle: 32-u_i-a-b-...-32.
    #     The only way to get back to 32 is via u_i.
    #     So the cycle is 32-u_i-(path in C_i not using u_i)-u_i-32.
    #     Since C_i is K4, the vertices other than u_i form a K3.
    #     Max simple path in K3 is length 2 (a-b-c).
    #     So cycle is 32-u_i-a-b-c-u_i-32? No, u_i is start and end.
    #     Cycle: 32-u_i-a-b-u_i-32. Length 5?
    #     Vertices: 32, u_i, a, b, u_i. Not simple (u_i repeated).
    #     Simple cycle: 32-u_i-a-b-32? b not connected to 32.
    #     So no C5.
    
    # Independence number:
    # If we pick 32, we cannot pick any vertex from C1.
    # We can pick at most 1 vertex from each C2..C8, avoiding the connected ones.
    # So max independent set containing 32 is 1 (for 32) + 0 (from C1) + 7 (from C2..C8) = 8.
    # If we do not pick 32, we can pick at most 1 from each C1..C8.
    # Max independent set not containing 32 is 8.
    # So alpha = 8 < 9.
    
    edges = []
    
    # Add edges for 8 disjoint K4s
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    # Add edges for vertex 32
    # Connect to all in C1 (0, 1, 2, 3)
    for v in range(4):
        edges.append([32, v])
        
    # Connect to one vertex in C2..C8 (4, 8, 12, 16, 20, 24, 28)
    for b in range(1, 8):
        v = 4 * b
        edges.append([32, v])
        
    return {'n': 33, 'edges': edges}

# EVOLVE-BLOCK-END
