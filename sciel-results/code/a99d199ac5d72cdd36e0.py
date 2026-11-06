import time

def construct():
    # Grötzsch graph edges
    # Vertices 0-4: Cycle
    # Vertices 5-9: Copies
    # Vertex 10: Apex
    edges = []
    
    # Cycle edges
    for i in range(5):
        edges.append([i, (i + 1) % 5])
        
    # Copy edges
    for i in range(5):
        edges.append([i, 5 + i])
        edges.append([i, 5 + (i - 1) % 5])
        
    # Apex edges
    for i in range(5):
        edges.append([10, 5 + i])
        
    # Add 5 new vertices 11-15
    # Partition A: 11, 12
    # Partition B: 13, 14, 15
    
    # Independent sets in Grötzsch
    S_A = [5, 6, 7, 8, 9]
    S_B = [10, 0, 2]
    
    # Connect A to S_A
    for v in [11, 12]:
        for u in S_A:
            edges.append([v, u])
            
    # Connect B to S_B
    for v in [13, 14, 15]:
        for u in S_B:
            edges.append([v, u])
            
    # Connect A and B (K_2,3)
    for v_a in [11, 12]:
        for v_b in [13, 14, 15]:
            edges.append([v_a, v_b])
            
    return edges
