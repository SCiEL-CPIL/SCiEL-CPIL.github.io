# EVOLVE-BLOCK-START
def construct():
    n = 35
    edges = []
    
    # 6 disjoint copies of K4
    # Vertices 0..23
    for b in range(6):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
        
    # 1 component on vertices 24..34 (11 vertices)
    # This component is the complement of a triangle-free, C5-free graph.
    # Bar-G structure on local indices 0..10:
    # A = {0, 1, 2}, B = {3, 4, 5}, L = {6, 7, 8, 9, 10}
    # Edges in Bar-G:
    # 1. All (a, b) for a in A, b in B
    # 2. (0, l) for all l in L
    # 
    # Edges in G (complement):
    # 1. All pairs within A
    # 2. All pairs within B
    # 3. All pairs within L
    # 4. (a, l) for a in {1, 2}, l in L
    # 5. (b, l) for all b in B, l in L
    
    base = 24
    A = [base + 0, base + 1, base + 2]
    B = [base + 3, base + 4, base + 5]
    L = [base + 6, base + 7, base + 8, base + 9, base + 10]
    
    # Within A
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            edges.append([A[i], A[j]])
            
    # Within B
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            edges.append([B[i], B[j]])
            
    # Within L
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            edges.append([L[i], L[j]])
            
    # A-L edges (except 0-L)
    # A[0] is base+0, A[1] is base+1, A[2] is base+2
    for a in A[1:]:
        for l in L:
            edges.append([a, l])
            
    # B-L edges (all)
    for b in B:
        for l in L:
            edges.append([b, l])
            
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
