"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    edges = set()
    
    # Clique A: 0..19
    for i in range(20):
        for j in range(i+1, 20):
            edges.add((i, j))
            
    # Clique B: 20..39
    for i in range(20):
        for j in range(i+1, 20):
            edges.add((i+20, j+20))
            
    # Matching
    for i in range(20):
        edges.add((i, i+20))
        
    edges_list = sorted(edges)
    
    cycles = []
    for k in range(3, N+1):
        if k <= 20:
            # Cycle in A
            cycles.append(list(range(k)))
        elif k == 21:
            # k_A=11, k_B=10? Or k_A=10, k_B=11?
            # Let's use the odd construction: k_A = (k+1)/2 = 11, k_B = 10.
            # S_A = 0..10, S_B = 21..30.
            # Cycle: 0..10 -> 30..21 -> 1 -> 0
            # Wait, 1 is in S_A.
            # Path A: 0->1->...->10.
            # Match 10->30.
            # Path B: 30->29->...->21.
            # Match 21->1.
            # Close 1->0.
            # Vertices: 0..10, 21..30. Distinct.
            cyc = list(range(11)) + list(range(30, 20, -1))
            # Check: 0,1,2,3,4,5,6,7,8,9,10, 30,29,28,27,26,25,24,23,22,21
            # Edges: (10,30) Match. (21,1) Match. (1,0) A.
            cycles.append(cyc)
        elif k % 2 == 0:
            # Even k
            half = k // 2
            # S_A = 0..half-1, S_B = 20..20+half-1
            # Cycle: 0..half-1 -> 20+half-1..20 -> 0
            cyc = list(range(half)) + list(range(20+half-1, 19, -1))
            cycles.append(cyc)
        else:
            # Odd k > 21
            k_A = (k + 1) // 2
            k_B = (k - 1) // 2
            # S_A = 0..k_A-1, S_B = 21..20+k_B
            # Cycle: 0..k_A-1 -> 20+k_B..21 -> 1 -> 0
            cyc = list(range(k_A)) + list(range(20+k_B, 20, -1))
            cycles.append(cyc)
            
    return {'edges': edges_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
