"""
Dense 20-vertex 4-cycle rainbow edge-coloured graph.

Approach:
- 4-partite graph with parts of size 5: A, B, C, D.
- Include matchings between all 6 pairs of parts (complete 4-partite with matchings).
- Total edges: 6 * 5 = 30.
- Coloring strategy:
  For a 4-cycle in a 4-partite graph, it visits 4 distinct parts.
  The 4 edges belong to 4 distinct part-pairs.
  We need the 4 colors to be distinct.
  
  We have 20 colors and 6 part-pairs.
  Assign colors as follows:
    AB: 0-4
    BC: 5-9
    CD: 10-14
    DA: 15-19
    AC: 0-4 (reuse)
    BD: 5-9 (reuse)
  
  Now check all possible 4-cycles. A 4-cycle in a 4-partite graph uses 4 edges from 4 distinct part-pairs.
  The possible sets of 4 part-pairs that form a cycle are:
    1. AB, BC, CD, DA -> colors 0-4, 5-9, 10-14, 15-19 (all distinct)
    2. AB, BD, DC, CA -> AB(0-4), BD(5-9), DC(10-14), CA(0-4). 
       Wait, CA is same as AC which is 0-4. AB is also 0-4.
       So we could have AB color i and AC color j where i==j.
       This would violate the rainbow condition if a 4-cycle uses an AB edge and an AC edge with the same color.
       
  Let's re-examine. In a 4-partite graph with parts A, B, C, D, a 4-cycle must use 4 edges connecting 4 distinct parts.
  The edges must form a cycle. The possible cycles of parts are:
    A-B-C-D-A (uses AB, BC, CD, DA)
    A-B-D-C-A (uses AB, BD, DC, CA)
    A-C-B-D-A (uses AC, CB, BD, DA)
    
  So we have 3 types of 4-cycles (up to rotation/reversal).
  
  Type 1: AB, BC, CD, DA
    Colors: AB(0-4), BC(5-9), CD(10-14), DA(15-19). All disjoint. OK.
  
  Type 2: AB, BD, DC, CA
    Colors: AB(0-4), BD(5-9), DC(10-14), CA(0-4).
    AB and CA both use colors 0-4. We need to ensure that for any specific 4-cycle, the AB edge color != CA edge color.
    The AB edge is between A[i] and B[i] (color i).
    The CA edge is between C[j] and A[j] (color j).
    The 4-cycle is A[i]-B[i]-D[k]-C[j]-A[i]? No, the cycle is A-B-D-C-A.
    Vertices: A[i], B[i], D[k], C[j].
    Edges:
      A[i]-B[i] (AB, color i)
      B[i]-D[k] (BD, color ?)
      D[k]-C[j] (CD/DC, color ?)
      C[j]-A[i] (CA, color ?)
    
    Wait, my matching definitions:
    AB: A[i]-B[i] color i
    BC: B[i]-C[i] color 5+i
    CD: C[i]-D[i] color 10+i
    DA: D[i]-A[i] color 15+i
    AC: A[i]-C[i] color i
    BD: B[i]-D[i] color 5+i
    
    Let's trace a Type 2 cycle: A[i] - B[i] - D[k] - C[j] - A[i].
    Edge 1: A[i]-B[i]. Color = i.
    Edge 2: B[i]-D[k]. This is a BD edge. In my matching BD: B[m]-D[m].
    So for B[i] to connect to D[k], we need i=k. So B[i]-D[i]. Color = 5+i.
    Edge 3: D[i]-C[j]. This is a CD edge. In my matching CD: C[m]-D[m].
    So for D[i] to connect to C[j], we need i=j. So D[i]-C[i]. Color = 10+i.
    Edge 4: C[i]-A[i]. This is an AC edge. In my matching AC: A[m]-C[m].
    So C[i]-A[i]. Color = i.
    
    The colors are: i, 5+i, 10+i, i.
    The first and last colors are both i. Collision!
    
    So this coloring fails for Type 2 cycles.
    
    We need a different coloring for AC and BD.
    
    Let's try:
    AB: 0-4
    BC: 5-9
    CD: 10-14
    DA: 15-19
    AC: 1-5 (shifted)
    BD: 6-10 (shifted)
    
    Type 2: A[i]-B[i]-D[i]-C[i]-A[i]
    AB: i
    BD: 6+i
    CD: 10+i
    CA: 1+i
    Colors: i, 6+i, 10+i, 1+i.
    Distinct?
    i vs 1+i: distinct.
    i vs 6+i: distinct.
    i vs 10+i: distinct.
    1+i vs 6+i: distinct.
    1+i vs 10+i: distinct.
    6+i vs 10+i: distinct.
    Yes, all distinct.
    
    Type 3: A[i]-C[i]-B[j]-D[k]-A[i].
    Wait, let's trace carefully.
    A[i]-C[i] (AC, color 1+i)
    C[i]-B[j] (BC, color 5+j? No, BC is B[m]-C[m]. So C[i]-B[i]. Color 5+i.)
    B[i]-D[k] (BD, color 6+k? BD is B[m]-D[m]. So B[i]-D[i]. Color 6+i.)
    D[i]-A[i] (DA, color 15+i)
    
    Colors: 1+i, 5+i, 6+i, 15+i.
    Distinct?
    1+i vs 5+i: yes.
    1+i vs 6+i: yes.
    1+i vs 15+i: yes.
    5+i vs 6+i: yes.
    5+i vs 15+i: yes.
    6+i vs 15+i: yes.
    Yes, all distinct.
    
    So this coloring works!
    
    Total edges: 30.
"""

def construct():
    n = 20
    parts = [[i for i in range(5*j, 5*(j+1))] for j in range(4)]
    A, B, C, D = parts
    
    edges = []
    
    # AB: A[i]-B[i] color i
    for i in range(5):
        edges.append([A[i], B[i], i])
        
    # BC: B[i]-C[i] color 5+i
    for i in range(5):
        edges.append([B[i], C[i], 5+i])
        
    # CD: C[i]-D[i] color 10+i
    for i in range(5):
        edges.append([C[i], D[i], 10+i])
        
    # DA: D[i]-A[i] color 15+i
    for i in range(5):
        edges.append([D[i], A[i], 15+i])
        
    # AC: A[i]-C[i] color 1+i
    for i in range(5):
        edges.append([A[i], C[i], 1+i])
        
    # BD: B[i]-D[i] color 6+i
    for i in range(5):
        edges.append([B[i], D[i], 6+i])
        
    return edges
