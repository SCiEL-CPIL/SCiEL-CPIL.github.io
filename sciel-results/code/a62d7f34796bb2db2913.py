import time
from itertools import combinations

def construct():
    """
    Construct a 3-uniform hypergraph on 9 vertices that is 3-chromatic.
    Uses the Fano plane (7 edges) on vertices 0-6, which is the smallest
    non-2-colorable 3-uniform hypergraph. Vertices 7 and 8 are isolated.
    Returns a dict with 'edges' (list of lists of ints) and 'colors' (list of ints).
    """
    # The Fano plane can be represented by non-zero vectors in F2^3.
    # We map the 7 non-zero 3-bit binary vectors to vertices 0..6.
    # v: int 0..6 -> binary representation with 3 bits (MSB to LSB corresponding to dimensions)
    # Actually, let's just use the standard mapping:
    # 0: 100
    # 1: 010
    # 2: 001
    # 3: 110
    # 4: 101
    # 5: 011
    # 6: 111
    
    vertices = list(range(7))
    
    # Define the Fano plane edges based on vector addition in F2
    # Edge if x ^ y ^ z == 0
    edges = []
    for combo in combinations(vertices, 3):
        x, y, z = combo
        # Check if they form a line in the Fano plane (sum to 0 in F2^3)
        # We need to map index 0..6 to vector.
        # 0->100, 1->010, 2->001, 3->110, 4->101, 5->011, 6->111
        # Let's create a helper or just hardcode the check logic using a precomputed table or bitwise logic
        # Mapping:
        # idx 0: 4 (100)
        # idx 1: 2 (010)
        # idx 2: 1 (001)
        # idx 3: 6 (110)
        # idx 4: 5 (101)
        # idx 5: 3 (011)
        # idx 6: 7 (111)
        
        def vec(i):
            if i == 0: return 4
            if i == 1: return 2
            if i == 2: return 1
            if i == 3: return 6
            if i == 4: return 5
            if i == 5: return 3
            if i == 6: return 7
            return 0
            
        if (vec(x) ^ vec(y) ^ vec(z)) == 0:
            edges.append([x, y, z])
            
    # The Fano plane has exactly 7 edges.
    # We add isolated vertices 7 and 8.
    # Coloring: 
    # We established a valid 3-coloring for indices 0..6 as [0, 1, 2, 1, 0, 1, 2].
    # Let's verify this specific coloring one last time.
    colors_fano = [0, 1, 2, 1, 0, 1, 2]
    
    # Assign color 0 to isolated vertices 7 and 8.
    colors = colors_fano + [0, 0]
    
    # Ensure types are standard Python types
    final_edges = []
    for e in edges:
        final_edges.append([int(x) for x in e])
        
    final_colors = [int(c) for c in colors]
    
    return {
        'edges': final_edges,
        'colors': final_colors
    }
