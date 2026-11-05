"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    # Create a list of 120 edges, initialized to 0 (background color)
    colors = [0] * (N * (N - 1) // 2)
    
    # Define 5 disjoint triangles covering vertices 0-14. Vertex 15 is isolated.
    # T1: 0, 1, 2
    # T2: 3, 4, 5
    # T3: 6, 7, 8
    # T4: 9, 10, 11
    # T5: 12, 13, 14
    
    triangles = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12, 13, 14]
    ]
    
    current_color = 1
    
    # Helper to get index of pair (i, j) in lexicographic order
    def get_idx(i, j):
        if i > j:
            i, j = j, i
        # Number of pairs with first element < i
        # Pairs are (0,1)..(0,15), (1,2)..(1,15), ...
        # Count = i*(N-1) - i*(i-1)/2 ? No.
        # Standard formula for index of (i,j) in upper triangular matrix:
        # idx = i*N - i*(i+1)//2 + (j - i - 1)
        return i * N - i * (i + 1) // 2 + (j - i - 1)
        
    for tri in triangles:
        v1, v2, v3 = tri
        # Assign unique colors to the 3 edges of the triangle
        idx1 = get_idx(v1, v2)
        idx2 = get_idx(v2, v3)
        idx3 = get_idx(v1, v3)
        
        colors[idx1] = current_color
        current_color += 1
        colors[idx2] = current_color
        current_color += 1
        colors[idx3] = current_color
        current_color += 1
        
    return colors
# EVOLVE-BLOCK-END
