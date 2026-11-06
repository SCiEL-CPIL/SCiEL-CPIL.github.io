import time

N = 16

# EVOLVE-BLOCK-START
def construct():
    # Base case for K4
    # Vertices 0,1,2,3
    # Edges (i,j) with i<j
    # (0,1): 1
    # (0,2): 4
    # (0,3): 1
    # (1,2): 2
    # (1,3): 4
    # (2,3): 3
    # Colors used: {1, 2, 3, 4}
    
    def get_base_coloring():
        # Returns a dictionary mapping (local_i, local_j) to color
        # and the max color value used
        base = {
            (0, 1): 1,
            (0, 2): 4,
            (0, 3): 1,
            (1, 2): 2,
            (1, 3): 4,
            (2, 3): 3
        }
        return base, 4

    def build_coloring(n):
        # Returns a dictionary mapping (local_i, local_j) to color
        # and the max color value used
        if n == 4:
            return get_base_coloring()
        
        n1 = n // 2
        n2 = n - n1
        
        # Recursively build for first half and second half
        # We need to handle n=1,2,3 as well?
        # The problem is for N=16, so we will hit 8, 4.
        # But what if n is not a power of 2?
        # For n=16, splits are 8, 8. Then 4, 4. So we only hit 4.
        # If we had n=10, 5, 5. 5 splits to 2, 3.
        # We need base cases for n=1,2,3.
        
        if n == 1:
            return {}, 0
        if n == 2:
            return {(0, 1): 1}, 1
        if n == 3:
            # K3. Any 3 edges. No C4. Can use 3 colors.
            return {(0, 1): 1, (0, 2): 2, (1, 2): 3}, 3
            
        col1, max1 = build_coloring(n1)
        col2, max2 = build_coloring(n2)
        
        # Shift colors of col2 to be above max1
        shifted_col2 = {}
        for (i, j), c in col2.items():
            shifted_col2[(i, j)] = c + max1
            
        bridge_color = max1 + max2 + 1
        
        new_col = {}
        
        # Edges in first half
        for (i, j), c in col1.items():
            new_col[(i, j)] = c
            
        # Edges in second half (shift indices)
        for (i, j), c in shifted_col2.items():
            new_col[(i + n1, j + n1)] = c
            
        # Edges between halves
        for i in range(n1):
            for j in range(n1, n):
                new_col[(i, j)] = bridge_color
                
        return new_col, bridge_color

    # Build for N=16
    coloring_dict, max_color = build_coloring(N)
    
    # Create the final array
    # Lexicographically ordered pairs (i,j), 0<=i<j<N
    result = [0] * (N * (N - 1) // 2)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            result[idx] = coloring_dict[(i, j)]
            idx += 1
            
    return result
# EVOLVE-BLOCK-END
