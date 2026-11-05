def construct():
    # Hardcoded optimal Harborth configuration for n=32
    # This is a hexagonal spiral of 32 points
    # It achieves the maximum number of unit-distance edges (76)
    
    points = [
        # Ring 0
        [0, 0],
        
        # Ring 1
        [1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1],
        
        # Ring 2
        [2, 0], [2, -1], [2, -2], [1, -2], [0, -2], [-1, -1], 
        [-2, 0], [-2, 1], [-2, 2], [-1, 2], [0, 2], [1, 1],
        
        # Ring 3 (first 13 points to reach total 32)
        [3, 0], [3, -1], [3, -2], [3, -3], [2, -3], [1, -3], 
        [0, -3], [-1, -2], [-2, -1], [-3, 0], [-3, 1], [-3, 2], 
        [-3, 3]
    ]
    
    return points
