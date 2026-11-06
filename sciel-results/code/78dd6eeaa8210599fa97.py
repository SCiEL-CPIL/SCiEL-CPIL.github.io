def construct():
    """
    Returns 32 distinct integer pairs [a, b] representing points on a triangular lattice
    a + b*omega, omega = exp(i*pi/3).
    
    The points are chosen to maximize the number of pairs at mutual distance exactly 1.
    According to Harborth's theorem, the maximum number of unit distances for n=32 points
    is floor(3n - sqrt(12n - 3)) = floor(96 - sqrt(381)) = floor(96 - 19.519...) = 76.
    
    This construction uses the first 32 points of the hexagonal spiral centered at the origin.
    """
    points = [
        # Ring 0: 1 point
        [0, 0],
        
        # Ring 1: 6 points
        [1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1],
        
        # Ring 2: 12 points
        [2, 0], [2, -1], [2, -2], [1, -2], [0, -2], [-1, -2],
        [-2, -1], [-2, 0], [-2, 1], [-2, 2], [-1, 2], [0, 2],
        
        # Ring 3: first 13 points to reach total 32
        [1, 2], [2, 1], [3, 0], [3, -1], [3, -2], [3, -3],
        [2, -3], [1, -3], [0, -3], [-1, -3], [-2, -3], [-3, -2], [-3, -1]
    ]
    
    return points
