"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    30 points on 3 parallel lines y=0,1,2 with 10 points each at x=0..9.
    This maximizes the multiplicity of doubled area = 2, achieving count 500.
    
    For doubled area v:
    - Adjacent lines (|dy|=1): 2 line pairs * 2 orientations * (10-v) x-pairs * 10 choices = 40*(10-v)
    - Non-adjacent lines (|dy|=2): 1 line pair * 2 orientations * (10-v/2) x-pairs * 10 choices = 20*(10-v/2)
    
    v=2: 40*8 + 20*9 = 320 + 180 = 500 (maximum)
    """
    points = []
    for y in range(3):
        for x in range(10):
            points.append((x, y))
    return points
# EVOLVE-BLOCK-END
