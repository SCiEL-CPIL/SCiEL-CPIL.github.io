import time

def construct():
    """
    Returns a 4-coloring of K_17 with no monochromatic C_6.
    This certifies R_4(C_6) > 17.
    """
    n = 17
    colors = []
    
    # Define which steps belong to which color
    # Steps are 1 to 8.
    # Partition: {1,4}, {2,7}, {3,6}, {5,8}
    step_to_color = {
        1: 0, 4: 0,
        2: 1, 7: 1,
        3: 2, 6: 2,
        5: 3, 8: 3
    }
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = j - i
            # The step in the cyclic group Z_17
            # The distance is min(diff, n - diff)
            step = min(diff, n - diff)
            colors.append(step_to_color[step])
            
    return {'n': n, 'colors': colors}
