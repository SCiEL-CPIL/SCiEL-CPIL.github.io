"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # We aim to construct a 4-coloring of K_17 with no monochromatic C6.
    # This certifies R_4(C6) > 17.
    # The construction uses the cosets of the subgroup of order 4 in the multiplicative group of Z_17.
    # The multiplicative group Z_17* is cyclic of order 16.
    # A generator is 3.
    # The subgroup of order 4 is H = <3^4> = <13> = {1, 13, 13^2=16, 13^3=4}.
    # The cosets are:
    # H0 = {1, 4, 13, 16}
    # H1 = 3*H0 = {3, 12, 11, 5}
    # H2 = 9*H0 = {9, 7, 2, 14}
    # H3 = 10*H0 = {10, 15, 6, 8}
    
    n = 17
    
    # Map each non-zero difference d in Z_17 to a color 0, 1, 2, or 3
    color_map = {}
    
    set_0 = {1, 4, 13, 16}
    set_1 = {3, 5, 11, 12}
    set_2 = {2, 7, 9, 14}
    set_3 = {6, 8, 10, 15}
    
    for d in set_0:
        color_map[d] = 0
    for d in set_1:
        color_map[d] = 1
    for d in set_2:
        color_map[d] = 2
    for d in set_3:
        color_map[d] = 3
        
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            # d is in 1..16
            colors.append(color_map[d])
            
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
