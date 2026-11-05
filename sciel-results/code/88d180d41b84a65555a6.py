"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 17
    colors = []
    # Partition differences 1..8 into 4 sets:
    # 0: {1, 2}
    # 1: {3, 4}
    # 2: {5, 6}
    # 3: {7, 8}
    # This ensures no C6 in any color class for K_17.
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            if d > 8:
                d = 17 - d
            if d <= 2:
                c = 0
            elif d <= 4:
                c = 1
            elif d <= 6:
                c = 2
            else:
                c = 3
            colors.append(c)
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
