def construct():
    n = 17
    colors = []
    # Distances 1-8 map to colors 0-3
    # dist 1,2 -> 0
    # dist 3,4 -> 1
    # dist 5,6 -> 2
    # dist 7,8 -> 3
    for i in range(n):
        for j in range(i+1, n):
            d = j - i
            # Normalize distance to 1..8
            if d > n // 2:
                d = n - d
            # d is now in 1..8
            color = (d - 1) // 2
            colors.append(color)
    return {'n': n, 'colors': colors}
