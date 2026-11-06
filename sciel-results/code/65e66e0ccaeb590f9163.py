def _greedy_fallback():
    colors = []
    for x in range(1, 129):
        # Assign a new unique color to ensure validity? No, that's 128 colors.
        # We can just use x-1.
        colors.append(x - 1)
    return colors
