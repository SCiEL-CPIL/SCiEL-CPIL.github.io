import random
import time

def construct():
    n = 128
    
    def can_place(colors, x, color):
        for d in range(1, x // 3 + 1):
            if x - 3 * d < 0:
                break
            c1 = colors[x - 3 * d]
            c2 = colors[x - 2 * d]
            c3 = colors[x - d]
            s = {c1, c2, c3, color}
            if len(s) < 3:
                return False
        return True

    def solve(colors, pos, palette, max_nodes):
        nonlocal nodes_visited
        if pos == n:
            return colors
        
        nodes_visited += 1
        if nodes_visited > max_nodes:
            return None

        color_order = palette[:]
        random.shuffle(color_order)
        
        for color in color_order:
            if can_place(colors, pos, color):
                colors.append(color)
                result = solve(colors, pos + 1, palette, max_nodes)
                if result is not None:
                    return result
                colors.pop()
        
        return None

    # Try 3 colors
    nodes_visited = 0
    random.seed(42)
    result = solve([], 0, [0, 1, 2], max_nodes=50000)
    
    if result is not None:
        return result

    # Try 4 colors
    nodes_visited = 0
    random.seed(43)
    result = solve([], 0, [0, 1, 2, 3], max_nodes=50000)
    
    if result is not None:
        return result

    # Fallback: Greedy 4-color
    colors = []
    for x in range(n):
        placed = False
        for color in range(4):
            if can_place(colors, x, color):
                colors.append(color)
                placed = True
                break
        if not placed:
            colors.append(0)
    return colors
