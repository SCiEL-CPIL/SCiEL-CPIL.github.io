import random

def construct():
    n = 128
    
    def can_place(colors, x, color):
        for d in range(1, x//3 + 1):
            idx1 = x - 3*d
            if idx1 < 0:
                break
            c1 = colors[idx1]
            c2 = colors[idx1 + d]
            c3 = colors[idx1 + 2*d]
            if len({c1, c2, c3, color}) < 3:
                return False
        return True

    def solve(palette, seed):
        random.seed(seed)
        colors = []
        
        def backtrack(pos):
            if pos == n:
                return True
            
            color_order = palette[:]
            random.shuffle(color_order)
            
            for color in color_order:
                if can_place(colors, pos, color):
                    colors.append(color)
                    if backtrack(pos + 1):
                        return True
                    colors.pop()
            return False
        
        if backtrack(0):
            return colors
        return None

    # Try 3 colors with multiple seeds
    palette_3 = [0, 1, 2]
    for seed in range(20):
        result = solve(palette_3, seed)
        if result is not None:
            return result

    # Try 4 colors with multiple seeds
    palette_4 = [0, 1, 2, 3]
    for seed in range(20):
        result = solve(palette_4, seed)
        if result is not None:
            return result

    # Fallback: Greedy with 4 colors
    colors = []
    palette = [0, 1, 2, 3]
    for x in range(n):
        placed = False
        for color in palette:
            if can_place(colors, x, color):
                colors.append(color)
                placed = True
                break
        if not placed:
            colors.append(0)
    return colors
