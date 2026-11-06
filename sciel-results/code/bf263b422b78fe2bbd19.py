import random

def construct():
    n = 128
    
    def can_place(colors, x, color):
        # Check if placing 'color' at position x (0-indexed) is valid
        # for all 4-term APs ending at x
        for d in range(1, x//3 + 1):
            if x - 3*d < 0:
                break
            c1 = colors[x-3*d]
            c2 = colors[x-2*d]
            c3 = colors[x-d]
            S = {c1, c2, c3}
            ls = len(S)
            if ls == 1:
                return False
            if ls == 2:
                if color in S:
                    return False
        return True

    def solve_backtrack(k_colors, max_nodes=100000):
        colors = [0] * n
        nodes_visited = [0]
        
        def dfs(pos):
            if nodes_visited[0] > max_nodes:
                return False
            if pos == n:
                return True
            
            # Randomize color order
            palette = list(range(k_colors))
            random.shuffle(palette)
            
            for c in palette:
                nodes_visited[0] += 1
                if can_place(colors, pos, c):
                    colors[pos] = c
                    if dfs(pos + 1):
                        return True
            return False
        
        return dfs(0), colors

    # Try 3 colors with several restarts
    random.seed(42)
    for _ in range(20):
        success, colors = solve_backtrack(3, max_nodes=50000)
        if success:
            return colors

    # If 3 colors fail, try 4 colors
    for _ in range(5):
        success, colors = solve_backtrack(4, max_nodes=50000)
        if success:
            return colors

    # Fallback: deterministic greedy with 4 colors
    colors = [0] * n
    for i in range(n):
        for c in range(4):
            if can_place(colors, i, c):
                colors[i] = c
                break
    return colors
