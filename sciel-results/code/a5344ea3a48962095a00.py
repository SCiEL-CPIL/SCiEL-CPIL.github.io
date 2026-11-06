"""Design a D4-symmetric 32x32 photonic crystal unit cell maximizing TM band-4/5 gap."""
import math

# EVOLVE-BLOCK-START
def design(problem):
    size = problem["pixels"]
    n = size
    
    # Start with dielectric background
    cell = [[1] * n for _ in range(n)]
    
    def coords(i, j):
        x = (j + 0.5) / n - 0.5
        y = (i + 0.5) / n - 0.5
        return x, y
    
    # Air holes in a D4-symmetric "diamond grid" pattern.
    # Central large hole + 4 holes at edge-midpoints + 4 at diagonal positions.
    # This creates a dielectric lattice with strong periodic modulation.
    
    holes = [
        # (cx, cy, radius)
        (0.0, 0.0, 0.16),        # center
        (0.0, 0.32, 0.11),       # top
        (0.0, -0.32, 0.11),      # bottom
        (0.32, 0.0, 0.11),       # right
        (-0.32, 0.0, 0.11),      # left
        (0.32, 0.32, 0.07),      # top-right
        (0.32, -0.32, 0.07),     # bottom-right
        (-0.32, 0.32, 0.07),     # top-left
        (-0.32, -0.32, 0.07),    # bottom-left
    ]
    
    for i in range(n):
        for j in range(n):
            x, y = coords(i, j)
            for (cx, cy, r) in holes:
                if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                    cell[i][j] = 0
                    break
    
    # Enforce D4 symmetry via orbit majority
    def d4_images(i, j):
        return [
            (i, j),
            (j, n - 1 - i),
            (n - 1 - i, n - 1 - j),
            (n - 1 - j, i),
            (i, n - 1 - j),
            (n - 1 - i, j),
            (n - 1 - j, n - 1 - i),
            (j, i),
        ]
    
    visited = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if visited[i][j]:
                continue
            imgs = d4_images(i, j)
            vals = [cell[pi][pj] for pi, pj in imgs]
            val = 1 if sum(vals) >= len(vals) / 2 else 0
            for pi, pj in imgs:
                cell[pi][pj] = val
                visited[pi][pj] = True
    
    return cell
# EVOLVE-BLOCK-END
