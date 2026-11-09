import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # Try to return the known n=26 Lazebnik-Woldar construction
    # Vertices labeled 0..25
    # Use algebraic construction: vertices as (a,b) in Z5xZ5 plus infinity
    
    n = 26
    colors = []
    
    # Map vertex index to (a,b) or infinity
    def v2coord(v):
        if v == 25:
            return None  # infinity
        return (v // 5, v % 5)
    
    def coord2v(a, b):
        return a * 5 + b
    
    # For two affine vertices (a,b) and (a',b'):
    # If a != a': color = (b - b') * (a - a')^{-1} mod 5
    # If a == a': color = (b - b') mod 5  (but need to avoid C4)
    # 
    # Actually, let's use: color based on the "direction" in the affine plane
    # Directions in AG(2,5): 5 slopes + 1 vertical = 6 directions, but we only have 5 colors
    #
    # Alternative: use the construction from the paper directly.
    
    # Let me use a different known construction:
    # Color of edge (i,j) for i,j in 0..24 (affine part):
    # Let a=i//5, b=i%5, a'=j//5, b'=j%5
    # If a==a': color = (b'-b) mod 5
    # Else: color = ((b'-b) * pow(a'-a, 4, 5)) mod 5  # since inv(x) = x^4 mod 5
    
    for i in range(n):
        for j in range(i+1, n):
            ci, cj = v2coord(i), v2coord(j)
            if ci is None or cj is None:
                # One or both is infinity
                if ci is None and cj is None:
                    colors.append(0)  # shouldn't happen
                else:
                    # Edge from infinity to affine vertex
                    v = cj if ci is None else ci
                    a, b = v
                    colors.append((a + b) % 5)
            else:
                a, b = ci
                a2, b2 = cj
                if a == a2:
                    colors.append((b2 - b) % 5)
                else:
                    inv = pow((a2 - a) % 5, 4, 5)
                    colors.append((b2 - b) * inv % 5)
    
    # Verify no C4 in any color class
    # For each color c, check that no two vertices share two common neighbors in color c
    valid = True
    for c in range(5):
        # Build adjacency for color c
        adj = [set() for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if colors[idx] == c:
                    adj[i].add(j)
                    adj[j].add(i)
                idx += 1
        # Check for C4: for each pair of vertices, count common neighbors
        for i in range(n):
            for j in range(i+1, n):
                common = adj[i] & adj[j]
                if len(common) >= 2:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {'n': n, 'colors': colors}
    
    # Fallback: cyclic n=11 construction
    n = 11
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
