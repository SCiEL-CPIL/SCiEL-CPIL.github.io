import time

def construct():
    n = 10
    squares = []
    
    # Generate all non-zero pairs (u, v) in Z5 x Z2
    pairs = []
    for u in range(5):
        for v in range(2):
            if u != 0 or v != 0:
                pairs.append((u, v))
                
    # Sort for determinism (optional)
    # pairs will be (1,0), (2,0)... (4,0), (0,1), (1,1)...
    
    for u, v in pairs:
        L = [[0]*n for _ in range(n)]
        for i in range(n):
            i1, i2 = i % 5, i // 5
            for j in range(n):
                j1, j2 = j % 5, j // 5
                
                # Z5 part
                part5 = (i1 + j1 + u * i2 * j2) % 5
                # Z2 part
                part2 = (i2 + j2 + v * i1 * j1) % 2
                
                L[i][j] = part5 * 5 + part2
        squares.append(L)
        
    return squares
