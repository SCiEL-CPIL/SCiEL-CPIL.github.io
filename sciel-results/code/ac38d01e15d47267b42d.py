import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    n = 16
    
    # Try cyclic constructions (i + j) % M for M in range(13, 18)
    # We want to minimize M, so check from low to high.
    # However, we must ensure the coloring is valid.
    
    def is_valid_coloring(colors, n):
        # colors is a list of length n*(n-1)/2
        # Map (i, j) to index
        # idx = i * (2*n - i - 1) // 2 + (j - i - 1)
        # Or just use a 2D array for speed
        
        c = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c[i][j] = colors[idx]
                c[j][i] = colors[idx]
                idx += 1
        
        # Check all K4s
        # 16 choose 4 = 1820
        # For each K4, check distinct colors
        for quad in itertools.combinations(range(n), 4):
            v0, v1, v2, v3 = quad
            e01 = c[v0][v1]
            e02 = c[v0][v2]
            e03 = c[v0][v3]
            e12 = c[v1][v2]
            e13 = c[v1][v3]
            e23 = c[v2][v3]
            
            distinct = {e01, e02, e03, e12, e13, e23}
            if len(distinct) < 5:
                return False
        return True

    # Test M from 13 to 17
    # Note: If M < 16, the colors are 0..M-1.
    # The problem asks for colors in 0..119, so any M <= 120 is fine.
    
    for M in range(13, 18):
        if time.time() > deadline:
            break
        colors = [(i + j) % M for i in range(n) for j in range(i + 1, n)]
        if is_valid_coloring(colors, n):
            return colors

    # Fallback to seed (M=17) if none of the above worked (unlikely for M=17)
    # But M=17 is the seed, so it should work.
    # If M=13,14,15,16 failed, M=17 is the default.
    # We can just return the M=17 version explicitly to be safe if the loop didn't hit it or timed out.
    # Actually, the loop includes 17.
    
    # If we reach here, it means either time ran out or no M in 13..17 worked.
    # But M=17 is known to work.
    # So we return the seed.
    return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
