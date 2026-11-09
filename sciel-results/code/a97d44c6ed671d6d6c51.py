import sys
import random
import time

def construct():
    """
    Attempts to find a 4-coloring of K_49 without monochromatic triangles
    using local search. Falls back to n=32 if unsuccessful.
    """
    target_n = 49
    colors = [random.randint(0, 3) for _ in range(target_n * (target_n - 1) // 2)]
    
    def get_color(i, j):
        if i > j: i, j = j, i
        idx = i * target_n - (i + 1) // 2 + (j - i - 1)
        # Simpler index:
        # Number of edges before row i: i * (i - 1) // 2 + i * (target_n - i - 1) ? No.
        # Lexicographic order:
        # (0,1), (0,2), ..., (0,n-1)
        # (1,2), (1,3), ..., (1,n-1)
        # ...
        # Index for (i,j) with i<j:
        # Sum of (n - k - 1) for k=0 to i-1 is i*n - i*(i+1)//2 - i ?
        # Let's just use a 2D array for speed in the search.
        pass

    # Use 2D array for speed
    n = target_n
    C = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            C[i][j] = random.randint(0, 3)
            C[j][i] = C[i][j]

    def count_triangles():
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if C[i][j] == C[j][k] == C[i][k]:
                        count += 1
        return count

    def fix_triangles():
        # Find a monochromatic triangle and change one edge
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if C[i][j] == C[j][k] == C[i][k]:
                        # Change C[i][j]
                        old_c = C[i][j]
                        new_c = (old_c + 1) % 4
                        C[i][j] = new_c
                        C[j][i] = new_c
                        return True
        return False

    start_time = time.time()
    max_time = 110 # Leave 10s for validation
    
    # Initial check
    if count_triangles() == 0:
        return _finalize(n, C)

    while time.time() - start_time < max_time:
        if fix_triangles():
            if count_triangles() == 0:
                return _finalize(n, C)
        else:
            # No triangles found?
            return _finalize(n, C)
            
    # If timeout, fall back to seed
    return _seed_construct()

def _finalize(n, C):
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            colors.append(C[i][j])
    # Final verification
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if C[i][j] == C[j][k] == C[i][k]:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    if valid:
        return {'n': n, 'colors': colors}
    else:
        return _seed_construct()

def _seed_construct():
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}
