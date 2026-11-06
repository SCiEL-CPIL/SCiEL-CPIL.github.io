import time
import itertools

def construct():
    start_time = time.time()
    deadline = 140  # 140s safety margin
    
    def check_triangle_free(D, N):
        """
        Checks if the circulant graph defined by difference set D (subset of 1..N//2)
        is triangle-free.
        D contains differences d. Edges exist for d and -d (which is N-d).
        """
        # E contains all non-zero differences that form edges
        E = set()
        for d in D:
            E.add(d % N)
            E.add((-d) % N)
        
        # Check for x, y in E such that x+y in E and x != -y
        # Optimization: iterate through E
        for x in E:
            for y in E:
                if x == (-y) % N:
                    continue
                if (x + y) % N in E:
                    return False
        return True

    def solve_circulant(N):
        """
        Attempts to partition {1, ..., N//2} into 4 sets D0, D1, D2, D3
        such that each defines a triangle-free circulant graph.
        """
        half = N // 2
        elements = list(range(1, half + 1))
        
        # Backtracking search
        # We assign each element to one of 4 sets
        # To reduce symmetry, we can fix the first element to set 0, etc.
        # But simple backtracking with pruning might be enough for N=50 (25 elements)
        
        D = [set() for _ in range(4)]
        
        # Pre-compute potential conflicts?
        # A triangle is formed if we add a new difference d to D_k,
        # and there exist x, y in D_k (or -D_k) such that x+y=d or d-x=y etc.
        
        def backtrack(idx):
            if idx == len(elements):
                return True
            
            d = elements[idx]
            
            for k in range(4):
                # Add d to D[k]
                D[k].add(d)
                
                # Check if D[k] is still triangle-free
                if check_triangle_free(D[k], N):
                    if backtrack(idx + 1):
                        return True
                
                # Remove d from D[k]
                D[k].remove(d)
                
            return False

        # Heuristic: sort elements? Maybe larger differences are more constrained?
        # Or just 1..half.
        
        return backtrack(0), D

    # Try N=50
    if time.time() < deadline - 30:
        sol, D = solve_circulant(50)
        if sol:
            n = 50
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    diff = (j - i) % n
                    if diff > n // 2:
                        diff = n - diff
                    # Find which set contains diff
                    for k in range(4):
                        if diff in D[k]:
                            colors.append(k)
                            break
                    else:
                        # Should not happen
                        colors.append(0)
            return {'n': n, 'colors': colors}

    # Try N=40
    if time.time() < deadline - 15:
        sol, D = solve_circulant(40)
        if sol:
            n = 40
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    diff = (j - i) % n
                    if diff > n // 2:
                        diff = n - diff
                    for k in range(4):
                        if diff in D[k]:
                            colors.append(k)
                            break
                    else:
                        colors.append(0)
            return {'n': n, 'colors': colors}

    # Fallback to N=32 using the seed's method (GF(16))
    # This guarantees score 1.0
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
