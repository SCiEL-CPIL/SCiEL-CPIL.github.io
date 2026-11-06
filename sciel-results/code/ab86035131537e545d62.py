import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Try n=17 down to 9
    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
            
        # Try multiple restarts
        for _ in range(500):
            if time.time() > deadline:
                break
            res = _solve_n(n)
            if res is not None:
                return {'n': n, 'colors': res}
                
    # Fallback n=9
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def _solve_n(n):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    A = [[0] * n for _ in range(4)]
    
    # Precompute masks for bits
    # Not needed, using 1<<k
    
    def has_path_len_4(u, v, c):
        # Fast bitset check for walk of length 4
        # R1 = A[c]
        # R2[i] = OR of R1[j] for j in R1[i]
        # R4[i] = OR of R2[j] for j in R2[i]
        
        # Compute R2
        R2 = [0] * n
        for i in range(n):
            mask = A[c][i]
            union = 0
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                union |= A[c][j]
                mask ^= lsb
            R2[i] = union
            
        # Compute R4
        R4 = [0] * n
        for i in range(n):
            mask = R2[i]
            union = 0
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                union |= R2[j]
                mask ^= lsb
            R4[i] = union
            
        # Check if v is in R4[u]
        if not (R4[u] & (1 << v)):
            return False
            
        # If walk exists, verify path with DFS
        def dfs(node, depth, path_mask):
            if depth == 4:
                return node == v
            if node == v:
                return False
            neighbors = A[c][node]
            avail = neighbors & ~path_mask
            while avail:
                lsb = avail & -avail
                w = lsb.bit_length() - 1
                avail ^= lsb
                if dfs(w, depth + 1, path_mask | lsb):
                    return True
            return False
            
        return dfs(u, 0, 1 << u)

    def backtrack(idx):
        if idx == len(edges):
            return True
        u, v = edges[idx]
        colors_to_try = [0, 1, 2, 3]
        random.shuffle(colors_to_try)
        
        for c in colors_to_try:
            if has_path_len_4(u, v, c):
                continue
                
            A[c][u] |= 1 << v
            A[c][v] |= 1 << u
            
            if backtrack(idx + 1):
                return True
                
            A[c][u] &= ~(1 << v)
            A[c][v] &= ~(1 << u)
            
        return False

    if backtrack(0):
        res = []
        for i in range(n):
            for j in range(i + 1, n):
                for c in range(4):
                    if A[c][i] & (1 << j):
                        res.append(c)
                        break
        return res
    return None

# EVOLVE-BLOCK-END
