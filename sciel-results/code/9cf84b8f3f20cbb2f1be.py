import random
import time

def construct():
    n = 10
    deadline = time.time() + 55.0
    
    def generate_latin_square():
        grid = [[-1 for _ in range(n)] for _ in range(n)]
        
        # Fix first row to 0..9
        for j in range(n):
            grid[0][j] = j
            
        # Fix first column to 0..9
        for i in range(n):
            grid[i][0] = i
            
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if grid[i][j] != -1:
                    row_used[i].add(grid[i][j])
                    col_used[j].add(grid[i][j])

        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == n * n:
                return True
                
            i = idx // n
            j = idx % n
            
            if grid[i][j] != -1:
                return backtrack(idx + 1)
                
            candidates = [v for v in range(n) if v not in row_used[i] and v not in col_used[j]]
            
            if not candidates:
                return False
                
            random.shuffle(candidates)
            
            for v in candidates:
                grid[i][j] = v
                row_used[i].add(v)
                col_used[j].add(v)
                
                if backtrack(idx + 1):
                    return True
                    
                grid[i][j] = -1
                row_used[i].remove(v)
                col_used[j].remove(v)
                
            return False
            
        if backtrack(0):
            return grid
        return None

    def find_orthogonal_mate(L1):
        L2 = [[-1 for _ in range(n)] for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        pair_used = [set() for _ in range(n)]
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == n * n:
                return True
                
            i = idx // n
            j = idx % n
            a = L1[i][j]
            
            candidates = [v for v in range(n) if v not in row_used[i] and v not in col_used[j] and v not in pair_used[a]]
            
            if not candidates:
                return False
                
            random.shuffle(candidates)
            
            for v in candidates:
                L2[i][j] = v
                row_used[i].add(v)
                col_used[j].add(v)
                pair_used[a].add(v)
                
                if backtrack(idx + 1):
                    return True
                    
                L2[i][j] = -1
                row_used[i].remove(v)
                col_used[j].remove(v)
                pair_used[a].remove(v)
                
            return False
            
        if backtrack(0):
            return L2
        return None

    attempts = 0
    max_attempts = 50
    
    while attempts < max_attempts and time.time() < deadline - 5:
        attempts += 1
        
        L1 = generate_latin_square()
        if L1 is None:
            continue
            
        # Optional: Verify L1
        # (Omitted for speed, assuming generator is correct)
        
        L2 = find_orthogonal_mate(L1)
        if L2 is not None:
            # Verify Orthogonality just in case
            pairs = set()
            for i in range(n):
                for j in range(n):
                    pairs.add((L1[i][j], L2[i][j]))
            if len(pairs) == n*n:
                L1_clean = [[int(L1[i][j]) for j in range(n)] for i in range(n)]
                L2_clean = [[int(L2[i][j]) for j in range(n)] for i in range(n)]
                return [L1_clean, L2_clean]

    # Fallback
    L1_fallback = [[(i+j)%n for j in range(n)] for i in range(n)]
    return [L1_fallback]
