import random
import time

P = {'n': 10}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # 1. Construct the first Latin Square L1
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # 2. Attempt to find a second Latin Square L2 orthogonal to L1
    # We use a random restart local search
    best_L2 = None
    
    for restart in range(50):  # Max 50 restarts
        if time.time() > deadline:
            break
            
        # Initialize L2 with random permutations in each row
        L2 = []
        for i in range(n):
            row = list(range(n))
            random.shuffle(row)
            L2.append(row)
        
        # Local search
        current_conflicts = count_conflicts(L1, L2, n)
        best_conflicts = current_conflicts
        best_L2_candidate = [row[:] for row in L2]
        
        iterations = 0
        max_iterations = 20000
        while iterations < max_iterations:
            if time.time() > deadline:
                break
                
            # Pick a random cell to change
            r = random.randint(0, n - 1)
            c = random.randint(0, n - 1)
            
            # Try to swap with a random other column in the same row
            c2 = random.randint(0, n - 1)
            if c == c2:
                continue
                
            # Perform swap
            L2[r][c], L2[r][c2] = L2[r][c2], L2[r][c]
            new_conflicts = count_conflicts(L1, L2, n)
            
            # Accept if better or equal (with some probability for escape)
            if new_conflicts <= current_conflicts:
                current_conflicts = new_conflicts
                if new_conflicts < best_conflicts:
                    best_conflicts = new_conflicts
                    best_L2_candidate = [row[:] for row in L2]
                    if best_conflicts == 0:
                        break
            else:
                # Revert if worse (simple hill climbing)
                L2[r][c], L2[r][c2] = L2[r][c2], L2[r][c]
            
            iterations += 1
            
        if best_conflicts == 0:
            best_L2 = best_L2_candidate
            break
            
    if best_L2 is not None:
        # Ensure types are Python int
        result = []
        for sq in [L1, best_L2]:
            result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
        return result
    else:
        # Return just L1 if L2 not found
        result = [[int(L1[i][j]) for j in range(n)] for i in range(n)]
        return [result]

def count_conflicts(L1, L2, n):
    # Count column conflicts in L2
    col_conflicts = 0
    for c in range(n):
        col = [L2[r][c] for r in range(n)]
        if len(set(col)) != n:
            col_conflicts += 1 # Simple count, could be more precise
    
    # Count pair conflicts
    pairs = set()
    pair_conflicts = 0
    for i in range(n):
        for j in range(n):
            p = (L1[i][j], L2[i][j])
            if p in pairs:
                pair_conflicts += 1
            else:
                pairs.add(p)
                
    return col_conflicts + pair_conflicts
