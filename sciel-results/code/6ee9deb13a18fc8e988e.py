import time
import random

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 50.0

    # Standard reduced Latin square: L1[i][j] = (i + j) % n
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [L1]

    def find_orthogonal_mate(existing_squares):
        """
        Find a new Latin square L_new such that:
        1. L_new[0][j] == j for all j (first row fixed to identity)
        2. L_new is orthogonal to all squares in existing_squares
        """
        L_new = [[-1 for _ in range(n)] for _ in range(n)]
        # Fix first row
        for j in range(n):
            L_new[0][j] = j

        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        
        # Initialize used sets with first row
        for j in range(n):
            row_used[0].add(j)
            col_used[j].add(j)
            
        # Track used pairs for each existing square
        # pair_used[k] is a set of tuples (L_k[i][j], L_new[i][j])
        pair_used = [set() for _ in existing_squares]
        for idx, L_k in enumerate(existing_squares):
            for j in range(n):
                pair_used[idx].add((L_k[0][j], L_new[0][j]))

        def is_valid(i, j, val):
            if val in row_used[i]:
                return False
            if val in col_used[j]:
                return False
            for idx, L_k in enumerate(existing_squares):
                if (L_k[i][j], val) in pair_used[idx]:
                    return False
            return True

        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == n * n:
                return True
            
            i = pos // n
            j = pos % n
            
            # Skip first row
            if i == 0:
                return backtrack(pos + 1)
                
            # Get possible values
            possible = [v for v in range(n) if is_valid(i, j, v)]
            # Heuristic: Try values that are least constrained? 
            # For now, random shuffle is okay, but let's try to be smarter.
            # We can sort by the number of conflicts in the remaining cells of the row/col?
            # Simple: random
            random.shuffle(possible)
            
            for val in possible:
                L_new[i][j] = val
                row_used[i].add(val)
                col_used[j].add(val)
                
                new_pairs = []
                for idx, L_k in enumerate(existing_squares):
                    p = (L_k[i][j], val)
                    new_pairs.append(p)
                    pair_used[idx].add(p)
                
                if backtrack(pos + 1):
                    return True
                
                L_new[i][j] = -1
                row_used[i].remove(val)
                col_used[j].remove(val)
                for idx, p in enumerate(new_pairs):
                    pair_used[idx].remove(p)
        
        # Start from index n (row 1, col 0)
        if backtrack(n):
            return L_new
        return None

    # Try to find more squares
    current_squares = [L1]
    
    # We want to find up to 9 squares total
    # We have 1. Try to find 8 more.
    for _ in range(8):
        if time.time() > deadline:
            break
        next_sq = find_orthogonal_mate(current_squares)
        if next_sq is not None:
            current_squares.append(next_sq)
        else:
            # If we can't find another one with the current set, 
            # we might be stuck. 
            # Since we fixed the first row to identity, we might miss some solutions.
            # But for n=10, it's likely we can find a few.
            # If we fail, we break.
            break
            
    return current_squares
