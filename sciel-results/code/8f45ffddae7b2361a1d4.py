import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    squares = [L1]
    
    def is_valid_new_square(i, j, val, new_square, existing_squares):
        # Check row/col in new_square
        if val in new_square[i]:
            return False
        for r in range(n):
            if new_square[r][j] == val:
                return False
        
        # Check orthogonality with all existing squares
        # For each existing square, the tuple of values at (i,j) for all squares (including new) must be unique.
        # Actually, we check if the new tuple is already present in the set of tuples.
        # But we don't have a global set of tuples easily accessible in this recursive step without precomputation.
        # Alternative: For each existing square S_k, the pair (S_k[i][j], new_square[i][j]) must be unique across the grid?
        # No, the condition is that the vector of values (S_1[i][j], ..., S_k[i][j], new_square[i][j]) is unique for all (i,j).
        # This is equivalent to: For any other cell (r,c), the vector must not match.
        # This is expensive to check on the fly for every cell.
        
        # Better approach:
        # Maintain a set of used tuples.
        # When we place a value, we add the tuple to the set.
        # When we backtrack, we remove it.
        
        pass # This logic is handled in the search function below

    def search_for_square(existing_squares):
        new_square = [[-1 for _ in range(n)] for _ in range(n)]
        used_tuples = set()
        
        # Precompute existing tuples? No, they change as we fill new_square.
        # But the existing parts of the tuple are fixed.
        
        def get_tuple(i, j, val):
            t = tuple(s[i][j] for s in existing_squares)
            return t + (val,)

        def is_valid(i, j, val):
            # Row/Col check
            if val in new_square[i]:
                return False
            for r in range(n):
                if new_square[r][j] == val:
                    return False
            
            # Orthogonality check
            # The tuple must not be in used_tuples
            # AND it must not conflict with any already filled cell?
            # Wait, if we maintain used_tuples correctly, it handles conflicts with filled cells.
            # But we also need to ensure that for any UNFILLED cell, it doesn't conflict?
            # No, backtracking handles that.
            # The only check needed is:
            # 1. Row/Col uniqueness in new_square.
            # 2. The tuple (existing values + val) is not already in used_tuples.
            
            t = get_tuple(i, j, val)
            if t in used_tuples:
                return False
                
            return True

        def backtrack_mrv():
            if time.time() > deadline:
                return False
            
            min_count = n + 1
            min_i = -1
            min_j = -1
            min_vals = []
            
            for i in range(n):
                for j in range(n):
                    if new_square[i][j] == -1:
                        vals = []
                        for val in range(n):
                            if is_valid(i, j, val):
                                vals.append(val)
                        if len(vals) < min_count:
                            min_count = len(vals)
                            min_i = i
                            min_j = j
                            min_vals = vals
                            if min_count == 0:
                                return False
                            if min_count == 1:
                                break
                if min_count == 1:
                    break
            
            if min_i == -1:
                return True # All filled
            
            for val in min_vals:
                new_square[min_i][min_j] = val
                t = get_tuple(min_i, min_j, val)
                used_tuples.add(t)
                
                if backtrack_mrv():
                    return True
                
                new_square[min_i][min_j] = -1
                used_tuples.remove(t)
            
            return False

        if backtrack_mrv():
            # Final verification
            # Check Latin property
            for i in range(n):
                if sorted(new_square[i]) != list(range(n)):
                    return None
            for j in range(n):
                if sorted([new_square[i][j] for i in range(n)]) != list(range(n)):
                    return None
            
            # Check Orthogonality
            # used_tuples should have size n*n
            if len(used_tuples) != n * n:
                return None
                
            return new_square

    # Try to find L2
    L2 = search_for_square(squares)
    if L2:
        squares.append(L2)
        
        # Try to find L3
        L3 = search_for_square(squares)
        if L3:
            squares.append(L3)
            
    # Ensure types
    for s in squares:
        for i in range(n):
            for j in range(n):
                if not isinstance(s[i][j], int):
                    s[i][j] = int(s[i][j])
            if not isinstance(s[i], list):
                s[i] = list(s[i])
        if not isinstance(s, list):
            s = list(s)
            
    if not isinstance(squares, list):
        squares = list(squares)
        
    return squares
