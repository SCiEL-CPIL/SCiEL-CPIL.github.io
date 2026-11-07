import time

def construct():
    deadline = time.time() + 55.0
    n = 10
    
    # Base square: addition table mod 10
    A0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [A0]
    
    def find_second_square():
        grid = [[0]*n for _ in range(n)]
        row_masks = [0]*n
        col_masks = [0]*n
        # pair_used[a][b] is True if the pair (a, b) has been used in orthogonality check
        # a is from A0, b is from A1
        pair_used = [[False]*n for _ in range(n)]
        
        # Initialize first row of A1 to identity (0..9)
        # This is a valid normalization for Latin squares orthogonal to A0
        # because we can permute symbols of A1 to make the first row identity
        # (since A0[0][j] = j, the pairs are (j, j). If we permute symbols in A1,
        # we change the second component of the pairs.
        # Wait, if we fix A1[0][j] = j, we are fixing the permutation.
        # Is it always possible to find a solution with A1[0][j] = j?
        # Yes, because for any orthogonal mate A1', we can apply a permutation pi to the symbols of A1'
        # such that pi(A1'[0][j]) = j for all j.
        # Since A1'[0] is a permutation of 0..9, such a pi exists.
        # This permutation preserves the Latin property and orthogonality.
        
        for j in range(n):
            grid[0][j] = j
            row_masks[0] |= (1 << j)
            col_masks[j] |= (1 << j)
            pair_used[A0[0][j]][j] = True
            
        def backtrack(i, j):
            if time.time() > deadline:
                return False
            if i == n:
                return True
            
            next_i, next_j = (i+1, 0) if j == n-1 else (i, j+1)
            
            for s in range(n):
                bit = 1 << s
                if (row_masks[i] & bit) or (col_masks[j] & bit):
                    continue
                if pair_used[A0[i][j]][s]:
                    continue
                
                # Place s
                grid[i][j] = s
                row_masks[i] |= bit
                col_masks[j] |= bit
                pair_used[A0[i][j]][s] = True
                
                if backtrack(next_i, next_j):
                    return True
                
                # Unplace
                row_masks[i] &= ~bit
                col_masks[j] &= ~bit
                pair_used[A0[i][j]][s] = False
                
            return False
        
        if backtrack(1, 0):
            return grid
        return None

    A1 = find_second_square()
    if A1 is not None:
        # Verify strictly
        ok = True
        # Check Latin
        for i in range(n):
            if len(set(A1[i])) != n:
                ok = False
                break
        if ok:
            for j in range(n):
                col = [A1[i][j] for i in range(n)]
                if len(set(col)) != n:
                    ok = False
                    break
        if ok:
            # Check Orthogonal
            seen_pairs = set()
            for i in range(n):
                for j in range(n):
                    p = (A0[i][j], A1[i][j])
                    if p in seen_pairs:
                        ok = False
                        break
                    seen_pairs.add(p)
                if not ok:
                    break
        if ok:
            squares.append(A1)
            
    return squares
