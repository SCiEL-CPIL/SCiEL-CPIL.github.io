import time
import random

def construct():
    n = 10
    # Base square
    L0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    squares = [L0]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds limit to be safe
    
    # Try to find more orthogonal squares
    # We will try to find up to 8 more squares (total 9)
    # Using backtracking
    
    def is_orthogonal(L1, L2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pair = (L1[i][j], L2[i][j])
                if pair in pairs:
                    return False
                pairs.add(pair)
        return True

    def backtrack(square, row, col, used_rows, used_cols, used_pairs):
        if row == n:
            if col > 0:
                return True
            return False
        
        if col == n:
            return backtrack(square, row + 1, 0, used_rows, used_cols, used_pairs)
        
        # Try all symbols
        for sym in range(n):
            if sym in used_rows[row]:
                continue
            if sym in used_cols[col]:
                continue
            
            # Check orthogonality with all existing squares
            valid = True
            for k in range(len(squares)):
                existing_sym = squares[k][row][col]
                pair = (existing_sym, sym)
                if pair in used_pairs[k]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Place symbol
            square[row][col] = sym
            used_rows[row].add(sym)
            used_cols[col].add(sym)
            for k in range(len(squares)):
                existing_sym = squares[k][row][col]
                used_pairs[k].add((existing_sym, sym))
            
            if backtrack(square, row, col + 1, used_rows, used_cols, used_pairs):
                return True
            
            # Unplace
            square[row][col] = -1
            used_rows[row].remove(sym)
            used_cols[col].remove(sym)
            for k in range(len(squares)):
                existing_sym = squares[k][row][col]
                used_pairs[k].remove((existing_sym, sym))
        
        return False

    # Try to find additional squares
    for _ in range(8):  # Try to find 8 more
        if time.time() > deadline:
            break
        
        new_square = [[-1 for _ in range(n)] for _ in range(n)]
        used_rows = [set() for _ in range(n)]
        used_cols = [set() for _ in range(n)]
        used_pairs = [set() for _ in range(len(squares))]
        
        # Randomize order of symbols to avoid deterministic failure?
        # For now, try standard order.
        
        if backtrack(new_square, 0, 0, used_rows, used_cols, used_pairs):
            squares.append(new_square)
        else:
            # If we fail, we might break or try again with different randomization
            # But backtracking is exhaustive, so if it fails, no solution exists with this prefix?
            # No, backtracking explores all possibilities. If it returns False, no Latin square orthogonal to existing ones exists?
            # That would mean we found a complete set.
            # But we know 9 exist. So it should find them.
            # If it fails, it might be too slow.
            break

    return squares
