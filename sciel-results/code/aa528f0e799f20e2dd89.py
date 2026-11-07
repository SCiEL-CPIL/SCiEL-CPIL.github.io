import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s to leave margin for 90s limit
    n = 10
    
    # First square: standard cyclic
    squares = [[[(i + j) % n for j in range(n)] for i in range(n)]]
    
    def is_valid_lsq(sq):
        """Check if sq is a valid Latin square of order n."""
        for i in range(n):
            row = set(sq[i])
            if row != set(range(n)):
                return False
            col = set(sq[i][j] for i in range(n))
            if col != set(range(n)):
                return False
        return True
    
    def check_orthogonal(sq1, sq2):
        """Check if two Latin squares are orthogonal."""
        pairs = set()
        for i in range(n):
            for j in range(n):
                pair = (sq1[i][j], sq2[i][j])
                if pair in pairs:
                    return False
                pairs.add(pair)
        return len(pairs) == n * n
    
    def check_orthogonal_to_all(sq, existing_squares):
        """Check if sq is orthogonal to all existing squares."""
        for es in existing_squares:
            if not check_orthogonal(sq, es):
                return False
        return True
    
    def backtrack(sq, row, col, row_masks, col_masks, used_pairs):
        """
        Backtracking to fill Latin square sq.
        row_masks[i] is a bitmask of symbols used in row i.
        col_masks[j] is a bitmask of symbols used in col j.
        used_pairs is a list of sets, one per existing square, tracking (existing_val, new_val) pairs.
        """
        if row == n:
            return True
        
        next_col = col + 1
        next_row = row
        if next_col == n:
            next_row = row + 1
            next_col = 0
        
        # Determine available symbols
        available = ((1 << n) - 1) & ~row_masks[row] & ~col_masks[col]
        
        # For each available symbol, check orthogonality constraints
        for sym in range(n):
            if not (available & (1 << sym)):
                continue
            
            # Check orthogonality with all existing squares
            valid = True
            for k, es in enumerate(existing_squares_ref):
                if used_pairs[k] is not None and (es[row][col], sym) in used_pairs[k]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Place the symbol
            sq[row][col] = sym
            row_masks[row] |= (1 << sym)
            col_masks[col] |= (1 << sym)
            
            # Update used_pairs
            for k, es in enumerate(existing_squares_ref):
                if used_pairs[k] is not None:
                    used_pairs[k].add((es[row][col], sym))
            
            if backtrack(sq, next_row, next_col, row_masks, col_masks, used_pairs):
                return True
            
            # Undo
            sq[row][col] = -1
            row_masks[row] &= ~(1 << sym)
            col_masks[col] &= ~(1 << sym)
            for k, es in enumerate(existing_squares_ref):
                if used_pairs[k] is not None:
                    used_pairs[k].discard((es[row][col], sym))
        
        return False
    
    # Global reference for closure
    existing_squares_ref = squares
    
    def try_find_new_square(max_restarts=100):
        """Try to find a new Latin square orthogonal to all existing ones."""
        global existing_squares_ref
        existing_squares_ref = squares
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize square with first row and column fixed
            sq = [[-1] * n for _ in range(n)]
            for i in range(n):
                sq[0][i] = i  # First row: 0,1,2,...,n-1
                sq[i][0] = i  # First column: 0,1,2,...,n-1
            
            row_masks = [0] * n
            col_masks = [0] * n
            for i in range(n):
                row_masks[0] |= (1 << i)
                col_masks[i] |= (1 << i)
            
            used_pairs = [set() for _ in range(len(squares))]
            for k, es in enumerate(squares):
                for i in range(n):
                    for j in range(n):
                        if sq[i][j] != -1:
                            used_pairs[k].add((es[i][j], sq[i][j]))
            
            # Shuffle the order of trying symbols for diversity
            # We'll just run backtracking with randomized symbol order
            success = backtrack_randomized(sq, 1, 0, row_masks, col_masks, used_pairs)
            
            if success:
                # Verify it's a valid Latin square
                if is_valid_lsq(sq):
                    # Verify orthogonality
                    if check_orthogonal_to_all(sq, squares):
                        return sq
        
        return None
    
    def backtrack_randomized(sq, row, col, row_masks, col_masks, used_pairs):
        """Backtracking with randomized symbol order."""
        if row == n:
            return True
        
        next_col = col + 1
        next_row = row
        if next_col == n:
            next_row = row + 1
            next_col = 0
        
        available = ((1 << n) - 1) & ~row_masks[row] & ~col_masks[col]
        
        # Get available symbols and shuffle
        avail_syms = [s for s in range(n) if (available & (1 << s))]
        random.shuffle(avail_syms)
        
        for sym in avail_syms:
            # Check orthogonality
            valid = True
            for k, es in enumerate(existing_squares_ref):
                if (es[row][col], sym) in used_pairs[k]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            sq[row][col] = sym
            row_masks[row] |= (1 << sym)
            col_masks[col] |= (1 << sym)
            
            for k, es in enumerate(existing_squares_ref):
                used_pairs[k].add((es[row][col], sym))
            
            if backtrack_randomized(sq, next_row, next_col, row_masks, col_masks, used_pairs):
                return True
            
            sq[row][col] = -1
            row_masks[row] &= ~(1 << sym)
            col_masks[col] &= ~(1 << sym)
            for k, es in enumerate(existing_squares_ref):
                used_pairs[k].discard((es[row][col], sym))
        
        return False
    
    # Try to find additional squares
    max_attempts = 8  # Try to find up to 8 more (total 9, but realistically few)
    for _ in range(max_attempts):
        if time.time() > deadline:
            break
        
        new_sq = try_find_new_square(max_restarts=50)
        if new_sq is not None:
            squares.append(new_sq)
        else:
            # Could not find another square, stop
            break
    
    # Ensure we return at least 1 square (we already have the first one)
    # Convert to pure Python lists of ints
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
    
    return result
