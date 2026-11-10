import time

DEADLINE_MARGIN = 8.0  # Safety margin before 60s limit

def _is_latin(sq):
    n = len(sq)
    for i in range(n):
        if len(set(sq[i])) != n:
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if len(set(col)) != n:
            return False
    return True

def _is_orthogonal(a, b):
    n = len(a)
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (a[i][j], b[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True

def _find_orthogonal_cyclic(existing_squares, n, deadline):
    """
    Find a Latin square orthogonal to all existing squares.
    Uses bitmask-based constraint tracking with MRV heuristic.
    
    For a cyclic Latin square S1[i][j] = (i+j)%n, orthogonality means
    each broken diagonal (cells with same (i+j)%n) must be a permutation.
    """
    # Precompute diagonal index for each cell based on first existing square
    # For cyclic square: diag[i][j] = (i+j) % n
    # For general case: diag[i][j] = existing_squares[0][i][j]
    diag_of = [[existing_squares[0][i][j] for j in range(n)] for i in range(n)]
    
    # For multiple existing squares, we need pair constraints
    # But for the first square (cyclic), the diagonal constraint is sufficient
    # For subsequent squares, we need full pair checking
    
    sq = [[-1] * n for _ in range(n)]
    row_masks = [0] * n
    col_masks = [0] * n
    diag_masks = [0] * n  # For orthogonality to first (cyclic) square
    
    # For additional existing squares, track used pairs per square
    num_existing = len(existing_squares)
    # pairs_used[k][sym_at_position] = bitmask of symbols used in square k at positions where existing_squares[k] has that value
    # Actually, for orthogonality to square k: for each symbol value v in square k,
    # the new square must use each symbol exactly once at positions where square k has value v
    # So: pair_mask[k][v] = bitmask of symbols used in new square at positions where square k = v
    pair_masks = [[0] * n for _ in range(num_existing)]
    
    # Count remaining cells
    remaining = n * n
    
    def get_candidates(i, j):
        """Get bitmask of valid symbols for position (i,j)"""
        d = diag_of[i][j]
        # Combine all constraint masks
        mask = row_masks[i] | col_masks[j] | diag_masks[d]
        for k in range(num_existing):
            v = existing_squares[k][i][j]
            mask |= pair_masks[k][v]
        
        # Available symbols: bits not set in mask
        avail = 0
        for s in range(n):
            if not (mask & (1 << s)):
                avail |= (1 << s)
        return avail
    
    def count_available(i, j):
        """Count available symbols for cell (i,j)"""
        avail = get_candidates(i, j)
        count = 0
        while avail:
            count += 1
            avail &= avail - 1
        return count
    
    def backtrack(remaining_count):
        nonlocal sq, row_masks, col_masks, diag_masks, pair_masks, remaining
        
        if time.time() > deadline:
            return False
        
        if remaining_count == 0:
            return True
        
        # MRV: Find cell with fewest available symbols
        best_i, best_j = -1, -1
        best_count = n + 1
        best_avail = 0
        
        for i in range(n):
            for j in range(n):
                if sq[i][j] == -1:
                    avail = get_candidates(i, j)
                    if avail == 0:
                        return False  # Dead end
                    cnt = 0
                    temp = avail
                    while temp:
                        cnt += 1
                        temp &= temp - 1
                    if cnt < best_count:
                        best_count = cnt
                        best_i, best_j = i, j
                        best_avail = avail
                        if cnt == 1:
                            break
            if best_count == 1:
                break
        
        if best_i == -1:
            return True
        
        i, j = best_i, best_j
        d = diag_of[i][j]
        
        # Try each available symbol
        avail = best_avail
        while avail:
            bit = avail & (-avail)
            avail ^= bit
            sym = bit.bit_length() - 1
            
            # Check all constraints (already filtered by get_candidates, but double-check)
            if (row_masks[i] & bit) or (col_masks[j] & bit) or (diag_masks[d] & bit):
                continue
            
            valid = True
            for k in range(num_existing):
                v = existing_squares[k][i][j]
                if pair_masks[k][v] & bit:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Place
            sq[i][j] = sym
            row_masks[i] |= bit
            col_masks[j] |= bit
            diag_masks[d] |= bit
            for k in range(num_existing):
                v = existing_squares[k][i][j]
                pair_masks[k][v] |= bit
            
            if backtrack(remaining_count - 1):
                return True
            
            # Undo
            sq[i][j] = -1
            row_masks[i] &= ~bit
            col_masks[j] &= ~bit
            diag_masks[d] &= ~bit
            for k in range(num_existing):
                v = existing_squares[k][i][j]
                pair_masks[k][v] &= ~bit
        
        return False
    
    # Symmetry breaking: fix sq[0][0] to reduce search space
    # Try different starting symbols
    for first_sym in range(n):
        bit = 1 << first_sym
        # Check if first_sym is valid at (0,0)
        d00 = diag_of[0][0]
        if row_masks[0] & bit or col_masks[0] & bit or diag_masks[d00] & bit:
            continue
        
        valid = True
        for k in range(num_existing):
            v = existing_squares[k][0][0]
            if pair_masks[k][v] & bit:
                valid = False
                break
        
        if not valid:
            continue
        
        # Place first_sym at (0,0)
        sq[0][0] = first_sym
        row_masks[0] |= bit
        col_masks[0] |= bit
        diag_masks[d00] |= bit
        for k in range(num_existing):
            v = existing_squares[k][0][0]
            pair_masks[k][v] |= bit
        
        if backtrack(n * n - 1):
            return sq
        
        # Undo
        sq[0][0] = -1
        row_masks[0] &= ~bit
        col_masks[0] &= ~bit
        diag_masks[d00] &= ~bit
        for k in range(num_existing):
            v = existing_squares[k][0][0]
            pair_masks[k][v] &= ~bit
    
    return None

def construct():
    start_time = time.time()
    n = 10
    deadline = start_time + (60.0 - DEADLINE_MARGIN)
    
    # S1: Cyclic Latin square
    s1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [s1]
    
    # Try to find S2 orthogonal to S1
    s2 = _find_orthogonal_cyclic(squares, n, deadline)
    if s2 is not None:
        if _is_latin(s2) and all(_is_orthogonal(s2, sq) for sq in squares):
            squares.append(s2)
    
    # Try to find S3 if we have time
    remaining_time = deadline - time.time()
    if len(squares) >= 2 and remaining_time > 10.0:
        s3 = _find_orthogonal_cyclic(squares, n, deadline)
        if s3 is not None:
            if _is_latin(s3) and all(_is_orthogonal(s3, sq) for sq in squares):
                squares.append(s3)
    
    # Try to find S4 if we have time
    remaining_time = deadline - time.time()
    if len(squares) >= 3 and remaining_time > 15.0:
        s4 = _find_orthogonal_cyclic(squares, n, deadline)
        if s4 is not None:
            if _is_latin(s4) and all(_is_orthogonal(s4, sq) for sq in squares):
                squares.append(s4)
    
    # Convert to list of lists of ints (ensure Python int type)
    final_res = []
    for sq in squares:
        new_sq = []
        for row in sq:
            new_row = [int(v) for v in row]
            new_sq.append(new_row)
        final_res.append(new_sq)
    
    return final_res
