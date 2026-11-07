import time

N = 10
SYMS = list(range(N))

def _validate(squares):
    if not squares:
        return False
    for L in squares:
        for r in range(N):
            if sorted(L[r]) != SYMS:
                return False
        for c in range(N):
            col = [L[r][c] for r in range(N)]
            if sorted(col) != SYMS:
                return False
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            seen = set()
            for r in range(N):
                for c in range(N):
                    p = (squares[i][r][c], squares[j][r][c])
                    if p in seen:
                        return False
                    seen.add(p)
    return True

def _build_square(base_squares, deadline):
    n = N
    # We assume base_squares has exactly 1 square for the 2nd square search
    # If more, we can extend, but for n=10 we only need 2.
    if not base_squares:
        return None
        
    # Bitmask representation for constraints
    # row_masks[r] has bits set for symbols used in row r
    # col_masks[c] has bits set for symbols used in col c
    # pair_masks[idx][sym_base] has bits set for symbols used in pair (base_sym, new_sym)
    # Actually, for orthogonality with multiple base squares:
    # For each base square idx, and for each symbol b in that base square,
    # we need to ensure that the new symbol s is unique for each position (r,c) 
    # where base[idx][r][c] == b.
    # This is equivalent to: for each base square idx, the mapping 
    # (base[idx][r][c] -> new[r][c]) must be a bijection? No.
    # Orthogonality: The set of pairs {(base[r][c], new[r][c])} must be all distinct.
    # This means for a fixed base symbol b, the new symbols assigned to cells 
    # where base has b must be all distinct.
    
    # So for each base square idx, we maintain a mask for each symbol b in 0..9
    # pair_masks[idx][b] is a bitmask of new symbols already used in cells 
    # where base[idx] has symbol b.
    
    row_masks = [0] * n
    col_masks = [0] * n
    pair_masks = [[0] * n for _ in range(len(base_squares))]
    
    grid = [[-1] * n for _ in range(n)]
    
    # Fix first row to identity to break symmetry
    # This is valid if we can permute symbols of the new square.
    # We assume we can.
    for j in range(n):
        s = j
        grid[0][j] = s
        row_masks[0] |= (1 << s)
        col_masks[j] |= (1 << s)
        for idx, B in enumerate(base_squares):
            b_sym = B[0][j]
            pair_masks[idx][b_sym] |= (1 << s)
            
    def backtrack(r, c):
        if time.time() > deadline:
            return False
        if r == n:
            return True
        if c == n:
            return backtrack(r + 1, 0)
            
        # Determine candidates for grid[r][c]
        # Must not be in row_masks[r]
        # Must not be in col_masks[c]
        # Must not be in pair_masks[idx][base[idx][r][c]] for all idx
        
        # Start with all symbols
        candidates = (1 << n) - 1
        candidates &= ~row_masks[r]
        candidates &= ~col_masks[c]
        
        for idx, B in enumerate(base_squares):
            b_sym = B[r][c]
            candidates &= ~pair_masks[idx][b_sym]
            
        # Iterate through set bits in candidates
        while candidates:
            # Get lowest set bit
            bit = candidates & -candidates
            candidates ^= bit
            s = bit.bit_length() - 1
            
            # Place s
            grid[r][c] = s
            row_masks[r] |= bit
            col_masks[c] |= bit
            for idx, B in enumerate(base_squares):
                b_sym = B[r][c]
                pair_masks[idx][b_sym] |= bit
            
            if backtrack(r, c + 1):
                return True
                
            # Undo
            grid[r][c] = -1
            row_masks[r] &= ~bit
            col_masks[c] &= ~bit
            for idx, B in enumerate(base_squares):
                b_sym = B[r][c]
                pair_masks[idx][b_sym] &= ~bit
                
        return False

    if backtrack(1, 0):
        # Convert to list of lists of ints
        return [[int(grid[r][c]) for c in range(n)] for r in range(n)]
    return None

def construct():
    start = time.time()
    deadline = start + 55.0  # Safety margin for 60s limit
    
    # Base square: addition table mod 10
    base = [[(i + j) % N for j in range(N)] for i in range(N)]
    squares = [base]
    
    # Find the 2nd square
    # Give it most of the time
    deadline_2 = start + 20.0
    nxt = _build_square(squares, deadline_2)
    if nxt is not None:
        # Verify strictly before adding
        if _validate(squares + [nxt]):
            squares.append(nxt)
        else:
            # Should not happen if _build_square is correct, but safety check
            pass
            
    # Attempt to find a 3rd square?
    # Mathematically impossible for n=10.
    # But if we have time, we could try, but it will fail.
    # Better to just return 2.
    
    # Final safety validation
    if _validate(squares):
        out = []
        for L in squares:
            out.append([[int(L[r][c]) for c in range(N)] for r in range(N)])
        return out

    # Fallback: single valid square
    return [[int((i + j) % N) for j in range(N)] for i in range(N)]
