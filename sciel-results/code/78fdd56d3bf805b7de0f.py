import time

def construct():
    n = 10
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    # Start with the cyclic Latin square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [L1]
    
    # Try to find additional orthogonal squares
    while time.time() < deadline:
        L_next = _find_orthogonal(squares, n, deadline)
        if L_next is not None:
            # Verify validity
            if _is_valid_latin(L_next, n) and all(_is_orthogonal(s, L_next, n) for s in squares):
                squares.append(L_next)
            else:
                break
        else:
            break
    
    # Ensure we return at least 1 square
    if not squares:
        squares = [L1]
    
    # Convert to ensure pure Python types
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
    
    return result

def _find_orthogonal(existing_squares, n, deadline):
    """Find a Latin square orthogonal to all existing squares."""
    
    L = [[-1] * n for _ in range(n)]
    
    # Bitmasks for used symbols in each row, col, and for each existing square's diagonal
    row_masks = [0] * n
    col_masks = [0] * n
    # For each existing square, track which symbols are used on each "diagonal"
    # diagonal indexed by the symbol in that square
    diag_masks = [[0] * n for _ in existing_squares]
    
    # Precompute which cell has which symbol in each existing square
    # For square k, diag_masks[k][symbol] tracks which cells with that symbol are filled
    
    cells = [(i, j) for i in range(n) for j in range(n)]
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if idx == n * n:
            return True
            
        i, j = cells[idx]
        
        # Try all symbols 0..n-1
        for sym in range(n):
            bit = 1 << sym
            # Check row constraint
            if row_masks[i] & bit:
                continue
            # Check col constraint
            if col_masks[j] & bit:
                continue
            # Check orthogonality constraints with all existing squares
            valid = True
            for k in range(len(existing_squares)):
                d = existing_squares[k][i][j]
                if diag_masks[k][d] & bit:
                    valid = False
                    break
            if not valid:
                continue
            
            # Place
            L[i][j] = sym
            row_masks[i] |= bit
            col_masks[j] |= bit
            for k in range(len(existing_squares)):
                d = existing_squares[k][i][j]
                diag_masks[k][d] |= bit
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            L[i][j] = -1
            row_masks[i] &= ~bit
            col_masks[j] &= ~bit
            for k in range(len(existing_squares)):
                d = existing_squares[k][i][j]
                diag_masks[k][d] &= ~bit
                
        return False
    
    if backtrack(0):
        return L
    return None

def _is_valid_latin(L, n):
    for i in range(n):
        if sorted(L[i]) != list(range(n)):
            return False
        col = [L[r][i] for r in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _is_orthogonal(L1, L2, n):
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (L1[i][j], L2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return len(pairs) == n * n
