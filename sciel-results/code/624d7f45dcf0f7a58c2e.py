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
    if not base_squares:
        return None
        
    row_masks = [0] * n
    col_masks = [0] * n
    pair_masks = [[0] * n for _ in range(len(base_squares))]
    
    grid = [[-1] * n for _ in range(n)]
    
    for j in range(n):
        s = j
        grid[0][j] = s
        row_masks[0] |= (1 << s)
        col_masks[j] |= (1 << s)
        for idx, B in enumerate(base_squares):
            b_sym = B[0][j]
            pair_masks[idx][b_sym] |= (1 << s)
            
    steps = 0
    def backtrack(r, c):
        nonlocal steps
        steps += 1
        if steps % 1000 == 0 and time.time() > deadline:
            return False
        if r == n:
            return True
        if c == n:
            return backtrack(r + 1, 0)
            
        candidates = (1 << n) - 1
        candidates &= ~row_masks[r]
        candidates &= ~col_masks[c]
        
        for idx, B in enumerate(base_squares):
            b_sym = B[r][c]
            candidates &= ~pair_masks[idx][b_sym]
            
        while candidates:
            bit = candidates & -candidates
            candidates ^= bit
            s = bit.bit_length() - 1
            
            grid[r][c] = s
            row_masks[r] |= bit
            col_masks[c] |= bit
            for idx, B in enumerate(base_squares):
                b_sym = B[r][c]
                pair_masks[idx][b_sym] |= bit
            
            if backtrack(r, c + 1):
                return True
                
            grid[r][c] = -1
            row_masks[r] &= ~bit
            col_masks[c] &= ~bit
            for idx, B in enumerate(base_squares):
                b_sym = B[r][c]
                pair_masks[idx][b_sym] &= ~bit
                
        return False

    if backtrack(1, 0):
        return [[int(grid[r][c]) for c in range(n)] for r in range(n)]
    return None

def construct():
    start = time.time()
    deadline = start + 55.0
    
    base = [[(i + j) % N for j in range(N)] for i in range(N)]
    squares = [base]
    
    deadline_2 = start + 10.0
    nxt = _build_square(squares, deadline_2)
    if nxt is not None:
        if _validate(squares + [nxt]):
            squares.append(nxt)
            
    if _validate(squares):
        out = []
        for L in squares:
            out.append([[int(L[r][c]) for c in range(N)] for r in range(N)])
        return out

    return [[int((i + j) % N) for j in range(N)] for i in range(N)]
