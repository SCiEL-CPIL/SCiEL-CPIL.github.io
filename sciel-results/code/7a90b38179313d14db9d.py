import time
import sys

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Build a 2x2 base square and 5x5 MOLS
    L2_2x2 = [[0, 1], [1, 0]]
    
    # 5x5 MOLS - we can construct 4 mutually orthogonal Latin squares of order 5
    # Using the finite field GF(5)
    def _make_mols_order_5():
        m = 5
        squares = []
        # For each multiplier a in 1..4, create a square: (a*i + j) % m
        for a in range(1, m):
            sq = [[(a * i + j) % m for j in range(m)] for i in range(m)]
            squares.append(sq)
        return squares
    
    mols_5 = _make_mols_order_5()
    
    # Construct 10x10 squares using Kronecker product approach
    # For each 5x5 square, create two 10x10 squares (one for each row of 2x2)
    squares_10 = []
    
    for idx, sq5 in enumerate(mols_5):
        for row2 in range(2):
            sq10 = [[0]*10 for _ in range(10)]
            for i in range(10):
                for j in range(10):
                    # Decompose i, j into (i2, i5), (j2, j5)
                    i2, i5 = i // 5, i % 5
                    j2, j5 = j // 5, j % 5
                    # The value combines the 2x2 square and 5x5 square
                    # Using: value = 2 * L2[i2][j2] + sq5[i5][j5]
                    val = 2 * L2_2x2[i2][j2] + sq5[i5][j5]
                    sq10[i][j] = val
            squares_10.append(sq10)
    
    # Verify all squares are valid Latin and mutually orthogonal
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
        return len(pairs) == n*n
    
    # Filter valid squares
    valid_squares = []
    for sq in squares_10:
        if not _is_valid_latin(sq, 10):
            continue
        # Check orthogonality with all previously added squares
        orthogonal = True
        for prev_sq in valid_squares:
            if not _is_orthogonal(prev_sq, sq, 10):
                orthogonal = False
                break
        if orthogonal:
            valid_squares.append(sq)
    
    if len(valid_squares) > 0:
        return valid_squares
    
    # Fallback: use the original approach if Kronecker product fails
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    L2 = _find_orthogonal(L1, n)
    
    if L2 is not None:
        if _is_valid_latin(L2, n) and _is_orthogonal(L1, L2, n):
            return [L1, L2]
    return [L1]

def _find_orthogonal(L1, n):
    # Backtracking search for L2
    # Constraints:
    # 1. L2 is Latin (rows and cols are permutations)
    # 2. L2 is orthogonal to L1 (on each broken diagonal of L1, L2 is a permutation)
    
    L2 = [[-1]*n for _ in range(n)]
    
    # Precompute diagonal index for each cell
    diag_of = [[L1[i][j] for j in range(n)] for i in range(n)]
    
    # Bitmasks for used symbols in each row, col, and diagonal
    row_masks = [0]*n
    col_masks = [0]*n
    diag_masks = [0]*n # indexed by L1[i][j]
    
    # Count available symbols for each cell (for MRV heuristic)
    # We'll recompute dynamically
    
    start_time = time.time()
    deadline = start_time + 40.0  # Safety margin
    
    # Precompute list of all cells
    all_cells = [(i, j) for i in range(n) for j in range(n)]
    
    def get_available_count(i, j):
        """Count available symbols for cell (i,j)"""
        d = diag_of[i][j]
        mask = row_masks[i] | col_masks[j] | diag_masks[d]
        # Count bits not set in mask
        avail = 0
        for s in range(n):
            if not (mask & (1 << s)):
                avail += 1
        return avail
    
    def backtrack(remaining_count):
        if time.time() > deadline:
            return False
        
        if remaining_count == 0:
            return True
        
        # Find most constrained cell (MRV heuristic)
        best_i, best_j = -1, -1
        best_count = n + 1
        
        for i in range(n):
            for j in range(n):
                if L2[i][j] == -1:
                    cnt = get_available_count(i, j)
                    if cnt < best_count:
                        best_count = cnt
                        best_i, best_j = i, j
                        if cnt <= 1:
                            break
            if best_count <= 1:
                break
        
        if best_i == -1:
            return True
        
        i, j = best_i, best_j
        d = diag_of[i][j]
        
        # Try all symbols 0..n-1
        for sym in range(n):
            bit = 1 << sym
            # Check constraints
            if not (row_masks[i] & bit) and not (col_masks[j] & bit) and not (diag_masks[d] & bit):
                # Place
                L2[i][j] = sym
                row_masks[i] |= bit
                col_masks[j] |= bit
                diag_masks[d] |= bit
                
                if backtrack(remaining_count - 1):
                    return True
                    
                # Undo
                L2[i][j] = -1
                row_masks[i] &= ~bit
                col_masks[j] &= ~bit
                diag_masks[d] &= ~bit
                
        return False

    if backtrack(n*n):
        return L2
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
    return len(pairs) == n*n
