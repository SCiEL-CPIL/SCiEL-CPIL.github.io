# EVOLVE-BLOCK-START
import time
import random

def _find_orthogonal_square_mr(base_sq, deadline):
    """
    Finds a Latin square orthogonal to base_sq using MRV backtracking.
    """
    n = len(base_sq)
    sq = [[0] * n for _ in range(n)]
    
    # Masks for row, col, pair
    row_mask = [0] * n
    col_mask = [0] * n
    pair_mask = [0] * (n * n)
    
    # To implement MRV, we need to track available values for each cell.
    # This is expensive to update. 
    # Alternative: just randomize the order of values and cells?
    # For n=10, simple backtracking with randomization might work if we restart.
    
    def solve(idx, depth):
        if time.time() > deadline:
            return False
        if idx == n * n:
            return True
            
        i = idx // n
        j = idx % n
        base_val = base_sq[i][j]
        
        # Randomize values to avoid bias
        vals = list(range(n))
        random.shuffle(vals)
        
        for v in vals:
            bit = 1 << v
            if row_mask[i] & bit:
                continue
            if col_mask[j] & bit:
                continue
            pair_bit = 1 << (base_val * n + v)
            if pair_mask[0] & pair_bit:
                continue
            
            sq[i][j] = v
            row_mask[i] |= bit
            col_mask[j] |= bit
            pair_mask[0] |= pair_bit
            
            if solve(idx + 1, depth + 1):
                return True
            
            sq[i][j] = 0
            row_mask[i] &= ~bit
            col_mask[j] &= ~bit
            pair_mask[0] &= ~pair_bit
            
        return False

    # Try multiple times with different random seeds/orders
    for attempt in range(50):
        if time.time() > deadline:
            break
        # Reset state
        sq = [[0] * n for _ in range(n)]
        row_mask = [0] * n
        col_mask = [0] * n
        pair_mask = [0] * (n * n)
        
        if solve(0, 0):
            return sq
            
    return None

def _construct_mols_10():
    n = 10
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [base]
    
    deadline = time.time() + 45  # Leave margin for overhead
    
    sq2 = _find_orthogonal_square_mr(base, deadline)
    if sq2 is not None:
        squares.append(sq2)
    
    return squares

_CACHE = None

def construct():
    global _CACHE
    if _CACHE is None:
        _CACHE = _construct_mols_10()
    result = []
    for sq in _CACHE:
        result.append([[int(v) for v in row] for row in sq])
    return result
# EVOLVE-BLOCK-END
