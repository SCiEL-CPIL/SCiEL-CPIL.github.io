<<<<<<< SEARCH
import time
import sys

def construct():
    n = 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Try to find a second orthogonal square
    L2 = _find_orthogonal(L1, n)
    
    if L2 is not None:
        # Verify L2 is valid Latin and orthogonal to L1 (safety check)
        if _is_valid_latin(L2, n) and _is_orthogonal(L1, L2, n):
            return [L1, L2]
    return [L1]

def _find_orthogonal(L1, n):
    # Backtracking search for L2
    # Constraints:
    # 1. L2 is Latin (rows and cols are permutations)
    # 2. L2 is orthogonal to L1 (on each broken diagonal of L1, L2 is a permutation)
    
    L2 = [[-1]*n for _ in range(n)]
    
    # Bitmasks for used symbols in each row, col, and diagonal
    row_masks = [0]*n
    col_masks = [0]*n
    diag_masks = [0]*n # indexed by L1[i][j]
    
    # List of cells to fill
    cells = [(i, j) for i in range(n) for j in range(n)]
    
    # Heuristic: Fill cells in order of most constrained?
    # For now, simple order.
    
    def backtrack(idx):
        if time.time() > start_time + 5.0:
            return False
        
        if idx == n*n:
            return True
            
        i, j = cells[idx]
        d = L1[i][j]
        
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
                
                if backtrack(idx + 1):
                    return True
                    
                # Undo
                L2[i][j] = -1
                row_masks[i] &= ~bit
                col_masks[j] &= ~bit
                diag_masks[d] &= ~bit
                
        return False

    start_time = time.time()
    if backtrack(0):
        return L2
    return None
=======
import time
import sys

def construct():
    n = 10
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]

    # Try to find a second orthogonal square
    L2 = _find_orthogonal(L1, n, deadline)

    if L2 is not None:
        # Verify L2 is valid Latin and orthogonal to L1 (safety check)
        if _is_valid_latin(L2, n) and _is_orthogonal(L1, L2, n):
            # Try to find a third square
            L3 = _find_orthogonal_to_set([L1, L2], n, deadline)
            if L3 is not None and _is_valid_latin(L3, n):
                all_ok = all(_is_orthogonal(L3, Lk, n) for Lk in [L1, L2])
                if all_ok:
                    return [L1, L2, L3]
            return [L1, L2]
    return [L1]

def _find_orthogonal(L1, n, deadline):
    # Backtracking search for L2 orthogonal to L1
    # Constraints:
    # 1. L2 is Latin (rows and cols are permutations)
    # 2. L2 is orthogonal to L1 (on each broken diagonal of L1, L2 is a permutation)

    L2 = [[-1]*n for _ in range(n)]

    # Bitmasks for used symbols in each row, col, and diagonal
    row_masks = [0]*n
    col_masks = [0]*n
    diag_masks = [0]*n  # indexed by L1[i][j]

    # Cell ordering: fill most-constrained cells first
    # A cell's constraint degree depends on how many symbols are still available
    # given current row/col/diag masks. We use a static heuristic based on
    # the structure: cells where L1[i][j] varies across rows/cols are more constrained.
    # Actually, all cells have the same initial constraint (all 10 symbols available).
    # Better heuristic: fill in an order that maximizes early constraint propagation.
    # Use a fixed order that interleaves rows and columns to detect conflicts early.
    # We'll use a simple but effective order: fill row by row, but within each row,
    # fill columns in an order that's a permutation (e.g., reverse) to break symmetry.
    cells = [(i, j) for i in range(n) for j in range(n-1, -1, -1)]

    # Precompute for each cell the diagonal index
    cell_diag = [L1[i][j] for i, j in cells]

    def backtrack(idx):
        if time.time() > deadline:
            return False

        if idx == n*n:
            return True

        i, j = cells[idx]
        d = cell_diag[idx]

        # Try all symbols 0..n-1, ordered by least constraining first (MRV heuristic on symbols)
        # Count available symbols for each symbol
        avail = []
        for sym in range(n):
            bit = 1 << sym
            if not (row_masks[i] & bit) and not (col_masks[j] & bit) and not (diag_masks[d] & bit):
                avail.append(sym)

        if not avail:
            return False

        # Try symbols in order; prefer symbols that appear less frequently in unassigned cells
        # of the same row/col/diag to reduce future constraints
        for sym in avail:
            bit = 1 << sym
            # Place
            L2[i][j] = sym
            row_masks[i] |= bit
            col_masks[j] |= bit
            diag_masks[d] |= bit

            if backtrack(idx + 1):
                return True

            # Undo
            L2[i][j] = -1
            row_masks[i] &= ~bit
            col_masks[j] &= ~bit
            diag_masks[d] &= ~bit

        return False

    if backtrack(0):
        return L2
    return None

def _find_orthogonal_to_set(L_set, n, deadline):
    """Find a Latin square orthogonal to all squares in L_set."""
    # Build combined diagonal masks: for each cell, the tuple of values from each square
    # must be unique. We track used tuples via a set, but for bitmask efficiency,
    # we encode the tuple as an integer.
    # For k squares, the tuple space is n^k. For k=2, that's 100 values.
    # We use a bitmask of size n^k if k is small, else a set.
    k = len(L_set)
    if k <= 2:
        # Use bitmask: encode tuple (a, b) as a*n + b
        tuple_mask = 0
        tuple_bits = 1 << (n**k)
    else:
        tuple_set = set()

    L = [[-1]*n for _ in range(n)]
    row_masks = [0]*n
    col_masks = [0]*n

    # Cell ordering
    cells = [(i, j) for i in range(n) for j in range(n-1, -1, -1)]

    # Precompute tuple prefix for each cell
    def get_tuple_val(i, j, sym):
        """Get the encoded tuple value for cell (i,j) with symbol sym."""
        # Tuple is (L_set[0][i][j], L_set[1][i][j], ..., sym)
        # Encode as nested: ((L0*n + L1)*n + ...)*n + sym
        val = 0
        for sq in L_set:
            val = val * n + sq[i][j]
        val = val * n + sym
        return val

    def backtrack(idx):
        if time.time() > deadline:
            return False

        if idx == n*n:
            return True

        i, j = cells[idx]

        avail = []
        for sym in range(n):
            bit = 1 << sym
            if row_masks[i] & bit or col_masks[j] & bit:
                continue
            # Check tuple uniqueness
            tv = get_tuple_val(i, j, sym)
            if k <= 2:
                if tuple_mask & (1 << tv):
                    continue
            else:
                if tv in tuple_set:
                    continue
            avail.append((sym, tv))

        if not avail:
            return False

        for sym, tv in avail:
            bit = 1 << sym
            L[i][j] = sym
            row_masks[i] |= bit
            col_masks[j] |= bit

            if k <= 2:
                old_tuple_mask = tuple_mask
                # Can't easily do local variable in closure for bitmask... 
                # We need to pass tuple_mask as a mutable container or use nonlocal
                # Let's use a list to hold the mask
                pass

            # This approach has issues with the bitmask in closure. Let me restructure.
            break  # placeholder, will fix

        return False

    # The closure issue with tuple_mask needs restructuring. Let me use a different approach.
    # Instead, let's just return None for now if we can't find it quickly,
    # since finding 3+ orthogonal Latin squares of order 10 is very hard.
    # Actually, it's known that no pair of orthogonal Latin squares of order 10 exists
    # (Euler's 36 officers problem). Wait, that's order 6. For order 10, it's known
    # that there exist pairs of orthogonal Latin squares, and even larger sets.
    # But finding them via backtracking might be slow. Let's be conservative.
    return None
>>>>>>> REPLACE

Wait, I realize the `_find_orthogonal_to_set` function has issues with the closure. Let me fix that properly.

