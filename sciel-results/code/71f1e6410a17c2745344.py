"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_latin(sq):
    """Check if sq is a valid Latin square of order n."""
    n = len(sq)
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _is_orthogonal(sq1, sq2):
    """Check if sq1 and sq2 are orthogonal Latin squares."""
    n = len(sq1)
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (sq1[i][j], sq2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True

def _find_orthogonal_square(base_sq, deadline):
    """
    Finds a Latin square orthogonal to base_sq using backtracking with constraint propagation.
    Returns the orthogonal square or None if not found before deadline.
    """
    n = len(base_sq)
    sq = [[-1 for _ in range(n)] for _ in range(n)]
    
    # Constraints:
    # 1. Each row must contain 0..n-1
    # 2. Each col must contain 0..n-1
    # 3. Each pair (base_sq[i][j], sq[i][j]) must be unique
    
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    pairs_used = set()
    
    # Pre-compute base values for quick access
    base = base_sq
    
    def solve(idx):
        if time.time() > deadline:
            return False
        if idx == n * n:
            return True
            
        i = idx // n
        j = idx % n
        
        base_val = base[i][j]
        
        # Candidates: 0..n-1
        # Filter by row, col, and pair constraints
        # To improve performance, we can iterate through possible values
        for v in range(n):
            if v in row_used[i]:
                continue
            if v in col_used[j]:
                continue
            pair = (base_val, v)
            if pair in pairs_used:
                continue
            
            # Place v
            sq[i][j] = v
            row_used[i].add(v)
            col_used[j].add(v)
            pairs_used.add(pair)
            
            if solve(idx + 1):
                return True
            
            # Backtrack
            sq[i][j] = -1
            row_used[i].discard(v)
            col_used[j].discard(v)
            pairs_used.discard(pair)
            
        return False

    # Start solving
    if solve(0):
        return sq
    return None

def _construct_mols_10():
    """
    Construct the maximum set of MOLS of order 10.
    It is known that N(10) = 2, so we can have at most 3 mutually orthogonal squares.
    """
    n = 10
    # Base square: cyclic group Z_10
    # L0[i][j] = (i + j) % 10
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [base]
    
    # Deadline: 90s limit, leave 10s margin for safety
    deadline = time.time() + 80
    
    # Find second square orthogonal to base
    # This should be very fast
    sq2 = _find_orthogonal_square(base, deadline)
    if sq2 is not None:
        squares.append(sq2)
        
    # Find third square orthogonal to base AND sq2
    # N(10)=2 implies this is impossible, but we attempt it with a short timeout
    # just in case the theoretical bound is being tested against a specific definition
    # or if there's a misunderstanding. However, standard theory says max 3 squares total.
    # Actually, if N(10)=2, we have L0, L1, L2. L0 and L1 are orthogonal. L0 and L2 are orthogonal. L1 and L2 are orthogonal.
    # The "number of MOLS" usually refers to the set of squares orthogonal to the identity.
    # If we return [L0, L1, L2], the evaluator checks pairwise orthogonality.
    # Since N(10)=2, there is no L3 orthogonal to L0 and L1.
    # So we stop at 2 additional squares? No, max 2 MOLS means max 2 squares orthogonal to identity.
    # So total 3 squares.
    
    # Let's try to find a 3rd square (index 2) orthogonal to L0 and L1.
    if len(squares) == 2:
        # We need a square orthogonal to L0 and L1.
        # We can use a similar backtracking but with two base squares.
        # Given N(10)=2, this will fail. We set a short deadline to avoid wasting time.
        short_deadline = time.time() + 5
        sq3 = _find_orthogonal_to_two(squares, short_deadline)
        if sq3 is not None:
            squares.append(sq3)
            
    return squares

def _find_orthogonal_to_two(existing_squares, deadline):
    """
    Finds a Latin square orthogonal to all squares in existing_squares.
    """
    n = len(existing_squares[0])
    sq = [[-1 for _ in range(n)] for _ in range(n)]
    
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    # pairs_used[k] is a set of (val_k, val_new)
    pairs_used = [set() for _ in range(len(existing_squares))]
    
    bases = existing_squares
    
    def solve(idx):
        if time.time() > deadline:
            return False
        if idx == n * n:
            return True
            
        i = idx // n
        j = idx % n
        
        # Check constraints against all existing squares
        for v in range(n):
            if v in row_used[i]:
                continue
            if v in col_used[j]:
                continue
            
            valid = True
            new_pairs = []
            for k, b_sq in enumerate(bases):
                b_val = b_sq[i][j]
                p = (b_val, v)
                if p in pairs_used[k]:
                    valid = False
                    break
                new_pairs.append((k, p))
            
            if not valid:
                continue
            
            # Place v
            sq[i][j] = v
            row_used[i].add(v)
            col_used[j].add(v)
            for k, p in new_pairs:
                pairs_used[k].add(p)
            
            if solve(idx + 1):
                return True
            
            # Backtrack
            sq[i][j] = -1
            row_used[i].discard(v)
            col_used[j].discard(v)
            for k, p in new_pairs:
                pairs_used[k].discard(p)
            
        return False

    if solve(0):
        return sq
    return None

_CACHE = None

def construct():
    global _CACHE
    if _CACHE is None:
        _CACHE = _construct_mols_10()
    # Ensure types are correct: list of lists of Python ints
    result = []
    for sq in _CACHE:
        result.append([[int(v) for v in row] for row in sq])
    return result
# EVOLVE-BLOCK-END
