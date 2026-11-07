"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'n': 10}

def _make_square(a, c):
    """Generate Latin square L[i][j] = (a*i + c*j) % 10.
    Valid if gcd(a, 10) == 1 and gcd(c, 10) == 1."""
    n = P['n']
    if math.gcd(a, n) != 1 or math.gcd(c, n) != 1:
        return None
    return [[int((a * i + c * j) % n) for j in range(n)] for i in range(n)]

def _are_orthogonal(sq1, sq2):
    """Check if two Latin squares are orthogonal."""
    n = P['n']
    pairs = set()
    for i in range(n):
        for j in range(n):
            pair = (sq1[i][j], sq2[i][j])
            if pair in pairs:
                return False
            pairs.add(pair)
    return len(pairs) == n * n

def _is_latin(sq):
    """Verify sq is a valid Latin square."""
    n = P['n']
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def construct():
    n = P['n']
    deadline = time.time() + 50  # safety margin before 60s limit
    
    # Generate candidate squares using linear construction
    # L_{a,c}[i][j] = (a*i + c*j) % 10
    # We search over a, c in {1, 3, 7, 9}
    candidates = []
    for a in [1, 3, 7, 9]:
        for c in [1, 3, 7, 9]:
            sq = _make_square(a, c)
            if sq is not None:
                candidates.append((a, c, sq))
    
    # Greedily select maximum pairwise orthogonal set
    selected = []
    for a, c, sq in candidates:
        if time.time() > deadline:
            break
        # Check against all already selected
        compatible = True
        for a2, c2, sel_sq in selected:
            # Quick check: determinant must be coprime to n
            det = a * c2 - a2 * c
            if math.gcd(det, n) != 1:
                compatible = False
                break
            if not _are_orthogonal(sq, sel_sq):
                compatible = False
                break
        if compatible:
            selected.append((a, c, sq))
    
    # Ensure at least 1 square
    if not selected:
        # Fallback to a known valid square
        sq = _make_square(1, 1)
        if sq is None:
            sq = [[0]*n for _ in range(n)] # Should not happen
        selected = [(1, 1, sq)]
    
    # Extract just the squares
    result = [sq for _, _, sq in selected]
    
    # Final validation
    for sq in result:
        assert _is_latin(sq), "Invalid Latin square"
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert _are_orthogonal(result[i], result[j]), "Squares not orthogonal"
    
    # Ensure all values are Python ints
    final = []
    for sq in result:
        final.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
    
    return final
# EVOLVE-BLOCK-END
