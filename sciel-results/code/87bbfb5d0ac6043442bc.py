"""Deterministic MOLS of order 10 via Kronecker (tensor) product of 2x2 and 5x5 MOLS."""
# EVOLVE-BLOCK-START

def _is_latin(sq):
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
    n = len(sq1)
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (sq1[i][j], sq2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True


def _construct_mols_10():
    """Construct MOLS of order 10 by tensor product of order-2 and order-5 MOLS.

    10 = 2 * 5.  Order 2 has 1 MOLS; order 5 (prime) has 4 MOLS.
    The tensor product of mutually orthogonal sets yields a mutually
    orthogonal set of size (1-1+1)*(4-1+1) = 4 squares of order 10.
    """
    # Order 2 Latin square: L2[a][b] = (a + b) % 2
    L2 = [[(a + b) % 2 for b in range(2)] for a in range(2)]

    # Order 5 MOLS: L5_k[a][b] = (a + k*b) % 5 for k = 0..3
    L5 = [[[(a + k * b) % 5 for b in range(5)] for a in range(5)] for k in range(4)]

    squares = []
    for k in range(4):
        sq = [[5 * L2[i // 5][j // 5] + L5[k][i % 5][j % 5]
               for j in range(10)]
              for i in range(10)]
        squares.append(sq)

    # Verify validity (fast, deterministic).
    for sq in squares:
        assert _is_latin(sq), "Constructed square is not Latin"
    for a in range(len(squares)):
        for b in range(a + 1, len(squares)):
            assert _is_orthogonal(squares[a], squares[b]), \
                "Constructed squares %d,%d are not orthogonal" % (a, b)

    return squares


_CACHE = None


def construct():
    global _CACHE
    if _CACHE is None:
        _CACHE = _construct_mols_10()
    # Ensure types are exactly Python int in list-of-lists form.
    result = []
    for sq in _CACHE:
        result.append([[int(v) for v in row] for row in sq])
    return result
# EVOLVE-BLOCK-END
