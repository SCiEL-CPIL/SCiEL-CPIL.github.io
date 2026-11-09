<<<<<<< SEARCH
def _verify_exact(U, V, W, n, m, p):
    """Verify with exact rational arithmetic."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if total != 1:
                    return False
    return True
=======
def _verify_exact(U, V, W, n, m, p):
    """Verify with exact rational arithmetic."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if total != 1:
                    return False
    return True
>>>>>>> REPLACE
