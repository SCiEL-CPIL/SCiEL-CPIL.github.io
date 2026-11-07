"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a list of mutually orthogonal Latin squares of order 10.
    Uses the product construction with Z_2 x Z_5 to build orthogonal squares.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 55.0

    # CRT mapping: (x in Z_2, y in Z_5) -> z in Z_10
    # z = (5*x + 6*y) % 10
    def crt(x, y):
        return (5 * x + 6 * y) % 10

    # Inverse: z in Z_10 -> (x in Z_2, y in Z_5)
    def inv_crt(z):
        return (z % 2, z % 5)

    # Build square A: A[r][c] = (r_p + c_p, r_q + c_q)
    square_a = [[0] * n for _ in range(n)]
    for r in range(n):
        rp, rq = inv_crt(r)
        for c in range(n):
            cp, cq = inv_crt(c)
            val = crt((rp + cp) % 2, (rq + cq) % 5)
            square_a[r][c] = val

    # Build square B: B[r][c] = (r_p + 2*c_p, r_q + 2*c_q)
    square_b = [[0] * n for _ in range(n)]
    for r in range(n):
        rp, rq = inv_crt(r)
        for c in range(n):
            cp, cq = inv_crt(c)
            val = crt((rp + 2 * cp) % 2, (rq + 2 * cq) % 5)
            square_b[r][c] = val

    # Verify both are Latin squares
    def is_latin(sq):
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
            col = [sq[r][i] for r in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True

    # Verify orthogonality
    def are_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pairs.add((sq1[i][j], sq2[i][j]))
        return len(pairs) == n * n

    if not is_latin(square_a) or not is_latin(square_b):
        # Fallback to single square
        square = [[(i + j) % n for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                square[i][j] = int(square[i][j])
        return [square]

    if not are_orthogonal(square_a, square_b):
        # Fallback to single square
        square = [[(i + j) % n for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                square[i][j] = int(square[i][j])
        return [square]

    # Enforce strict Python int types
    for sq in [square_a, square_b]:
        for i in range(n):
            for j in range(n):
                if not isinstance(sq[i][j], int) or isinstance(sq[i][j], bool):
                    sq[i][j] = int(sq[i][j])

    return [square_a, square_b]

# EVOLVE-BLOCK-END
