"""Matrix multiplication 2x4x5 - Strassen hybrid decomposition, rank 36."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    timeout_limit = 110.0

    total_rank = 36
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    cursor = 0

    def emit(ua, va, wa):
        nonlocal cursor
        if cursor >= total_rank:
            raise RuntimeError("rank overflow")
        for k, v in ua.items():
            U[k][cursor] = v
        for k, v in va.items():
            V[k][cursor] = v
        for k, v in wa.items():
            W[k][cursor] = v
        cursor += 1

    def do_strassen(ai0, ai1, aj0, aj1, bi0, bi1, bj0, bj1, ci0, ci1, cj0, cj1):
        def A(l):
            r, c = divmod(l, 2)
            return (ai0 + r) * m + (aj0 + c)
        def B(l):
            r, c = divmod(l, 2)
            return (bi0 + r) * p + (bj0 + c)
        def C(l):
            r, c = divmod(l, 2)
            return (cj0 + c) * n + (ci0 + r)

        emit({A(0): 1, A(3): 1}, {B(0): 1, B(3): 1}, {C(0): 1, C(3): 1})
        emit({A(2): 1, A(3): 1}, {B(0): 1}, {C(2): 1, C(3): -1})
        emit({A(0): 1}, {B(1): 1, B(3): -1}, {C(1): 1, C(3): 1})
        emit({A(3): 1}, {B(2): 1, B(0): -1}, {C(0): 1, C(2): 1})
        emit({A(0): 1, A(1): 1}, {B(3): 1}, {C(0): -1, C(1): 1})
        emit({A(2): 1, A(0): -1}, {B(0): 1, B(1): 1}, {C(3): 1})
        emit({A(1): 1, A(3): -1}, {B(2): 1, B(3): 1}, {C(0): 1})

    def do_dot2(ai0, aj0, bi, bj, ci, cj):
        def A(l):
            return (ai0 + l) * m + (aj0 + l)
        def B(l):
            return (bi + l) * p + bj
        def C(l):
            return cj * n + (ci + l)

        emit({A(0): 1}, {B(0): 1}, {C(0): 1})
        emit({A(0): 1}, {B(1): 1}, {C(0): 1})
        emit({A(1): 1}, {B(0): 1}, {C(1): 1})
        emit({A(1): 1}, {B(1): 1}, {C(1): 1})

    # Four 2x2x2 Strassen blocks for B columns 0-3
    do_strassen(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    do_strassen(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3)
    do_strassen(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1)
    do_strassen(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3)

    # Two 2x2x1 schoolbook blocks for B column 4
    do_dot2(0, 0, 0, 4, 0, 4)
    do_dot2(0, 2, 2, 4, 0, 4)

    if time.time() - t0 > timeout_limit:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
