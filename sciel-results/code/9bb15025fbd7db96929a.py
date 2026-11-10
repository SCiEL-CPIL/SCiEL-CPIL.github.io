import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    DEADLINE = 600
    p = 7
    q = 49
    k = 3

    add_tbl = np.zeros((q, q), dtype=np.int64)
    sub_tbl = np.zeros((q, q), dtype=np.int64)
    mul_tbl = np.zeros((q, q), dtype=np.int64)
    inv_tbl = np.zeros(q, dtype=np.int64)

    for i in range(q):
        a0, a1 = i // p, i % p
        for j in range(q):
            b0, b1 = j // p, j % p
            add_tbl[i, j] = ((a0 + b0) % p) * p + ((a1 + b1) % p)
            sub_tbl[i, j] = ((a0 - b0) % p) * p + ((a1 - b1) % p)
            mul_tbl[i, j] = ((a0 * b0 + a1 * b1 * k) % p) * p + ((a0 * b1 + a1 * b0) % p)
        if i != 0:
            denom = (a0 * a0 - a1 * a1 * k) % p
            if denom != 0:
                d_inv = pow(denom, p - 2, p)
                inv_tbl[i] = (a0 * d_inv % p) * p + ((-a1) % p * d_inv % p)

    total_points = q * q
    all_points_set = set(range(total_points))

    # Hermitian unital
    U = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        U.add((xa * p + xb) * q + (ya * p + yb))

    # Build lines
    lines = []
    for c in range(q):
        lines.append([c * q + y for y in range(q)])
    for m in range(q):
        for b in range(q):
            lines.append([x * q + add_tbl[mul_tbl[m, x], b] for x in range(q)])

    # For each line, intersection with U
    line_U = []
    for L in lines:
        I = set(pt for pt in L if pt in U)
        line_U.append(I)

    # Greedy: find largest H subset of U such that S = U \ H is valid.
    # Equivalently, minimize |H| subject to Nikodym property.
    # Use greedy set cover style: start with H = empty, S = U, check validity.
    # If invalid, add points to H to fix.
    
    # Actually, the current program does this well. Let me replicate and improve.
    H = set()
    uncovered = set(all_points_set - U)  # points not in U need to be covered
    
    # For each line, the points it covers outside U
    line_cov = []
    for L in lines:
        cov = set(pt for pt in L if pt not in U)
        line_cov.append(cov)

    for _ in range(10000):
        if time.time() - start_time > DEADLINE - 30:
            break
        if not uncovered:
            break
        best_i = -1
        best_score = -1
        for i in range(len(lines)):
            if not line_cov[i]:
                continue
            new_cov = sum(1 for pt in line_cov[i] if pt in uncovered)
            if new_cov == 0:
                continue
            new_H = sum(1 for pt in line_U[i] if pt not in H)
            score = new_cov / (new_H + 1)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i == -1:
            break
        H.update(line_U[best_i])
        for pt in lines[best_i]:
            if pt in uncovered:
                uncovered.remove(pt)

    S = U - H
    N = all_points_set - S

    points = []
    for pt in N:
        x, y = pt // q, pt % q
        points.append([[x // p, x % p], [y // p, y % p]])

    return np.array(points, dtype=np.int64)
