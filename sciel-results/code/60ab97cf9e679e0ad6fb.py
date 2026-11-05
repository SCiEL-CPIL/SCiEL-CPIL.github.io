"""Graph-embedding search for rational integral-point sets (D=1 plane)."""
import time
from fractions import Fraction as F
from math import isqrt, gcd

_DEADLINE = time.time() + 130.0

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _sqdist(a, b):
    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])

def _collinear3(P, i, j, k):
    a, b, c = P[i], P[j], P[k]
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0 and \
           (b[0] - a[0]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[0] - a[0]) == 0 and \
           (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]) == 0

def _concyclic4(P, i, j, k, l):
    a, b, c, d = P[i], P[j], P[k], P[l]
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    wx, wy, wz = d[0] - a[0], d[1] - a[1], d[2] - a[2]
    n0 = uy * vz - uz * vy
    n1 = uz * vx - ux * vz
    n2 = ux * vy - uy * vx
    if n0 == 0 and n1 == 0 and n2 == 0:
        return False
    cx, cy, cz = (wx * vz - wz * vy, wz * vx - wx * vz, wx * vy - wy * vx)
    return cx * ux + cy * uy + cz * uz == 0 and cx * vx + cy * vy + cz * vz == 0 and cx * wx + cy * wy + cz * wz == 0

def _add_point(P, cand):
    n = len(P)
    for i in range(n):
        if _sqdist(P[i], cand) < 0:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            if _collinear3(P, i, j, n):
                pass
    # collinear / concyclic with the candidate (index n)
    for i in range(n):
        for j in range(i + 1, n):
            if _collinear3(P + [cand], i, j, n):
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _concyclic4(P + [cand], i, j, k, n):
                    return False
    return True

def _add_point_fast(P, cand):
    n = len(P)
    for i in range(n):
        if _sqdist(P[i], cand) < 0:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            if _collinear3(P, i, j, n) or _collinear3(P, i, n, j):
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _concyclic4(P, i, j, k, n):
                    return False
    return True

def _add_point_full(P, cand):
    n = len(P)
    for i in range(n):
        for j in range(i + 1, n):
            if _collinear3(P, i, j, n):
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _concyclic4(P, i, j, k, n):
                    return False
    return True

def _intersections(P, i, j, k, r2i, r2j, r2k):
    A, B, C = P[i], P[j], P[k]
    r2A, r2B, r2C = r2i, r2j, r2k
    u0 = B[0] - A[0]; u1 = B[1] - A[1]; u2 = B[2] - A[2]
    v0 = C[0] - A[0]; v1 = C[1] - A[1]; v2 = C[2] - A[2]
    dAB = u0 * u0 + u1 * u1 + u2 * u2
    dAC = v0 * v0 + v1 * v1 + v2 * v2
    dBC = (C[0] - B[0]) * (C[0] - B[0]) + (C[1] - B[1]) * (C[1] - B[1]) + (C[2] - B[2]) * (C[2] - B[2])
    if dAB == 0 or dAC == 0 or dBC == 0:
        return []
    # M (projection of A onto plane BCD)
    M0 = (dAB * (dAB + dAC - dBC) * A[0] + dAC * (dAB + dBC - dAC) * B[0] + (dAB + dAC - dBC) * (dAB + dBC - dAC) * C[0]) / (2 * dAB * dAC)
    M1 = (dAB * (dAB + dAC - dBC) * A[1] + dAC * (dAB + dBC - dAC) * B[1] + (dAB + dAC - dBC) * (dAB + dBC - dAC) * C[1]) / (2 * dAB * dAC)
    M2 = (dAB * (dAB + dAC - dBC) * A[2] + dAC * (dAB + dBC - dAC) * B[2] + (dAB + dAC - dBC) * (dAB + dBC - dAC) * C[2]) / (2 * dAB * dAC)
    # circle radius
    r2A_m = (A[0] - M0) * (A[0] - M0) + (A[1] - M1) * (A[1] - M1) + (A[2] - M2) * (A[2] - M2)
    r2B_m = (B[0] - M0) * (B[0] - M0) + (B[1] - M1) * (B[1] - M1) + (B[2] - M2) * (B[2] - M2)
    r2C_m = (C[0] - M0) * (C[0] - M0) + (C[1] - M1) * (C[1] - M1) + (C[2] - M2) * (C[2] - M2)
    if r2A_m < 0 or r2B_m < 0 or r2C_m < 0:
        return []
    r2 = r2A_m - r2i
    if r2 < 0:
        return []
    # basis of plane BCD
    w0 = u0 * v2 - u2 * v1
    w1 = u1 * v0 - u0 * v2
    w2 = u2 * v1 - u1 * v0
    if w0 == 0 and w1 == 0 and w2 == 0:
        return []
    w2_len = w0 * w0 + w1 * w1 + w2 * w2
    if w2_len == 0:
        return []
    e0 = (w0 / w2_len, w1 / w2_len, w2 / w2_len)
    # find a perpendicular vector in the plane
    p0 = 0; p1 = 0; p2 = 0
    if abs(w2) >= abs(w0) and abs(w2) >= abs(w1):
        p0 = -w1 / w2; p1 = w0 / w2; p2 = 0
    elif abs(w0) >= abs(w1):
        p0 = 0; p1 = -w2 / w0; p2 = w1 / w0
    else:
        p0 = -w2 / w1; p1 = 0; p2 = w0 / w1
    p2_len = p0 * p0 + p1 * p1 + p2 * p2
    if p2_len == 0:
        return []
    e1 = (p0 / p2_len, p1 / p2_len, p2 / p2_len)
    # check orthogonality
    if e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2] != 0:
        return []
    if not _is_square(r2.numerator) or not _is_square(r2.denominator):
        return []
    r = F(isqrt(r2.numerator), isqrt(r2.denominator))
    # circle center in plane coords
    c0 = (M0 - A[0]) * e0[0] + (M1 - A[1]) * e0[1] + (M2 - A[2]) * e0[2]
    c1 = (M0 - A[0]) * e1[0] + (M1 - A[1]) * e1[1] + (M2 - A[2]) * e1[2]
    out = []
    for t in range(-12, 13):
        if t == 0:
            continue
        d = t * t + 1
        s = F(2 * t, d)
        c = F(t * t - 1, d)
        x0 = c0 + r * s
        x1 = c1 + r * c
        pt = (A[0] + x0 * e0[0] + x1 * e1[0],
              A[1] + x0 * e0[1] + x1 * e1[1],
              A[2] + x0 * e0[2] + x1 * e1[2])
        out.append(pt)
    return out

def _try_add(P, r2, cand, add_fn):
    if cand in P:
        return False
    if not add_fn(P, cand):
        return False
    P.append(cand)
    r2 = [r2 + [_sqdist(P[-1], P[i]) for i in range(len(P) - 1)]]
    return True

def _search_from(base):
    P = list(base)
    n = len(P)
    r2 = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(_sqdist(P[i], P[j]))
        r2.append(row)
    add_fn = _add_point_full
    while time.time() < _DEADLINE:
        found = False
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for cand in _intersections(P, i, j, k, r2[i], r2[j], r2[k]):
                        if cand in P:
                            continue
                        if not add_fn(P, cand):
                            continue
                        P.append(cand)
                        nr2 = []
                        for a in range(len(P) - 1):
                            row = []
                            for b in range(len(P) - 1):
                                row.append(r2[a][b])
                            row.append(_sqdist(P[-1], P[a]))
                            nr2.append(row)
                        nr2.append([_sqdist(P[-1], P[a]) for a in range(len(P) - 1)] + [F(0)])
                        r2 = nr2
                        n = len(P)
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
    return P

def construct():
    # base: rational tetrahedron with all 6 edges rational
    base = [(F(0), F(0), F(0)), (F(3), F(4), F(0)), (F(0), F(0), F(5)), (F(4), F(0), F(3))]
    best = base
    P = _search_from(base)
    if len(P) > len(best):
        best = P
    return {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for (x, y, z) in best]}
