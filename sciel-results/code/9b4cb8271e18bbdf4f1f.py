"""Integral point sets in general position: circle-intersection search."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    t_start = time.time()
    DEADLINE = t_start + 132  # safety margin before 150s kill

    def time_ok():
        return time.time() < DEADLINE

    def _gcd(a, b):
        a = abs(a); b = abs(b)
        while b:
            a, b = b, a % b
        return a

    def is_rational_square(num, den):
        if num == 0:
            return (0, 1)
        if num < 0 or den <= 0:
            return None
        g = _gcd(abs(num), den)
        n = num // g
        d = den // g
        rn = isqrt(n)
        if rn * rn != n:
            return None
        rd = isqrt(d)
        if rd * rd != d:
            return None
        return (rn, rd)

    def dist2(p, q):
        dxn = p[0][0]*q[0][1] - q[0][0]*p[0][1]
        dxd = p[0][1]*q[0][1]
        dyn = p[1][0]*q[1][1] - q[1][0]*p[1][1]
        dyd = p[1][1]*q[1][1]
        n = dxn*dxn*dyd*dyd + dyn*dyn*dxd*dxd
        d = dxd*dxd*dyd*dyd
        g = _gcd(n, d)
        return (n//g, d//g)

    def is_integer_dist(p, q):
        n, d = dist2(p, q)
        if n == 0:
            return False
        if d != 1:
            return False
        r = isqrt(n)
        return r*r == n and r > 0

    def _sub(a, b):
        return (a[0]*b[1] - b[0]*a[1], a[1]*b[1])

    def _mul(a, b):
        return (a[0]*b[0], a[1]*b[1])

    def collinear(p, q, r):
        t1 = _mul(_sub(q[0], p[0]), _sub(r[1], p[1]))
        t2 = _mul(_sub(q[1], p[1]), _sub(r[0], p[0]))
        return t1[0]*t2[1] == t2[0]*t1[1]

    def concyclic(a, b, c, d):
        def sq(p):
            n = p[0][0]**2 * p[1][1]**2 + p[1][0]**2 * p[0][1]**2
            dd = p[0][1]**2 * p[1][1]**2
            g = _gcd(n, dd)
            return (n//g, dd//g)
        rows = []
        for p in (a, b, c, d):
            rows.append([p[0], p[1], sq(p), (1,1)])
        def det3(R):
            (a1,b1,c1),(a2,b2,c2),(a3,b3,c3) = R
            t1 = _sub(_mul(b2,c3), _mul(c2,b3))
            t2 = _sub(_mul(a2,c3), _mul(c2,a3))
            t3 = _sub(_mul(a2,b3), _mul(b2,a3))
            n = (mul(a1,t1)[0]*mul(b1,t2)[1]*mul(c1,t3)[2]
                 - mul(b1,t2)[0]*mul(a1,t1)[1]*mul(c1,t3)[2]
                 + mul(c1,t3)[0]*mul(a1,t1)[1]*mul(b1,t2)[2])
            dd = mul(a1,t1)[1]*mul(b1,t2)[1]*mul(c1,t3)[1]
            g = _gcd(n, dd)
            return (n//g, dd//g)
        def mul(x, y):
            return (x[0]*y[0], x[1]*y[1])
        # 4x4 det via cofactor expansion along row 0
        total_n = 0; total_d = 1
        for j in range(4):
            minor = []
            for i in range(1,4):
                row = []
                for k in range(4):
                    if k != j:
                        row.append(rows[i][k])
                minor.append(row)
            val = det3(minor)
            sign = 1 if j % 2 == 0 else -1
            total_n = total_n*val[1] + sign*val[0]*total_d
            total_d = total_d*val[1]
            g = _gcd(abs(total_n), total_d)
            total_n //= g; total_d //= g
        return total_n == 0

    def validate(points):
        n = len(points)
        if n < 3:
            return False
        for i in range(n):
            for j in range(i+1, n):
                if not is_integer_dist(points[i], points[j]):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if collinear(points[i], points[j], points[k]):
                        return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    def ip(x, y):
        return [[x, 1], [y, 1]]

    seed = [ip(0,0), ip(-56,33), ip(-16,63), ip(0,33), ip(16,63)]
    if not validate(seed):
        seed = [ip(0,0), ip(3,4), ip(13,84)]
        if not validate(seed):
            seed = [ip(0,0), ip(4,3), ip(12,5)]
    best = seed[:]

    def try_add(points, cand):
        for p in points:
            if not is_integer_dist(p, cand):
                return False
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if collinear(points[i], points[j], cand):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if concyclic(points[i], points[j], points[k], cand):
                        return False
        return True

    def circle_candidates(points, max_r):
        n = len(points)
        seen = set()
        for i in range(n):
            for j in range(i+1, n):
                A = points[i]; B = points[j]
                d2n, d2d = dist2(A, B)
                if d2d != 1:
                    continue
                d = isqrt(d2n)
                if d*d != d2n or d == 0:
                    continue
                dxn = B[0][0]*A[0][1] - A[0][0]*B[0][1]
                dxd = A[0][1]*B[0][1]
                dyn = B[1][0]*A[1][1] - A[1][0]*B[1][1]
                dyd = A[1][1]*B[1][1]
                for ra in range(1, max_r+1):
                    for rb in range(1, max_r+1):
                        num = ra*ra - rb*rb + d*d
                        den = 2*d
                        g2 = _gcd(abs(num), den)
                        num //= g2; den //= g2
                        hn = ra*ra*den*den - num*num
                        hd = den*den
                        hs = is_rational_square(hn, hd)
                        if hs is None:
                            continue
                        hp, hq = hs
                        for sgn in (1, -1):
                            axn = num*dxn; axd = den*d*dxd
                            hxn = sgn*hp*(-dyn); hxd = hq*d*dyd
                            t1n = A[0][0]*axd + axn*A[0][1]; t1d = A[0][1]*axd
                            g3 = _gcd(abs(t1n), t1d); t1n//=g3; t1d//=g3
                            xnum = t1n*hxd + hxn*t1d; xden = t1d*hxd
                            g4 = _gcd(abs(xnum), xden); xnum//=g4; xden//=g4
                            ayn = num*dyn; ayd = den*d*dyd
                            hyn = sgn*hp*(dxn); hyd = hq*d*dxd
                            t2n = A[1][0]*ayd + ayn*A[1][1]; t2d = A[1][1]*ayd
                            g5 = _gcd(abs(t2n), t2d); t2n//=g5; t2d//=g5
                            ynum = t2n*hyd + hyn*t2d; yden = t2d*hyd
                            g6 = _gcd(abs(ynum), yden); ynum//=g6; yden//=g6
                            key = (xnum, xden, ynum, yden)
                            if key in seen:
                                continue
                            seen.add(key)
                            yield [[xnum, xden], [ynum, yden]]

    improved = True
    while improved and time_ok() and len(best) < 12:
        improved = False
        max_r = 40 + 15*len(best)
        for cand in circle_candidates(best, max_r):
            if not time_ok():
                break
            if try_add(best, cand):
                best.append(cand)
                improved = True
                break
    if not validate(best):
        best = seed[:]
    return {'D': 1, 'points': [[[p[0][0], p[0][1]], [p[1][0], p[1][1]]] for p in best]}
# EVOLVE-BLOCK-END
