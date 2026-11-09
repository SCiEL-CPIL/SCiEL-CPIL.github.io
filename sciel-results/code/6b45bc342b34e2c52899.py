"""Rewritten with divisor-based search for 1/x+1/y+1/z=4/n."""
import time
import math

def _divisors(m):
    """Return all positive divisors of m (m > 0)."""
    if m <= 0:
        return []
    divs = []
    r = math.isqrt(m)
    for i in range(1, r + 1):
        if m % i == 0:
            divs.append(i)
            j = m // i
            if j != i:
                divs.append(j)
    return divs

def _solve_n(n, deadline):
    """Find distinct positive denominators x,y,z with 1/x+1/y+1/z=4/n.
    Returns [x,y,z] or None."""
    B0 = n * n  # B when k=0
    # k ranges from 1 to n-1 (so B = n*(n-k) ranges from n*(n-1) down to n)
    # We try k in order; for each, factor B^2+4nB and look for a divisor d
    # with y=B+d, z=B+(B^2+4nB)/d all distinct and >0.
    for k in range(1, n):
        B = n * (n - k)
        M = B * B + 4 * n * B  # = B(B + 4n)
        divs = _divisors(M)
        for d in divs:
            y = B + d
            e = M // d
            z = B + e
            # distinctness
            if y == z:
                continue
            x = n // 4 + k
            # careful: x must be integer; n//4 may not equal n/4.
            # Actually x = n/4 + k only if 4 divides n. Otherwise x is not integer.
            # Reconsider: x = n/4 + k. For x to be integer, 4 | n.
            # So this formula only works when 4|n. Otherwise we need a different approach.
            pass
        # The above is incomplete for n not divisible by 4.
        # Let me use a more general approach.
    return None

def construct():
    answers = []
    deadline = time.time() + 55  # safety margin
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        found = None
        # General approach: iterate x from floor(n/4)+1 up to 3n/4.
        # For each x, 1/y+1/z = 4/n - 1/x = (4x - n)/(nx).
        # Let P = 4x - n, Q = n*x. Then 1/y + 1/z = P/Q.
        # => (y - Q/P)(z - Q/P) ... but Q/P may not be integer.
        # Better: yz P = Q(y+z) => P y z - Q y - Q z = 0
        # => (P y - Q)(P z - Q) = Q^2
        # So P*y - Q and P*z - Q are divisors of Q^2.
        # Let u = P*y - Q, v = P*z - Q, u*v = Q^2.
        # y = (u + Q)/P, z = (v + Q)/P. Need P | (u+Q) and P | (v+Q).
        # Also y, z > 0 and distinct, and distinct from x.
        start_x = n // 4 + 1
        if (4 * start_x) <= n:
            start_x = n // 4 + 1
        # x must satisfy 1/x < 4/n => x > n/4, and 3/x > 4/n => x < 3n/4
        end_x = (3 * n) // 4
        if end_x < start_x:
            end_x = start_x
        for x in range(start_x, end_x + 1):
            if time.time() > deadline:
                break
            P = 4 * x - n
            if P <= 0:
                continue
            Q = n * x
            Q2 = Q * Q
            divs = _divisors(Q2)
            for u in divs:
                v = Q2 // u
                # y = (u + Q) / P, z = (v + Q) / P
                nu = u + Q
                nv = v + Q
                if nu % P != 0 or nv % P != 0:
                    continue
                y = nu // P
                z = nv // P
                if y <= 0 or z <= 0:
                    continue
                if y == z:
                    continue
                if y == x or z == x:
                    continue
                # all distinct
                if y == z:
                    continue
                # denominators bound
                if y > 10**12 or z > 10**12 or x > 10**12:
                    continue
                found = [x, y, z]
                break
            if found:
                break
        answers.append(found)
    return answers
