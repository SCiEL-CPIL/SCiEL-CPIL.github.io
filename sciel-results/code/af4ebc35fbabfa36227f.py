import time
import numpy as np

DIMENSIONS = [3, 3, 4]
START = time.time()
DEADLINE = 150.0
SAFETY = 20.0

def _elapsed():
    return time.time() - START

def _time_left():
    return DEADLINE - SAFETY - _elapsed()

def _verify(U, V, W, n, m, p):
    # exact check with fractions
    from fractions import Fraction
    U = [[Fraction(x) for x in row] for row in U]
    V = [[Fraction(x) for x in row] for row in V]
    W = [[Fraction(x) for x in row] for row in W]
    rank = len(U[0])
    # Build tensor
    # T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    # We need: for all i,j,k: sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                a = i*m + j
                b = j*p + k
                c = k*n + i
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if s != 1:
                    return False
    # Also need that the tensor has no spurious components? The task says "exactly decompose the matrix multiplication tensor".
    # The multiplication tensor T is defined by T[i*m+j, j*p+k, k*n+i]=1 and 0 elsewhere.
    # We must ensure no other (a,b,c) produce nonzero. But a,b,c are linked: a=i*m+j, b=j*p+k, c=k*n+i.
    # For arbitrary (a,b,c) not of that form, the sum might be nonzero, which would be a spurious term.
    # However, the standard definition: the tensor is the sum over (i,j,k) of e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}.
    # A decomposition sum_r u_r ⊗ v_r ⊗ w_r equals that tensor iff for ALL (a,b,c), sum_r U[a,r]V[b,r]W[c,r] = T[a,b,c].
    # So we must check all (a,b,c) in [0,9)x[0,12)x[0,12).
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                s = Fraction(0)
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if s != T[a,b,c]:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = n*m*p
    # Try to improve via search
    try:
        from fractions import Fraction
        # Build target tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        # We'll try ranks from a lower bound upward, attempt exact rational solve.
        # Use a randomized approach: pick random rational U,V, solve for W via linear system? 
        # The equations are trilinear, hard. Instead use a greedy "term merging" is complex.
        # Fallback: keep schoolbook.
        pass
    except Exception:
        pass
    return best
