import time
import random
import math
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]
_START = time.time()
_DEADLINE = 145.0
_MARGIN = 5.0

def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)

def _residual(U, V, W, n, m, p):
    """Return max absolute tensor residual over all (i,j,k)."""
    worst = 0.0
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                uc = U[i * m + j]
                vc = V[j * p + k]
                wc = W[k * n + i]
                for c in range(r):
                    s += uc[c] * vc[c] * wc[c]
                diff = abs(s - 1.0)
                if diff > worst:
                    worst = diff
    return worst

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using rational arithmetic."""
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += Fraction(U[i * m + j][c]).limit_denominator(10**6) * \
                         Fraction(V[j * p + k][c]).limit_denominator(10**6) * \
                         Fraction(W[k * n + i][c]).limit_denominator(10**6)
                if s != Fraction(1):
                    return False
    return True

def _to_halfint(x):
    """Round to nearest half-integer."""
    return round(x * 2) / 2

def _refine(U, V, W, n, m, p, iters=200):
    """Gauss-Seidel refinement."""
    r = len(U[0]) if U else 0
    for _ in range(iters):
        if _remaining() < 10.0:
            break
        # Update W
        for i in range(n):
            for k in range(p):
                A = []
                for j in range(m):
                    row = [U[i * m + j][c] * V[j * p + k][c] for c in range(r)]
                    A.append(row)
                Aarr = np.array(A, dtype=float)
                b = np.ones(m)
                try:
                    sol, *_ = np.linalg.lstsq(Aarr, b, rcond=None)
                    for c in range(r):
                        W[k * n + i][c] = float(sol[c])
                except:
                    pass
        # Update U
        for i in range(n):
            for j in range(m):
                A = []
                for k in range(p):
                    row = [V[j * p + k][c] * W[k * n + i][c] for c in range(r)]
                    A.append(row)
                Aarr = np.array(A, dtype=float)
                b = np.ones(p)
                try:
                    sol, *_ = np.linalg.lstsq(Aarr, b, rcond=None)
                    for c in range(r):
                        U[i * m + j][c] = float(sol[c])
                except:
                    pass
        # Update V
        for j in range(m):
            for k in range(p):
                A = []
                for i in range(n):
                    row = [U[i * m + j][c] * W[k * n + i][c] for c in range(r)]
                    A.append(row)
                Aarr = np.array(A, dtype=float)
                b = np.ones(n)
                try:
                    sol, *_ = np.linalg.lstsq(Aarr, b, rcond=None)
                    for c in range(r):
                        V[j * p + k][c] = float(sol[c])
                except:
                    pass
    return U, V, W

def _try_round_verify(U, V, W, n, m, p):
    """Round to half-integers and verify."""
    Ur = [[_to_halfint(v) for v in row] for row in U]
    Vr = [[_to_halfint(v) for v in row] for row in V]
    Wr = [[_to_halfint(v) for v in row] for row in W]
    if _verify_exact(Ur, Vr, Wr, n, m, p):
        return {"U": Ur, "V": Vr, "W": Wr}
    return None

def _structured_init(rank, n, m, p, rng, strategy=0):
    """Generate structured initializations."""
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    if strategy == 0:
        # Block structure: pair up indices
        for c in range(rank):
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            U[i * m + j][c] = 1.0
            V[j * p + k][c] = 1.0
            W[k * n + i][c] = 1.0
    elif strategy == 1:
        # Random sparse half-integer
        vals = [-1.0, -0.5, 0.5, 1.0]
        for row in U:
            for c in range(rank):
                if rng.random() < 0.5:
                    row[c] = rng.choice(vals)
        for row in V:
            for c in range(rank):
                if rng.random() < 0.5:
                    row[c] = rng.choice(vals)
        for row in W:
            for c in range(rank):
                if rng.random() < 0.5:
                    row[c] = rng.choice(vals)
    elif strategy == 2:
        # Dense random
        vals = [-1.0, -0.5, 0.5, 1.0]
        for row in U:
            for c in range(rank):
                row[c] = rng.choice(vals)
        for row in V:
            for c in range(rank):
                row[c] = rng.choice(vals)
        for row in W:
            for c in range(rank):
                row[c] = rng.choice(vals)
    elif strategy == 3:
        # Float random
        for row in U:
            for c in range(rank):
                row[c] = rng.uniform(-2, 2)
        for row in V:
            for c in range(rank):
                row[c] = rng.uniform(-2, 2)
        for row in W:
            for c in range(rank):
                row[c] = rng.uniform(-2, 2)
    elif strategy == 4:
        # Symmetric pattern
        for c in range(rank):
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            U[i * m + j][c] = rng.choice([1.0, -1.0])
            V[j * p + k][c] = rng.choice([1.0, -1.0])
            W[k * n + i][c] = rng.choice([1.0, -1.0])
    
    return U, V, W

def _local_search(U, V, W, n, m, p, max_steps=2000):
    """Local search on half-integer grid."""
    r = len(U[0]) if U else 0
    Ur = [[_to_halfint(v) for v in row] for row in U]
    Vr = [[_to_halfint(v) for v in row] for row in V]
    Wr = [[_to_halfint(v) for v in row] for row in W]
    
    best_res = _residual(Ur, Vr, Wr, n, m, p)
    if best_res < 1e-10:
        rounded = _try_round_verify(Ur, Vr, Wr, n, m, p)
        if rounded:
            return rounded
    
    best_U = [row[:] for row in Ur]
    best_V = [row[:] for row in Vr]
    best_W = [row[:] for row in Wr]
    
    half_vals = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    
    for step in range(max_steps):
        if _remaining() < 15.0:
            break
        if best_res < 1e-10:
            break
        
        which = random.randint(0, 2)
        if which == 0:
            row_idx = random.randint(0, n * m - 1)
            col_idx = random.randint(0, r - 1)
            old = Ur[row_idx][col_idx]
            new = random.choice([v for v in half_vals if v != old])
            Ur[row_idx][col_idx] = new
            res = _residual(Ur, Vr, Wr, n, m, p)
            if res < best_res - 1e-12:
                best_res = res
                best_U = [row[:] for row in Ur]
                best_V = [row[:] for row in Vr]
                best_W = [row[:] for row in Wr]
                if best_res < 1e-10:
                    rounded = _try_round_verify(best_U, best_V, best_W, n, m, p)
                    if rounded:
                        return rounded
            else:
                Ur[row_idx][col_idx] = old
        elif which == 1:
            row_idx = random.randint(0, m * p - 1)
            col_idx = random.randint(0, r - 1)
            old = Vr[row_idx][col_idx]
            new = random.choice([v for v in half_vals if v != old])
            Vr[row_idx][col_idx] = new
            res = _residual(Ur, Vr, Wr, n, m, p)
            if res < best_res - 1e-12:
                best_res = res
                best_U = [row[:] for row in Ur]
                best_V = [row[:] for row in Vr]
                best_W = [row[:] for row in Wr]
                if best_res < 1e-10:
                    rounded = _try_round_verify(best_U, best_V, best_W, n, m, p)
                    if rounded:
                        return rounded
            else:
                Vr[row_idx][col_idx] = old
        else:
            row_idx = random.randint(0, p * n - 1)
            col_idx = random.randint(0, r - 1)
            old = Wr[row_idx][col_idx]
            new = random.choice([v for v in half_vals if v != old])
            Wr[row_idx][col_idx] = new
            res = _residual(Ur, Vr, Wr, n, m, p)
            if res < best_res - 1e-12:
                best_res = res
                best_U = [row[:] for row in Ur]
                best_V = [row[:] for row in Vr]
                best_W = [row[:] for row in Wr]
                if best_res < 1e-10:
                    rounded = _try_round_verify(best_U, best_V, best_W, n, m, p)
                    if rounded:
                        return rounded
            else:
                Wr[row_idx][col_idx] = old
    
    if best_res < 1e-10:
        rounded = _try_round_verify(best_U, best_V, best_W, n, m, p)
        if rounded:
            return rounded
    return None

def _try_rank(rank, n, m, p, rng, max_attempts=30):
    """Try to find valid decomposition of given rank."""
    for attempt in range(max_attempts):
        if _remaining() < 20.0:
            break
        
        strategy = attempt % 5
        U, V, W = _structured_init(rank, n, m, p, rng, strategy)
        
        U, V, W = _refine(U, V, W, n, m, p, iters=100)
        res = _residual(U, V, W, n, m, p)
        
        if res < 1e-6:
            rounded = _try_round_verify(U, V, W, n, m, p)
            if rounded:
                return rounded
            
            ls_result = _local_search(U, V, W, n, m, p, max_steps=1000)
            if ls_result:
                return ls_result
    
    return None

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    rng = random.Random(42)
    
    # Known lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # But actual rank for 2x3x6 is 12
    # Try from 12 upward
    for rank in range(12, best_rank):
        if _remaining() < 30.0:
            break
        
        result = _try_rank(rank, n, m, p, rng, max_attempts=20)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    return best

if __name__ == "__main__":
    print(construct())
