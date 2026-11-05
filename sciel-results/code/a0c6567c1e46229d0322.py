import numpy as np
from fractions import Fraction
import time

def construct():
    n, m, p = 3, 4, 5
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    def to_half_int(x):
        """Return Fraction if x is close to a half-integer, else None."""
        for denom in (1, 2):
            num = round(x * denom)
            if abs(x - num / denom) < 1e-9:
                return Fraction(num, denom)
        return None

    def verify_exact(U, V, W, rank):
        """Check exact decomposition using Fraction arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    # Try ranks from low to high
    for rank in range(18, 61):
        if time.time() > deadline:
            break

        # Multiple random trials with different structures
        for trial in range(300):
            if time.time() > deadline:
                break
            seed = trial * 7919 + rank * 104729
            np.random.seed(seed)

            # Strategy: use sparse binary matrices for U and V
            # Vary density based on trial
            if trial % 3 == 0:
                density = 0.3
            elif trial % 3 == 1:
                density = 0.5
            else:
                density = 0.7

            U = (np.random.rand(n * m, rank) < density).astype(float)
            V = (np.random.rand(m * p, rank) < density).astype(float)

            # Build linear system for W:
            # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            num_eq = n * m * p  # 60
            num_unk = p * n * rank

            A = np.zeros((num_eq, num_unk))
            b = np.ones(num_eq)

            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            col = (k * n + i) * rank + c
                            A[idx, col] = U[i * m + j, c] * V[j * p + k, c]
                        idx += 1

            try:
                W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=1e-12)
                W = W_flat.reshape(p * n, rank)

                # Quick float check
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                                      for c in range(rank))
                            max_err = max(max_err, abs(val - 1.0))

                if max_err < 1e-8:
                    # Try to convert to half-integers
                    U_frac = []
                    ok = True
                    for i in range(n * m):
                        row = []
                        for j in range(rank):
                            f = to_half_int(U[i, j])
                            if f is None:
                                ok = False
                                break
                            row.append(f)
                        if not ok:
                            break
                        U_frac.append(row)

                    if not ok:
                        continue

                    V_frac = []
                    ok = True
                    for i in range(m * p):
                        row = []
                        for j in range(rank):
                            f = to_half_int(V[i, j])
                            if f is None:
                                ok = False
                                break
                            row.append(f)
                        if not ok:
                            break
                        V_frac.append(row)

                    if not ok:
                        continue

                    W_frac = []
                    ok = True
                    for i in range(p * n):
                        row = []
                        for j in range(rank):
                            f = to_half_int(W[i, j])
                            if f is None:
                                ok = False
                                break
                            row.append(f)
                        if not ok:
                            break
                        W_frac.append(row)

                    if not ok:
                        continue

                    # Final exact verification
                    if verify_exact(U_frac, V_frac, W_frac, rank):
                        return {"U": U_frac, "V": V_frac, "W": W_frac}

            except Exception:
                pass

    # Fallback: schoolbook decomposition (rank 60)
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
