def _try_fixed_UV_solve_W(n, m, p, r, U, V):
    """Given U and V, solve for W. Returns W or None."""
    W = [[0]*r for _ in range(p*n)]
    for i in range(n):
        for k in range(p):
            # Build 3 x r system for W[k*n+i, :]
            A_sys = []
            b_sys = []
            for j in range(m):
                row = []
                for c in range(r):
                    row.append(U[i*m+j][c] * V[j*p+k][c])
                A_sys.append(row)
                b_sys.append(1)
            # Solve A_sys * x = b_sys where x = W[k*n+i, :]
            try:
                import numpy as np
                A_np = np.array(A_sys, dtype=float)
                b_np = np.array(b_sys, dtype=float)
                if r == m:
                    # Square system
                    det = np.linalg.det(A_np)
                    if abs(det) < 1e-10:
                        return None
                    x = np.linalg.solve(A_np, b_np)
                elif r > m:
                    # Overdetermined, use least squares but check residual
                    x, residuals, rank, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
                    if residuals and residuals[0] > 1e-8:
                        return None
                else:
                    # Underdetermined, underdetermined - skip
                    return None
                W[k*n+i] = [float(v) for v in x]
            except:
                return None
    return W
