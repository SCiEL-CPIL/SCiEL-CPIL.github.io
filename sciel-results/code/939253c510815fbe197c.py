import numpy as np
import time
from fractions import Fraction

def _strassen_factors():
    """Strassen rank-7 for 2x2x2. Convention: U row i*2+j, V row j*2+k, W row k*2+i."""
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)
    # M1=(A00+A11)(B00+B11)
    U1[0,0]=1; U1[3,0]=1; V1[0,0]=1; V1[3,0]=1; W1[0,0]=1; W1[3,0]=1
    # M2=(A10+A11)B00
    U1[2,1]=1; U1[3,1]=1; V1[0,1]=1; W1[1,1]=1; W1[3,1]=-1
    # M3=A00(B01-B11)
    U1[0,2]=1; V1[1,2]=1; V1[3,2]=-1; W1[2,2]=1; W1[3,2]=1
    # M4=A11(B10-B00)
    U1[3,3]=1; V1[2,3]=1; V1[0,3]=-1; W1[0,3]=1; W1[1,3]=1
    # M5=(A00+A01)B11
    U1[0,4]=1; U1[1,4]=1; V1[3,4]=1; W1[0,4]=-1; W1[2,4]=1
    # M6=(A10-A00)(B00+B01)
    U1[2,5]=1; U1[0,5]=-1; V1[0,5]=1; V1[1,5]=1; W1[3,5]=1
    # M7=(A01-A11)(B10+B11)
    U1[1,6]=1; U1[3,6]=-1; V1[2,6]=1; V1[3,6]=1; W1[0,6]=1
    return U1, V1, W1


def _build_rank42():
    """Build rank-42 decomposition via 6 independent Strassen blocks."""
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()
    U = np.zeros((n*m, 42), dtype=float)
    V = np.zeros((m*p, 42), dtype=float)
    W = np.zeros((p*n, 42), dtype=float)
    for j_block in range(3):
        for side in range(2):
            b = j_block * 2 + side
            co = b * 7
            for i in range(2):
                for jl in range(2):
                    a = side*2 + jl
                    for s in range(7):
                        c = U1[i*2+jl, s]
                        if c != 0:
                            U[i*m+a, co+s] += c
            for jl in range(2):
                a = side*2 + jl
                for kl in range(2):
                    bc = j_block*2 + kl
                    for s in range(7):
                        c = V1[jl*2+kl, s]
                        if c != 0:
                            V[a*p+bc, co+s] += c
            for i in range(2):
                for kl in range(2):
                    bc = j_block*2 + kl
                    for s in range(7):
                        c = W1[kl*2+i, s]
                        if c != 0:
                            W[bc*n+i, co+s] += c
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] == 1 for all i,j,k."""
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                v = 0.0
                for s in range(R):
                    v += U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
                if abs(v - 1.0) > tol:
                    return False
    return True


def _try_drop_column(U, V, W, n, m, p, drop_idx):
    """Try to express column drop_idx as a linear combination of other columns.
    Returns (new_U, new_V, new_W) if successful, else None."""
    R = U.shape[1]
    # Keep columns: all except drop_idx
    keep = [s for s in range(R) if s != drop_idx]
    Uk = U[:, keep]
    Vk = V[:, keep]
    Wk = W[:, keep]

    # For each (i,j,k) the constraint is:
    # sum_s U[i*m+j,s] V[j*p+k,s] W[k*n+i,s] = 1
    # We need to find alpha such that for all (i,j,k):
    # U[i*m+j,drop] V[j*p+k,drop] W[k*n+i,drop] + sum_{s in keep} alpha[s] U[i*m+j,s] V[j*p+k,s] W[k*n+i,s] = 1

    # This is a linear system in alpha. Build it.
    n_constraints = n*m*p  # 48
    # Unknowns: alpha for each kept column = R-1 = 41
    n_unknowns = R - 1

    # Matrix A_mat: (n_constraints, n_unknowns), rhs: (n_constraints,)
    A_mat = np.zeros((n_constraints, n_unknowns), dtype=float)
    rhs = np.zeros(n_constraints, dtype=float)

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Contribution from drop column
                drop_val = U[i*m+j, drop_idx] * V[j*p+k, drop_idx] * W[k*n+i, drop_idx]
                # Move to rhs
                rhs[idx] = 1.0 - drop_val
                # Contribution from kept columns
                for t, s in enumerate(keep):
                    A_mat[idx, t] = U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
                idx += 1

    # Solve least squares
    try:
        alpha, residuals, rank, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
    except Exception:
        return None

    # Check residual
    if residuals and len(residuals) > 0:
        res = residuals[0]
        if res > 1e-6:
            return None
    else:
        # Compute residual manually
        res = np.linalg.norm(A_mat @ alpha - rhs)
        if res > 1e-6:
            return None

    # Build new matrices without the drop column
    new_U = Uk.copy()
    new_V = Vk.copy()
    new_W = Wk.copy()
    return new_U, new_V, new_W


def _to_output(U, V, W):
    """Convert numpy arrays to lists, using int where possible."""
    def convert(M):
        out = []
        for row in M:
            r = []
            for x in row:
                if abs(x - round(x)) < 1e-10:
                    r.append(int(round(x)))
                else:
                    r.append(float(x))
            out.append(r)
        return out
    return {"U": convert(U), "V": convert(V), "W": convert(W)}


def _schoolbook():
    """Fallback: rank 48 schoolbook decomposition."""
    n, m, p = 2, 4, 6
    R = n*m*p  # 48
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = 2, 4, 6
    start = time.time()
    deadline = start + 100.0

    # Start with rank-42
    U, V, W = _build_rank42()

    # Try to drop columns one at a time
    best_U, best_V, best_W = U, V, W
    best_rank = 42

    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = 42

    # Try dropping each column
    for attempt in range(current_rank):
        if time.time() > deadline - 10.0:
            break
        for drop_idx in range(current_rank):
            if time.time() > deadline - 10.0:
                break
            result = _try_drop_column(current_U, current_V, current_W, n, m, p, drop_idx)
            if result is not None:
                new_U, new_V, new_W = result
                if _verify(new_U, new_V, new_W, n, m, p):
                    current_U, current_V, current_W = new_U, new_V, new_W
                    current_rank -= 1
                    if current_rank < best_rank:
                        best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
                        best_rank = current_rank
                    # Restart the drop loop with new matrices
                    break

    # Return best found
    if best_rank <= current_rank and _verify(best_U, best_V, best_W, n, m, p):
        return _to_output(best_U, best_V, best_W)

    # Fallback to schoolbook if something went wrong
    U_fb, V_fb, W_fb = _schoolbook()
    return _to_output(U_fb, V_fb, W_fb)
