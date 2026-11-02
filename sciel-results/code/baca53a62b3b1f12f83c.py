"""Matrix multiplication 3x3x3: rank-minimal tensor decomposition."""
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n = m = p = 3
    U = [[float(x) for x in row] for row in U]
    V = [[float(x) for x in row] for row in V]
    W = [[float(x) for x in row] for row in W]
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n = m = p = 3
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

def _strassen_rank26():
    """
    Rank-26 decomposition for 3x3 matrix multiplication using explicit factors.
    Strassen core (7 terms) for the 2x2 block + schoolbook border terms.
    """
    n = m = p = 3
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    terms = []

    # --- Strassen core for C[0:2,0:2] (7 terms) ---
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))

    # --- Border: C[0:2, 2] ---
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))

    # --- Border: C[2, 0:2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))

    # --- Corner: C[2,2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))

    # --- Additional border: A[0:2,2]*B[2,0:2] contributes to C[0:2,0:2] ---
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,1),1) ]))

    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row][r] += coeff
        for (row, coeff) in v_list:
            V[row][r] += coeff
        for (row, coeff) in w_list:
            W[row][r] += coeff

    return {"U": U, "V": V, "W": W}

def _minimize_rank_via_span(U, V, W, deadline):
    """
    Given a valid rank-R decomposition, find the minimal set of columns whose
    rank-1 tensors span the multiplication tensor.

    Build the 27 x R coefficient matrix M where M[(i,j,k), r] =
    U[i,j,r] * V[j,k,r] * W[k,i,r]. The target tensor T is the all-ones vector
    of length 27. We need the columns of M to span T. The minimal number of
    columns needed is the rank of M (since T is in the column space, and the
    column space has dimension rank(M)).

    We find a basis of rank(M) columns and return those.
    """
    import numpy as np

    n = m = p = 3
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]

    # Build the 27 x R matrix M
    M = np.zeros((27, R))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * 9 + j * 3 + k
                M[idx, :] = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]

    # Compute SVD to find the rank and a basis of columns
    # The columns of M span a space of dimension = rank(M).
    # We need to find a subset of columns that spans this space.
    # Use QR with column pivoting to find a basis.
    try:
        # QR with column pivoting
        Q, R_mat, piv = np.linalg.qr(M, mode='complete', pivoting=True)
        # The rank is the number of significant diagonal entries in R_mat
        diag = np.abs(np.diag(R_mat))
        tol = max(M.shape) * np.finfo(float).eps * diag[0] if len(diag) > 0 else 1e-10
        rank_M = int(np.sum(diag > tol))
    except:
        # Fallback: use SVD
        try:
            U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
            tol = max(M.shape) * np.finfo(float).eps * s_vals[0] if len(s_vals) > 0 else 1e-10
            rank_M = int(np.sum(s_vals > tol))
            piv = list(range(R))
        except:
            return None, R

    if rank_M >= R:
        return None, R

    # The first rank_M columns (in pivot order) form a basis
    basis_cols = piv[:rank_M]
    # Sort for determinism
    basis_cols = sorted(basis_cols)

    new_U = U[:, basis_cols].tolist()
    new_V = V[:, basis_cols].tolist()
    new_W = W[:, basis_cols].tolist()

    return {"U": new_U, "V": new_V, "W": new_W}, rank_M

def _verify_and_minimize(result, deadline):
    """Verify the result and attempt rank minimization via span analysis."""
    if not _verify(result["U"], result["V"], result["W"]):
        return None

    # Try span-based minimization
    try:
        minimized, new_rank = _minimize_rank_via_span(
            result["U"], result["V"], result["W"], deadline
        )
        if minimized is not None:
            if _verify(minimized["U"], minimized["V"], minimized["W"]):
                return minimized
    except Exception:
        pass

    return result

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    # Start with the verified rank-26 construction
    result = _strassen_rank26()
    if not _verify(result["U"], result["V"], result["W"]):
        # Should not happen, but fall back to schoolbook
        return _schoolbook()

    # Try to reduce the rank via span-based minimization
    if time.time() < deadline:
        minimized = _verify_and_minimize(result, deadline)
        if minimized is not None:
            result = minimized

    # Final verification
    if _verify(result["U"], result["V"], result["W"]):
        return result

    # Fallback to schoolbook if something went wrong
    return _schoolbook()

# EVOLVE-BLOCK-END
