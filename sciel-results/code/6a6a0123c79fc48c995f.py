import numpy as np
import time

def _verify(U, V, W, tol=1e-6):
    """Verify the tensor identity:
    sum_t U[i*n+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1 for all (i,j,k)
    n = p = 4
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0.0
                u = U[i*4+j, :]
                v = V[j*4+k, :]
                w = W[k*4+i, :]
                for t in range(rank):
                    s += u[t] * v[t] * w[t]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2():
    """
    Rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    rank = 7
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((4, rank), dtype=int)
    W = np.zeros((4, rank), dtype=int)
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0*2+0][0] = 1; U[1*2+1][0] = 1
    V[0*2+0][0] = 1; V[1*2+1][0] = 1
    W[0*2+0][0] = 1; W[1*2+1][0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    U[1*2+0][1] = 1; U[1*2+1][1] = 1
    V[0*2+0][1] = 1
    W[0*2+1][1] = 1; W[1*2+1][1] = -1
    
    # Term 3: M3 = A00(B01-B11)
    U[0*2+0][2] = 1
    V[0*2+1][2] = 1; V[1*2+1][2] = -1
    W[1*2+0][2] = 1; W[1*2+1][2] = 1
    
    # Term 4: M4 = A11(B10-B00)
    U[1*2+1][3] = 1
    V[1*2+0][3] = 1; V[0*2+0][3] = -1
    W[0*2+0][3] = 1; W[0*2+1][3] = 1
    
    # Term 5: M5 = (A00+A01)B11
    U[0*2+0][4] = 1; U[0*2+1][4] = 1
    V[1*2+1][4] = 1
    W[0*2+0][4] = -1; W[1*2+0][4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[1*2+0][5] = 1; U[0*2+0][5] = -1
    V[0*2+0][5] = 1; V[0*2+1][5] = 1
    W[1*2+1][5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[0*2+1][6] = 1; U[1*2+1][6] = -1
    V[1*2+0][6] = 1; V[1*2+1][6] = 1
    W[0*2+0][6] = 1
    
    return U, V, W

def _rank49_kronecker():
    """
    Construct the rank-49 decomposition using Kronecker product of Strassen 2x2.
    U, V, W each of shape (16, 49).
    U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    U2, V2, W2 = _strassen_2x2()
    
    # Kronecker product: rank = 7*7 = 49
    # U_kron has shape (4*4, 7*7) = (16, 49)
    # Row index of U_kron: (i1*2+i2)*4 + (j1*2+j2) = 4*i1*2 + 4*i2 + 2*j1 + j2
    #   = 8*i1 + 4*i2 + 2*j1 + j2
    # We need: row index = i*4 + j where i = 2*i1+i2, j = 2*j1+j2
    #   = (2*i1+i2)*4 + (2*j1+j2) = 8*i1 + 4*i2 + 2*j1 + j2
    # So the Kronecker row index already matches! No permutation needed for U.
    
    U4 = np.kron(U2, U2)  # (16, 49)
    V4 = np.kron(V2, V2)  # (16, 49)
    W4 = np.kron(W2, W2)  # (16, 49)
    
    # For U: row = i*4+j, i=2*i1+i2, j=2*j1+j2
    # kron(U2,U2) row = (i1*2+i2)*4 + (j1*2+j2) = 8*i1+4*i2+2*j1+j2
    # Target row = i*4+j = (2*i1+i2)*4+(2*j1+j2) = 8*i1+4*i2+2*j1+j2
    # They match! No permutation.
    
    # For V: row = j*4+k, j=2*j1+j2, k=2*k1+k2
    # kron(V2,V2) row = (j1*2+j2)*4 + (k1*2+k2) = 8*j1+4*j2+2*k1+k2
    # Target row = j*4+k = (2*j1+j2)*4+(2*k1+k2) = 8*j1+4*j2+2*k1+k2
    # They match! No permutation.
    
    # For W: row = k*4+i, k=2*k1+k2, i=2*i1+i2
    # kron(W2,W2) row = (k1*2+k2)*4 + (i1*2+i2) = 8*k1+4*k2+2*i1+i2
    # Target row = k*4+i = (2*k1+k2)*4+(2*i1+i2) = 8*k1+4*k2+2*i1+i2
    # They match! No permutation.
    
    return U4, V4, W4

def _to_outlist(A):
    """Convert numpy array to list of lists with int or half-int entries."""
    out = []
    for row in A:
        r = []
        for v in row:
            v = float(v)
            if abs(v - round(v)) < 1e-10:
                r.append(int(round(v)))
            else:
                if abs(2*v - round(2*v)) < 1e-10:
                    r.append(int(round(2*v)) / 2.0)
                else:
                    r.append(v)
        out.append(r)
    return out

def _greedy_prune(U, V, W, tol=1e-9, time_limit=None, t0=None):
    """
    Try to greedily remove columns (rank terms) while maintaining validity.
    Returns the pruned U, V, W.
    """
    if t0 is None:
        t0 = time.time()
    
    U_cur = U.astype(np.float64)
    V_cur = V.astype(np.float64)
    W_cur = W.astype(np.float64)
    rank_cur = U_cur.shape[1]
    
    # Keep trying to remove columns
    changed = True
    while changed:
        if time_limit is not None and time.time() - t0 > time_limit:
            break
        changed = False
        for rem in range(rank_cur):
            if time_limit is not None and time.time() - t0 > time_limit:
                break
            keep = [x for x in range(rank_cur) if x != rem]
            if len(keep) == 0:
                continue
            U_t = U_cur[:, keep]
            V_t = V_cur[:, keep]
            W_t = W_cur[:, keep]
            if _verify(U_t, V_t, W_t, tol=tol):
                U_cur = U_t
                V_cur = V_t
                W_cur = W_t
                rank_cur -= 1
                changed = True
                break  # Restart the removal loop with the new rank
    
    return U_cur, V_cur, W_cur, rank_cur

def construct():
    t0 = time.time()
    
    # Build rank-49 Kronecker product decomposition
    U49, V49, W49 = _rank49_kronecker()
    
    # Verify it's correct
    if not _verify(U49, V49, W49, tol=1e-9):
        # Should not happen, but fallback to schoolbook
        rank = 64
        U = np.zeros((16, rank), dtype=int)
        V = np.zeros((16, rank), dtype=int)
        W = np.zeros((16, rank), dtype=int)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return {"U": _to_outlist(U), "V": _to_outlist(V), "W": _to_outlist(W)}
    
    # Try to prune rank-49 to lower rank
    # Deadline guard: leave 20s margin before 120s limit
    remaining = 120.0 - (time.time() - t0) - 20.0
    if remaining > 5:
        U_p, V_p, W_p, rank_p = _greedy_prune(U49, V49, W49, tol=1e-9, time_limit=remaining, t0=t0)
        if rank_p < 49:
            return {
                "U": _to_outlist(U_p),
                "V": _to_outlist(V_p),
                "W": _to_outlist(W_p)
            }
    
    # Return rank-49
    return {
        "U": _to_outlist(U49),
        "V": _to_outlist(V49),
        "W": _to_outlist(W49)
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Valid:", _verify(res['U'], res['V'], res['W']))
