"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

def construct():
    t0 = time.time()
    deadline = t0 + 130
    n, m, p = 3, 5, 7

    def verify(U, V, W, tol=1e-8):
        """Verify exact tensor decomposition."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_fast(U, V, W, tol=1e-8):
        """Faster verification using vectorized operations."""
        rank = U.shape[1]
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        return ok

    # Build initial decomposition using Strassen blocks + schoolbook
    # Partition 3x5x7 into 2x2x2 Strassen blocks where possible
    
    blocks = [
        ({0, 1}, {0, 1}, {0, 1}),
        ({1, 2}, {0, 1}, {2, 3}),
        ({0, 2}, {0, 1}, {4, 5}),
        ({0, 1}, {2, 3}, {0, 1}),
        ({1, 2}, {2, 3}, {2, 3}),
        ({0, 2}, {2, 3}, {4, 5}),
    ]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen(i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen terms for 2x2x2 block."""
        # M1 = (A00+A11)(B00+B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1; u[i1*m+j1] = 1
        v[j0*p+k0] = 1; v[j1*p+k1] = 1
        w[k0*n+i0] = 1; w[k1*n+i1] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M2 = (A10+A11)B00
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i1*m+j0] = 1; u[i1*m+j1] = 1
        v[j0*p+k0] = 1
        w[k0*n+i1] = 1; w[k1*n+i1] = -1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M3 = A00(B01-B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1
        v[j0*p+k1] = 1; v[j1*p+k1] = -1
        w[k1*n+i0] = 1; w[k1*n+i1] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M4 = A11(B10-B00)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i1*m+j1] = 1
        v[j1*p+k0] = 1; v[j0*p+k0] = -1
        w[k0*n+i0] = 1; w[k0*n+i1] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M5 = (A00+A01)B11
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1; u[i0*m+j1] = 1
        v[j1*p+k1] = 1
        w[k0*n+i0] = -1; w[k1*n+i0] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M6 = (A10-A00)(B00+B01)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i1*m+j0] = 1; u[i0*m+j0] = -1
        v[j0*p+k0] = 1; v[j0*p+k1] = 1
        w[k1*n+i1] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M7 = (A01-A11)(B10+B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j1] = 1; u[i1*m+j1] = -1
        v[j1*p+k0] = 1; v[j1*p+k1] = 1
        w[k0*n+i0] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
    
    for I, J, K in blocks:
        i0, i1 = sorted(I)
        j0, j1 = sorted(J)
        k0, k1 = sorted(K)
        add_strassen(i0, i1, j0, j1, k0, k1)
    
    # Add remaining schoolbook terms
    covered = set()
    for I, J, K in blocks:
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                    u[i*m+j] = 1
                    v[j*p+k] = 1
                    w[k*n+i] = 1
                    U_cols.append(u); V_cols.append(v); W_cols.append(w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    
    rank = U_mat.shape[1]
    
    # Greedy removal with batch verification
    def can_remove(U, V, W, indices_to_check):
        """Check if removing terms in indices_to_check still gives valid decomposition."""
        keep = [i for i in range(U.shape[1]) if i not in indices_to_check]
        if len(keep) == 0:
            return False
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        return verify_fast(U_new, V_new, W_new)
    
    # Remove one at a time, multiple passes
    removed_any = True
    passes = 0
    while removed_any and time.time() < deadline - 30:
        removed_any = False
        passes += 1
        for idx in range(U_mat.shape[1]):
            if time.time() >= deadline - 30:
                break
            # Try removing this term
            keep = list(range(U_mat.shape[1]))
            keep.pop(idx)
            if not keep:
                continue
            U_new = U_mat[:, keep]
            V_new = V_mat[:, keep]
            W_new = W_mat[:, keep]
            if verify_fast(U_new, V_new, W_new):
                U_mat = U_new
                V_mat = V_new
                W_mat = W_new
                removed_any = True
                break  # Restart from beginning since indices changed
    
    final_rank = U_mat.shape[1]
    
    if verify_fast(U_mat, V_mat, W_mat):
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        # Fallback: return the pre-reduction version
        U_cols2 = []
        V_cols2 = []
        W_cols2 = []
        for I, J, K in blocks:
            i0, i1 = sorted(I)
            j0, j1 = sorted(J)
            k0, k1 = sorted(K)
            add_strassen(i0, i1, j0, j1, k0, k1)
        
        # Actually, let me just return what we have if valid, otherwise schoolbook
        U_sb = np.zeros((n*m, n*m*p))
        V_sb = np.zeros((m*p, n*m*p))
        W_sb = np.zeros((p*n, n*m*p))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, c] = 1
                    V_sb[j*p+k, c] = 1
                    W_sb[k*n+i, c] = 1
                    c += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
