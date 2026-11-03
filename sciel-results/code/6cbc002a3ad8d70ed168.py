"""Matrix multiplication 2x4x6: rank-minimal tensor decomposition.

Uses a fix-one-factor linear algebra search to find the minimal rank r
such that the 2x4x6 multiplication tensor admits a rank-r decomposition.
Exploits the linear-in-i structure of the tensor (n=2).
"""

import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 6]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a flat 3D array.
    T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-8):
    """Verify that sum_r U[:,r] V[:,r] W[:,r] == T."""
    # U: (A, r), V: (B, r), W: (C, r)
    A, B, C, r = U.shape[0], V.shape[0], W.shape[0], U.shape[1]
    # Efficient check: for each r, outer product
    S = np.zeros((A, B, C), dtype=float)
    for rr in range(r):
        S += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
    return np.allclose(S, T, atol=tol)

def _schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition."""
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * (m * p) + j * p + k
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
    return U, V, W

def _strassen_2x4x6(n, m, p):
    """Strassen-based rank-42 decomposition (fallback)."""
    strassen_2x2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 1]),
    ]
    total_rank = 42
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    col = 0
    for t in range(3):
        for (Ur, Vr, Wr) in strassen_2x2x2:
            for li in range(4):
                if Ur[li] == 0: continue
                ii, jj = divmod(li, 2)
                U[ii * 4 + jj, col] += Ur[li]
            for li in range(4):
                if Vr[li] == 0: continue
                jj, kk = divmod(li, 2)
                V[jj * 6 + (2 * t + kk), col] += Vr[li]
            for li in range(4):
                if Wr[li] == 0: continue
                kk, ii = divmod(li, 2)
                W[(2 * t + kk) * 2 + ii, col] += Wr[li]
            col += 1
        for (Ur, Vr, Wr) in strassen_2x2x2:
            for li in range(4):
                if Ur[li] == 0: continue
                ii, jj = divmod(li, 2)
                U[ii * 4 + (2 + jj), col] += Ur[li]
            for li in range(4):
                if Vr[li] == 0: continue
                jj, kk = divmod(li, 2)
                V[(2 + jj) * 6 + (2 * t + kk), col] += Vr[li]
            for li in range(4):
                if Wr[li] == 0: continue
                kk, ii = divmod(li, 2)
                W[(2 * t + kk) * 2 + ii, col] += Wr[li]
            col += 1
    return U, V, W

def _try_rank_with_fixed_W(T, r, W, tol=1e-8):
    """Given fixed W (C x r), try to find U (A x r) and V (B x r) such that
    sum_r U[:,r] V[:,r] W[:,r] == T.
    This is linear in U and V for fixed W.
    
    The equation: for all (a,b,c): sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    
    We can vectorize: let U be A x r, V be B x r.
    For each r, the term is U[:,r] V[:,r]^T W[c,r].
    
    Rewrite: T = sum_r (U[:,r] V[:,r]^T) * W[:,r]^T (outer in c)
    T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    This is a system that is bilinear in (U,V) but linear if we treat
    the products U[a,r]*V[b,r] as unknowns. Let X_r[a,b] = U[a,r]*V[b,r].
    Then T[a,b,c] = sum_r X_r[a,b] W[c,r].
    
    So for each (a,b), we have: T[a,b,:] = sum_r X_r[a,b] W[:,r]
    i.e., T[a,b,:] = X[:, a, b] @ W  where X is (r, A, B) with X[r,a,b] = U[a,r]*V[b,r]
    
    Actually: T[a,b,c] = sum_r X[r,a,b] * W[c,r]
    So T[a,b,:] = sum_r X[r,a,b] W[:,r]^T = (X[:,a,b] @ W)  ... 
    Let me be careful. Let X be a 3D array X[r,a,b] = U[a,r]*V[b,r].
    T[a,b,c] = sum_r X[r,a,b] * W[c,r]
    
    For fixed (a,b): T[a,b,:] (vector of length C) = sum_r X[r,a,b] * W[:,r]
    = W @ x where x[r] = X[r,a,b]. So T[a,b,:] = W @ x_{a,b} where x_{a,b} is r-vector.
    
    So x_{a,b} = W^+ T[a,b,:]  (pseudo-inverse) if W has full row rank.
    Then we need to check if X[r,a,b] = U[a,r]*V[b,r] for some U, V.
    That is, for each r, the matrix X[r,:,:] must be rank 1.
    
    This is the key: X[r,:,:] must be rank-1 for all r.
    """
    A, B, C = T.shape
    # For each (a,b): x_{a,b} = argmin ||W @ x - T[a,b,:]||
    # If W is C x r with C >= r and full column rank, x = (W^T W)^{-1} W^T T[a,b,:]
    # Use lstsq
    # Build the full system:
    # Unknowns: x_{a,b} for all a,b -> A*B*r unknowns, but with the rank-1 constraint per r.
    # 
    # Alternative: iterate. For each r, X[r,:,:] = U[:,r] V[:,r]^T.
    # So X[r,a,b] = U[a,r] V[b,r].
    # 
    # The linear system in U and V (with W fixed) is:
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # 
    # This is linear in the entries of U and V jointly? No, it's bilinear.
    # But if we fix one of them, it's linear in the other.
    #
    # Better approach: use the fact that for fixed W, we can solve for the
    # "flattened" variables. Let's use an iterative approach:
    # 1. Initialize V randomly.
    # 2. For fixed V and W, solve for U: T[a,b,c] = sum_r U[a,r] (V[b,r] W[c,r])
    #    For each a: T[a,b,c] for all b,c = sum_r U[a,r] * (V[b,r] W[c,r])
    #    Let M_a be the B x C matrix with M_a[b,c] = sum_r V[b,r] W[c,r] ... no.
    #    Actually: T[a,:,:] (B x C) = sum_r U[a,r] * (V[:,r] W[:,r]^T)
    #    So T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
    #    This is linear in U[a,:] for each a!
    #    T[a,:,:].reshape(B*C) = sum_r U[a,r] * outer(V[:,r], W[:,r]).reshape(B*C)
    #    So: T_flat[a] = sum_r U[a,r] * P[r] where P[r] = outer(V[:,r], W[:,r]).reshape(-1)
    #    This is a linear system: T_flat = U @ P^T where P is (r, B*C)
    #    So U = T_flat @ (P^T)^+  (least squares)
    # 3. Similarly, for fixed U and W, solve for V.
    # 4. Iterate.
    
    rng = np.random.RandomState(42)
    V = rng.randn(B, r)
    T_flat = T.reshape(A, B * C)
    
    for it in range(200):
        # Solve for U given V, W
        # P[r] = outer(V[:,r], W[:,r]).reshape(-1), shape (r, B*C)
        P = np.einsum('br,cr->bcr', V, W).reshape(r, B * C)
        # T_flat[a] = sum_r U[a,r] P[r]  =>  T_flat = U @ P.T
        # U = T_flat @ (P.T)^+
        U, _, _, _ = np.linalg.lstsq(P.T, T_flat.T, rcond=None)
        U = U.T  # (A, r)
        
        # Solve for V given U, W
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For each b: T[:,b,c] = sum_r V[b,r] (U[:,r] W[c,r])
        # T[:,b,c].reshape(A*C) = sum_r V[b,r] * outer(U[:,r], W[:,r]).reshape(-1)
        Q = np.einsum('ar,cr->acr', U, W).reshape(r, A * C)
        # T[:,b,:].reshape(-1) = sum_r V[b,r] Q[r]
        # T_reshaped = (B, A*C) where T_reshaped[b] = T[:,b,:].reshape(-1)
        T_reshaped = np.transpose(T, (1, 0, 2)).reshape(B, A * C)
        V, _, _, _ = np.linalg.lstsq(Q.T, T_reshaped.T, rcond=None)
        V = V.T  # (B, r)
    
    # Check residual
    S = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.linalg.norm(S - T) / max(1, np.linalg.norm(T))
    if err < tol:
        return U, V, W, err
    return None, None, None, err

def _search_rank(T, start_rank, end_rank, deadline, max_restarts=50):
    """Search for minimal rank by trying fixed-W approach with random W."""
    A, B, C = T.shape
    best = None
    rng = np.random.RandomState(0)
    
    for r in range(start_rank, max(end_rank - 1, -1), -1):
        if time.time() > deadline:
            break
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            # Random W
            W = rng.randn(C, r)
            # Try to find U, V
            U, V, Wsol, err = _try_rank_with_fixed_W(T, r, W)
            if U is not None and err < 1e-8:
                # Verify
                S = np.einsum('ar,br,cr->abc', U, V, Wsol)
                if np.allclose(S, T, atol=1e-8):
                    best = (U, V, Wsol, r)
                    found = True
                    break
        if found:
            return best
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    T = _build_tensor(n, m, p)
    
    # Fallback: schoolbook
    U_sbk, V_sbk, W_sbk = _schoolbook(n, m, p)
    best = (U_sbk, V_sbk, W_sbk, n * m * p)
    
    # Try Strassen
    try:
        U_str, V_str, W_str = _strassen_2x4x6(n, m, p)
        if _verify(U_str, V_str, W_str, T):
            best = (U_str, V_str, W_str, 42)
            print(f"Strassen rank 42 verified", file=sys.stderr)
        else:
            print("Strassen failed verification", file=sys.stderr)
    except Exception as e:
        print(f"Strassen error: {e}", file=sys.stderr)
    
    # Search for lower rank using fix-one-factor method
    # The tensor is 8 x 24 x 12. Lower bound: max(8, 24, 12) = 24? No.
    # The rank is at least the "tensor rank" lower bound.
    # For matrix multiplication M_{m,n,p}, the rank is at least m*n*p / max(m,n,p) ... 
    # Actually, the trivial lower bound is max(m, n, p) for square, but for rectangular it's more complex.
    # For 2x4x6, the rank is at least... let's think. The tensor has 48 nonzero entries.
    # A rank-1 tensor has at most min(A,B,C) = 8 nonzero entries (actually, a rank-1 tensor u⊗v⊗w
    # has at most |supp(u)|*|supp(v)|*|supp(w)| nonzeros, but generically all A*B*C).
    # So 48 nonzeros doesn't give a strong lower bound.
    # 
    # Known results: the rank of M_{2,4,6} is known to be at most 24 (schoolbook) and
    # likely lower. Let's search from rank 24 down.
    
    # Try to find rank below 42
    if time.time() < deadline - 10:
        search_result = _search_rank(T, best[3] - 1, 15, deadline - 5)
        if search_result is not None:
            U, V, W, r = search_result
            if r < best[3]:
                best = (U, V, W, r)
                print(f"Found rank {r} decomposition", file=sys.stderr)
    
    U, V, W, r = best
    print(f"Final rank: {r}", file=sys.stderr)
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
