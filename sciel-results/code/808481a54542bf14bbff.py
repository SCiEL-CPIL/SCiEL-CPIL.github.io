import numpy as np
import time
import random
import sys
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                T[a, b, c] = 1.0
                nz_indices.append((a, b, c))
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    U_s, V_s, W_s = schoolbook()
    best_rank = U_s.shape[1]
    best_U, best_V, best_W = U_s, V_s, W_s

    # Deadline guard: 140s to leave 10s for overhead/serialization
    deadline = time.time() + 140

    def verify(U, V, W, tol=1e-10):
        """Check if the decomposition is exact, only checking non-zero entries."""
        for a, b, c in nz_indices:
            val = np.dot(U[a], W[c]) * V[b]  # This is wrong, let me fix
        # Correct: S[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        for a, b, c in nz_indices:
            val = 0.0
            for r in range(U.shape[1]):
                val += U[a, r] * V[b, r] * W[c, r]
            if abs(val - 1.0) > tol:
                return False
        # Also check that zero entries are zero (sample a few)
        # For speed, check a subset
        check_count = 0
        for _ in range(100):
            a = random.randint(0, nm-1)
            b = random.randint(0, mp-1)
            c = random.randint(0, pn-1)
            if (a, b, c) in nz_indices:
                continue
            val = 0.0
            for r in range(U.shape[1]):
                val += U[a, r] * V[b, r] * W[c, r]
            if abs(val) > tol:
                return False
            check_count += 1
        return True

    def verify_fast(U, V, W, tol=1e-10):
        """Fast verification using einsum, checking all entries."""
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(S, T, atol=tol)

    def round_and_verify(U, V, W):
        """
        Try to round U, V, W to integers or half-integers and verify.
        Returns (U_int, V_int, W_int) if successful, else None.
        """
        # Try integers first
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if verify_fast(U_int, V_int, W_int):
            return U_int, V_int, W_int
        
        # Try half-integers
        U_half = np.round(U * 2) / 2.0
        V_half = np.round(V * 2) / 2.0
        W_half = np.round(W * 2) / 2.0
        if verify_fast(U_half, V_half, W_half):
            return U_half, V_half, W_half
            
        # Try quarters (0.25) as a broader net
        U_q = np.round(U * 4) / 4.0
        V_q = np.round(V * 4) / 4.0
        W_q = np.round(W * 4) / 4.0
        if verify_fast(U_q, V_q, W_q):
            return U_q, V_q, W_q

        # Try eighths
        U_e = np.round(U * 8) / 8.0
        V_e = np.round(V * 8) / 8.0
        W_e = np.round(W * 8) / 8.0
        if verify_fast(U_e, V_e, W_e):
            return U_e, V_e, W_e

        return None

    def try_rank(R, max_attempts=5):
        """
        Attempt to find a decomposition of rank R.
        Uses structured initialization based on tensor structure.
        """
        if R >= best_rank:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Structured initialization: assign each non-zero entry to a random rank
            U_init = np.zeros((nm, R))
            V_init = np.zeros((mp, R))
            W_init = np.zeros((pn, R))
            
            # Assign each (a,b,c) to a random rank r, then set U[a,r], V[b,r], W[c,r] to contribute
            # Use a greedy assignment
            assignments = {}
            for a, b, c in nz_indices:
                r = random.randint(0, R-1)
                if r not in assignments:
                    assignments[r] = []
                assignments[r].append((a, b, c))
            
            # For each rank, compute the required values
            for r, entries in assignments.items():
                if not entries:
                    continue
                # We need sum over entries of U[a,r]*V[b,r]*W[c,r] to work out
                # Simple heuristic: set all to 1/sqrt(len(entries))
                k = len(entries)
                scale = 1.0 / (k ** 0.5)
                for a, b, c in entries:
                    U_init[a, r] += scale
                    V_init[b, r] += scale
                    W_init[c, r] += scale
            
            # Add small random perturbation
            U_init += np.random.randn(nm, R) * 0.1
            V_init += np.random.randn(mp, R) * 0.1
            W_init += np.random.randn(pn, R) * 0.1
            
            def objective(params):
                Ur = params[:nm*R].reshape(nm, R)
                Vr = params[nm*R:nm*R+mp*R].reshape(mp, R)
                Wr = params[nm*R+mp*R:].reshape(pn, R)
                
                # Only compute error on non-zero entries for speed
                err = 0.0
                for a, b, c in nz_indices:
                    val = np.dot(Ur[a], Vr[b])  # This is wrong
                    # S[a,b,c] = sum_r Ur[a,r]*Vr[b,r]*Wr[c,r]
                    # Can't compute this efficiently in a loop like this
                    pass
                
                # Use einsum for correctness, it's fast enough
                S = np.einsum('ar,br,cr->abc', Ur, Vr, Wr)
                diff = S - T
                return np.sum(diff * diff)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use L-BFGS-B
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                               options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 10000})
                
                if res.fun < 1e-6:
                    Ur = res.x[:nm*R].reshape(nm, R)
                    Vr = res.x[nm*R:nm*R+mp*R].reshape(mp, R)
                    Wr = res.x[nm*R+mp*R:].reshape(pn, R)
                    
                    # Try to round
                    result = round_and_verify(Ur, Vr, Wr)
                    if result:
                        return result
                        
            except Exception:
                pass
                
        return None

    # Try to reduce rank
    for R in range(best_rank - 1, 69, -1):
        if time.time() > deadline:
            break
            
        res = try_rank(R, max_attempts=3)
        
        if res is not None:
            Ur, Vr, Wr = res
            best_rank = R
            best_U, best_V, best_W = Ur, Vr, Wr

    def to_list(M):
        L = []
        for row in M:
            L.append([float(x) for x in row])
        return L

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
