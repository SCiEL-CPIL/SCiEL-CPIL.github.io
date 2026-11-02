import numpy as np
import time
import random

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = 140.0

    # Number of constraints: n*p*n*m*m*p
    n_eq = n * p * n * m * m * p  # 648
    n_U = n * m  # 6
    n_V = m * p  # 18
    n_W = p * n  # 12

    def build_system(U, V, r):
        """Build A (n_eq x n_W*r) and b (n_eq,) such that A @ W_flat = b."""
        A = np.zeros((n_eq, n_W * r))
        b = np.zeros(n_eq)
        eq = 0
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b_idx in range(m):
                        for bp in range(m):
                            for c in range(p):
                                b[eq] = 1.0 if (a == i and b_idx == bp and c == k) else 0.0
                                for l in range(r):
                                    col = (i * p + k) * r + l
                                    A[eq, col] = U[a * m + b_idx, l] * V[bp * p + c, l]
                                eq += 1
        return A, b

    def verify(U, V, W, atol=1e-8):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b_idx in range(m):
                        for bp in range(m):
                            for c in range(p):
                                expected = 1.0 if (a == i and b_idx == bp and c == k) else 0.0
                                s = 0.0
                                for l in range(r):
                                    s += W[k * n + i, l] * U[a * m + b_idx, l] * V[bp * p + c, l]
                                if abs(s - expected) > atol:
                                    return False
        return True

    def verify_fast(U, V, W, atol=1e-8):
        """Vectorized verification."""
        r = U.shape[1]
        # Build the full constraint matrix
        A, b = build_system(U, V, r)
        W_flat = W.flatten()
        residual = np.linalg.norm(A @ W_flat - b)
        return residual < atol * np.sqrt(n_eq)

    # Schoolbook baseline (rank 36)
    r_sb = n * m * p
    U_sb = np.zeros((n_U, r_sb))
    V_sb = np.zeros((n_V, r_sb))
    W_sb = np.zeros((n_W, r_sb))
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U_sb[ii * m + jj, col] = 1
                V_sb[jj * p + kk, col] = 1
                W_sb[kk * n + ii, col] = 1
                col += 1

    best_rank = r_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    # Try rank 33, 34, 35 with random restarts + coordinate descent
    # Key insight: for fixed U,V, W is determined by least squares.
    # We search over U,V to minimize the residual.
    
    random.seed(42)
    np.random.seed(42)

    def solve_for_W(U, V, r):
        """Given U,V, solve for W via least squares."""
        A, b = build_system(U, V, r)
        W_flat, residuals, rank_mat, sv = np.linalg.lstsq(A, b, rcond=None)
        W = W_flat.reshape(n_W, r)
        if residuals and len(residuals) > 0:
            res = residuals[0]
        else:
            res = np.linalg.norm(A @ W_flat - b)
        return W, res

    def random_init_UV(r, scale=1.0):
        """Initialize U,V with structured randomness."""
        # Start from a perturbed schoolbook-like structure
        U = np.zeros((n_U, r))
        V = np.zeros((n_V, r))
        # Fill first min(r, n*m*p) columns with schoolbook pattern
        n_fill = min(r, n * m * p)
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    if col >= r:
                        break
                    U[ii * m + jj, col] = 1
                    V[jj * p + kk, col] = 1
                    col += 1
        # Add small random perturbations to remaining columns
        for l in range(col, r):
            U[:, l] = np.random.randn(n_U) * scale
            V[:, l] = np.random.randn(n_V) * scale
        return U, V

    def coordinate_descent(U, V, r, max_iter=50, tol=1e-10):
        """Iteratively refine U,V by fixing one and solving for the other."""
        best_res = float('inf')
        W, res = solve_for_W(U, V, r)
        best_res = res
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() - start_time > deadline - 30:
                break
            # Try updating a random subset of U columns
            for trial in range(3):
                # Update one column of U
                l = random.randint(0, r - 1)
                U_new = U.copy()
                U_new[:, l] = np.random.randn(n_U) * 0.5
                W_trial, res_trial = solve_for_W(U_new, V, r)
                if res_trial < best_res * (1 - 1e-6):
                    U = U_new
                    W = W_trial
                    best_res = res_trial
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                # Update one column of V
                l = random.randint(0, r - 1)
                V_new = V.copy()
                V_new[:, l] = np.random.randn(n_V) * 0.5
                W_trial, res_trial = solve_for_W(U, V_new, r)
                if res_trial < best_res * (1 - 1e-6):
                    V = V_new
                    W = W_trial
                    best_res = res_trial
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_res < tol:
                break
        
        return best_U, best_V, best_W, best_res

    # Try ranks from 33 upward
    for target_r in [33, 34, 35]:
        if time.time() - start_time > deadline - 40:
            break
        
        found = False
        n_restarts = 30 if target_r == 33 else 15
        
        for restart in range(n_restarts):
            if time.time() - start_time > deadline - 20:
                break
            
            # Different initialization strategies
            if restart == 0:
                # Structured: use schoolbook pattern for first r columns
                U, V = random_init_UV(target_r, scale=0.1)
            elif restart == 1:
                # Block structure: try to exploit 2x3x3 + 2x3x3
                U = np.zeros((n_U, target_r))
                V = np.zeros((n_V, target_r))
                # First 18 columns: schoolbook for k=0,1,2
                col = 0
                for ii in range(n):
                    for jj in range(m):
                        for kk in range(3):
                            if col >= target_r:
                                break
                            U[ii * m + jj, col] = 1
                            V[jj * p + kk, col] = 1
                            col += 1
                # Remaining: random
                for l in range(col, target_r):
                    U[:, l] = np.random.randn(n_U) * 0.3
                    V[:, l] = np.random.randn(n_V) * 0.3
            else:
                # Fully random
                U = np.random.randn(n_U, target_r) * 0.5
                V = np.random.randn(n_V, target_r) * 0.5
            
            # Coordinate descent
            U_opt, V_opt, W_opt, res_opt = coordinate_descent(U, V, target_r, max_iter=40)
            
            if res_opt < 1e-8:
                # Verify
                if verify(U_opt, V_opt, W_opt):
                    found = True
                    best_rank = target_r
                    best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                    break
            elif res_opt < best_res if 'best_res' in dir() else True:
                pass
        
        if found:
            break
    
    # Final verification
    if verify(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['W'][0])}")
