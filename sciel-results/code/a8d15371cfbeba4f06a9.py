import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start = time.time()
    DEADLINE = 140.0

    n, m, p = DIMENSIONS

    def verify_fast(U, V, W):
        """Vectorized verification of tensor decomposition."""
        Um = np.array(U, dtype=float).reshape(n, m, -1)
        Vm = np.array(V, dtype=float).reshape(m, p, -1)
        Wm = np.array(W, dtype=float).reshape(p, n, -1)
        r = Um.shape[2]
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Use einsum for speed
        T = np.einsum('ijc,jkc,kic->ijk', Um, Vm, Wm)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    def build_strassen_rank42():
        """Build the rank-42 block Strassen decomposition."""
        rank_s = 42
        U_s = np.zeros((n * m, rank_s))
        V_s = np.zeros((m * p, rank_s))
        W_s = np.zeros((p * n, rank_s))

        subproblems = [
            (0, 0, 0, 0, 0),
            (0, 0, 0, 2, 2),
            (0, 0, 0, 4, 4),
            (0, 2, 2, 0, 0),
            (0, 2, 2, 2, 2),
            (0, 2, 2, 4, 4),
        ]

        U_strassen = [
            [1,0,0,1],
            [0,0,1,1],
            [1,0,0,0],
            [0,0,0,1],
            [1,1,0,0],
            [-1,0,1,0],
            [0,1,0,-1]
        ]
        V_strassen = [
            [1,0,0,1],
            [1,0,0,0],
            [0,1,0,-1],
            [-1,0,1,0],
            [0,0,0,1],
            [1,1,0,0],
            [0,1,1,1]
        ]
        W_strassen = [
            [1,0,0,1],
            [0,0,1,-1],
            [0,1,0,1],
            [1,0,1,0],
            [-1,1,0,0],
            [0,0,0,1],
            [1,0,0,1]
        ]

        for sp_idx, (a_i0, a_j0, b_j0, b_k0, c_k0) in enumerate(subproblems):
            for t in range(7):
                c = sp_idx * 7 + t
                u_vals = U_strassen[t]
                v_vals = V_strassen[t]
                w_vals = W_strassen[t]
                for li, lj in [(0,0), (0,1), (1,0), (1,1)]:
                    idx = li * 4 + (a_j0 + lj)
                    U_s[idx, c] = u_vals[li * 2 + lj]
                for lj, lk in [(0,0), (0,1), (1,0), (1,1)]:
                    idx = (b_j0 + lj) * 6 + (b_k0 + lk)
                    V_s[idx, c] = v_vals[lj * 2 + lk]
                for lk, li in [(0,0), (1,0), (0,1), (1,1)]:
                    idx = (c_k0 + lk) * 2 + li
                    W_s[idx, c] = w_vals[li * 2 + lk]

        return U_s, V_s, W_s

    def to_list(arr):
        return [[float(arr[i, c]) for c in range(arr.shape[1])] for i in range(arr.shape[0])]

    # Build baseline rank-42 solution
    U42, V42, W42 = build_strassen_rank42()
    best_rank = 42
    best_U, best_V, best_W = U42.copy(), V42.copy(), W42.copy()

    if not verify_fast(to_list(U42), to_list(V42), to_list(W42)):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        best_rank = rank_sb
        best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Structural approach: exploit the 2x4x6 rectangular structure
    # The key insight: since n=2 is small, we can use a "slice" approach
    # where we decompose along the i-dimension first.
    
    # Try a different family: use the fact that for 2x4x6, we can potentially
    # use a decomposition that shares structure across the two i-slices.
    
    # Approach: parameterize U, V, W with a structured low-rank form
    # and use gradient-free optimization (Nelder-Mead style) on a reduced set of params.
    
    def structured_search(rank_target, max_time=60.0):
        """Search for a rank-target decomposition using structured perturbation of Strassen."""
        if rank_target >= 42:
            return None
        
        U0 = U42[:, :rank_target].copy()
        V0 = V42[:, :rank_target].copy()
        W0 = W42[:, :rank_target].copy()
        
        # Try alternating optimization with vectorized solves
        def solve_W(Um, Vm, r):
            # A[k*n+i, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
            A = np.zeros((p * n, r))
            for c in range(r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            A[k * n + i, c] += Um[i * m + j, c] * Vm[j * p + k, c]
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A, np.ones(p * n), rcond=None)
                return W_flat.reshape(p * n, r)
            except:
                return None
        
        def solve_U(Vm, Wm, r):
            B = np.zeros((n * m, r))
            for c in range(r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            B[i * m + j, c] += Vm[j * p + k, c] * Wm[k * n + i, c]
            try:
                U_flat, _, _, _ = np.linalg.lstsq(B, np.ones(n * m), rcond=None)
                return U_flat.reshape(n * m, r)
            except:
                return None
        
        def solve_V(Um, Wm, r):
            C = np.zeros((m * p, r))
            for c in range(r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            C[j * p + k, c] += Um[i * m + j, c] * Wm[k * n + i, c]
            try:
                V_flat, _, _, _ = np.linalg.lstsq(C, np.ones(m * p), rcond=None)
                return V_flat.reshape(m * p, r)
            except:
                return None
        
        # Try multiple random initializations
        n_restarts = 8
        max_iters = 80
        
        for restart in range(n_restarts):
            if time.time() > DEADLINE - 20:
                return None
            
            if restart == 0:
                U_try = U0.copy()
                V_try = V0.copy()
                W_try = W0.copy()
            else:
                np.random.seed(12345 + restart * 7919)
                scale = 0.3
                U_try = U0 + np.random.randn(n * m, rank_target) * scale
                V_try = V0 + np.random.randn(m * p, rank_target) * scale
                W_try = W0 + np.random.randn(p * n, rank_target) * scale
            
            for it in range(max_iters):
                if time.time() > DEADLINE - 20:
                    return None
                
                W_new = solve_W(U_try, V_try, rank_target)
                if W_new is None:
                    break
                W_try = W_new
                
                U_new = solve_U(V_try, W_try, rank_target)
                if U_new is None:
                    break
                U_try = U_new
                
                V_new = solve_V(U_try, W_try, rank_target)
                if V_new is None:
                    break
                V_try = V_new
            
            # Check if we have a valid solution
            if verify_fast(to_list(U_try), to_list(V_try), to_list(W_try)):
                return (rank_target, U_try, V_try, W_try)
            
            # Try rounding to half-integers
            U_rounded = np.round(U_try * 2) / 2
            V_rounded = np.round(V_try * 2) / 2
            W_rounded = np.round(W_try * 2) / 2
            if verify_fast(to_list(U_rounded), to_list(V_rounded), to_list(W_rounded)):
                return (rank_target, U_rounded, V_rounded, W_rounded)
        
        return None

    # Try ranks from 35 down to 30 (more likely to succeed than 27)
    for r in [35, 34, 33, 32, 31, 30]:
        if time.time() > DEADLINE - 30:
            break
        
        result = structured_search(r, max_time=30.0)
        if result is not None:
            best_rank, best_U, best_V, best_W = result
            break
    
    # Also try a completely different structural approach:
    # Use the small n=2 to our advantage with a "dual" decomposition
    def try_dual_structural():
        """
        For 2x4x6, think of it as computing C[2,6] = A[2,4] @ B[4,6].
        We can write C[i,k] = sum_j A[i,j] B[j,k].
        
        A different structural idea: use the fact that the output has only
        12 elements. Try to find a decomposition where the rank is less
        than 42 by exploiting symmetries.
        """
        # This is a placeholder for a different structural approach
        # For now, try a few specific rank values with different initialization
        for r in [38, 36, 34]:
            if time.time() > DEADLINE - 15:
                break
            
            # Random initialization with different structure
            np.random.seed(999 + r)
            U_try = np.random.randn(n * m, r) * 0.2
            V_try = np.random.randn(m * p, r) * 0.2
            W_try = np.random.randn(p * n, r) * 0.2
            
            def solve_W(Um, Vm, rr):
                A = np.zeros((p * n, rr))
                for c in range(rr):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                A[k * n + i, c] += Um[i * m + j, c] * Vm[j * p + k, c]
                try:
                    W_flat, _, _, _ = np.linalg.lstsq(A, np.ones(p * n), rcond=None)
                    return W_flat.reshape(p * n, rr)
                except:
                    return None
            
            def solve_U(Vm, Wm, rr):
                B = np.zeros((n * m, rr))
                for c in range(rr):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                B[i * m + j, c] += Vm[j * p + k, c] * Wm[k * n + i, c]
                try:
                    U_flat, _, _, _ = np.linalg.lstsq(B, np.ones(n * m), rcond=None)
                    return U_flat.reshape(n * m, rr)
                except:
                    return None
            
            def solve_V(Um, Wm, rr):
                C = np.zeros((m * p, rr))
                for c in range(rr):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                C[j * p + k, c] += Um[i * m + j, c] * Wm[k * n + i, c]
                try:
                    V_flat, _, _, _ = np.linalg.lstsq(C, np.ones(m * p), rcond=None)
                    return V_flat.reshape(m * p, rr)
                except:
                    return None
            
            for it in range(100):
                if time.time() > DEADLINE - 15:
                    break
                
                W_new = solve_W(U_try, V_try, r)
                if W_new is None:
                    break
                W_try = W_new
                
                U_new = solve_U(V_try, W_try, r)
                if U_new is None:
                    break
                U_try = U_new
                
                V_new = solve_V(U_try, W_try, r)
                if V_new is None:
                    break
                V_try = V_new
            
            if verify_fast(to_list(U_try), to_list(V_try), to_list(W_try)):
                return (r, U_try, V_try, W_try)
            
            U_rounded = np.round(U_try * 2) / 2
            V_rounded = np.round(V_try * 2) / 2
            W_rounded = np.round(W_try * 2) / 2
            if verify_fast(to_list(U_rounded), to_list(V_rounded), to_list(W_rounded)):
                return (r, U_rounded, V_rounded, W_rounded)
        
        return None
    
    result2 = try_dual_structural()
    if result2 is not None and result2[0] < best_rank:
        best_rank, best_U, best_V, best_W = result2

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
