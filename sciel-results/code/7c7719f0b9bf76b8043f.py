"""Tensor decomposition for 2x4x5 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 5
    t0 = time.time()
    deadline = t0 + 140.0

    # Schoolbook rank-40 decomposition
    R0 = n * m * p  # 40
    U0 = np.zeros((n * m, R0))
    V0 = np.zeros((m * p, R0))
    W0 = np.zeros((p * n, R0))
    for c, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
        U0[i * m + j, c] = 1.0
        V0[j * p + k, c] = 1.0
        W0[k * n + i, c] = 1.0

    def tensor_vec(u, v, w):
        """Flatten outer product u ⊗ v ⊗ w."""
        return np.einsum('a,b,c->abc', u, v, w).reshape(-1)

    def check(U, V, W):
        """Verify exact decomposition. Returns True if correct."""
        r = U.shape[1]
        # Reconstruct tensor
        T = np.zeros((n * m, m * p, p * n))
        for c in range(r):
            T += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        # Check against schoolbook tensor
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1.0
        return np.allclose(T, T_target, atol=1e-9)

    def round_half(X):
        return np.round(2.0 * X) * 0.5

    # Verify schoolbook
    assert check(U0, V0, W0)

    # Strategy: Use algebraic structure. For 2x4x5, we can try to
    # find rank-39 by identifying if any schoolbook term is "redundant"
    # given the others with small adjustments.
    #
    # New structural approach: Systematic search over pairs of terms to merge
    # using the identity that for specific index combinations, we can use
    # (a+b)(c+d) = ac + ad + bc + bd - (a+b-c-d)*something...
    #
    # Actually, let's use a completely different method: 
    # Integer/half-integer search via structured random restarts with
    # much better initialization and local search.

    best_rank = R0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    # Try rank 39: remove one term and adjust
    # For each term to remove, set up the correction as a rank-1 tensor
    # that compensates for its removal.
    
    # More systematic: For rank r < 40, we need r rank-1 tensors.
    # Use the "one-at-a-time" removal with continuous relaxation.
    
    # Approach: Start with schoolbook. For each candidate removal,
    # solve for a correction tensor (sum of rank-1 terms from remaining)
    # that cancels the removed term's contribution on the support.
    
    # Since the tensor is sparse (only 40 ones), removing term (i,j,k)
    # means we need: sum_{c != c0} u_c v_c w_c = T_target - u0 v0 w0
    # where the RHS has -1 at (i,j,k) and 1 at all other 39 positions.
    
    # This is still a rank-39 tensor to decompose. Not helpful directly.
    
    # Let's try: Can we find a rank-1 tensor that equals the difference
    # between two schoolbook terms? Unlikely, but let's check structured cases.
    
    # Structural approach: Use the fact that 2*4*5 = 40, and known
    # lower bounds. For 2xn x nxp, the rank is at most 2*n*p - n - p + 1
    # via certain methods? Let's try a different decomposition.
    
    # Block decomposition: 2x4 x 4x5
    # Write A = [A1 | A2] where A1 is 2x2, A2 is 2x2
    # B = [B1; B2] where B1 is 2x5, B2 is 2x5
    # C = A1*B1 + A2*B2  (2x5 result)
    # 
    # A1*B1: 2x2 * 2x5 -> use 4 scalar mults (schoolbook)
    # A2*B2: 2x2 * 2x5 -> use 4 scalar mults (schoolbook)
    # Total: 8 mults... but that's for the block structure, not helpful.
    
    # Let me try: Can we use Laderman for the 2x2 * 2x2 subproblems?
    # 2x2 * 2x2 has rank 7 (Laderman). 
    # 2x2 * 2x5 = 2x2 * [2x2 | 2x1]
    # = (2x2*2x2) + (2x2*2x1)
    # Laderman for 2x2*2x2: rank 7
    # 2x2*2x1: 4 scalar mults (rank 4) or can we do better?
    # 2x2 * 2x1 is just 4 dot products, rank 4 (each is a bilinear form)
    # Actually 2x2*2x1: result is 2x1. Each entry is a bilinear form.
    # C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]  (2 bilinear forms combined? No, 4 mults)
    # Rank of 2x2*2x1 tensor = 4 (schoolbook, and it's actually 4)
    
    # So: A1*B1 where A1 is 2x2, B1 is 2x5 = [B1a | B1b] with B1a 2x2, B1b 2x1
    # A1*B1a: 2x2*2x2, rank 7 (Laderman)
    # A1*B1b: 2x2*2x1, rank 4
    # Total for A1*B1: 11
    
    # Similarly A2*B2: 11
    # Total: 22? That seems too good. Let me verify Laderman.
    
    # Laderman's 2x2*2x2 decomposition (rank 7):
    # Given X = [[x11, x12], [x21, x22]], Y = [[y11, y12], [y21, y22]]
    # Z = X*Y
    # m1 = (x11+x22)*(y11+y22)
    # m2 = (x12-x22)*(y21+y22)
    # m3 = (x11+x12)*y21
    # m4 = (x21+x22)*y12
    # m5 = (x11-x21)*(y11+y21)
    # m6 = (x12-x22)*(y12+y22)
    # m7 = x11*y11
    # Z11 = m1 - m2 + m5 - m6 + m7
    # Z12 = m3 - m5 + m6 + m7  ... wait let me get this right
    
    # Standard Laderman:
    # m1 = x11*y11
    # m2 = x11*y12 + x12*y11 + x12*y12
    # m3 = x21*y11 + x21*y12 + x22*y12
    # m4 = x22*y22
    # m5 = (x11+x22)*(y11+y22)
    # m6 = (x12-x22)*(y21+y22)
    # m7 = (x11+x12)*y21
    # Z11 = m5 - m6 + m7 - m3 + m1  ... I'm getting confused.
    
    # Let me just implement the known rank-7 decomposition directly.
    
    if time.time() > deadline:
        return {"U": round_half(U0).tolist(), "V": round_half(V0).tolist(), "W": round_half(W0).tolist()}

    # I'll implement a concrete rank-39 attempt using structured modifications.
    # Given the complexity, let me try a different structural approach:
    # Use numpy to solve a system of equations for a rank-39 decomposition.
    
    # For rank 39, we have 39 * (8 + 20 + 10) = 39 * 38 = 1482 unknowns
    # and 1600 equations (but only 40 are nonzero, so effectively 40 equations 
    # that must equal 1 and 1560 that must equal 0).
    
    # This is underdetermined (1482 < 1600), so we need to use structure.
    
    # Let me try: fix 38 of the 40 schoolbook terms, and solve for the 
    # 39th term plus small adjustments to the 38 fixed terms.
    
    # Actually, the cleanest structural change: use scipy to do a 
    # least-squares fit for rank-39 with a specific parameterization.
    
    try:
        from scipy.optimize import least_squares
        has_scipy = True
    except ImportError:
        has_scipy = False
    
    if has_scipy and time.time() < deadline:
        # Parameterize: start from schoolbook rank-40, remove column 0,
        # add 39 columns initialized as schoolbook[1:], then optimize.
        
        r = 39
        U_init = U0[:, 1:].copy()  # 8 x 39
        V_init = V0[:, 1:].copy()  # 20 x 39
        W_init = W0[:, 1:].copy()  # 10 x 39
        
        def residuals(params):
            # params: flattened U (8*39), V (20*39), W (10*39)
            sz_u, sz_v = n * m * r, m * p * r
            Uf = params[:sz_u].reshape(n * m, r)
            Vf = params[sz_u:sz_u + sz_v].reshape(m * p, r)
            Wf = params[sz_u + sz_v:].reshape(p * n, r)
            
            T = np.zeros((n * m, m * p, p * n))
            for c in range(r):
                T += np.einsum('a,b,c->abc', Uf[:, c], Vf[:, c], Wf[:, c])
            
            Tt = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Tt[i * m + j, j * p + k, k * n + i] = 1.0
            
            return (T - Tt).flatten()
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        result = least_squares(residuals, x0, method='lm', max_nfev=5000,
                               xtol=1e-12, ftol=1e-12)
        
        if result.success or np.max(np.abs(result.fun)) < 1e-8:
            sz_u, sz_v = n * m * r, m * p * r
            Uf = result.x[:sz_u].reshape(n * m, r)
            Vf = result.x[sz_u:sz_u + sz_v].reshape(m * p, r)
            Wf = result.x[sz_u + sz_v:].reshape(p * n, r)
            
            if check(Uf, Vf, Wf):
                best_rank = r
                best_U, best_V, best_W = Uf, Vf, Wf
                break
    
    if time.time() > deadline:
        return {"U": round_half(best_U).tolist(), "V": round_half(best_V).tolist(), "W": round_half(best_W).tolist()}

    # Try rank 38 similarly
    if best_rank == 40 and has_scipy and time.time() < deadline:
        r = 38
        U_init = U0[:, 2:].copy()
        V_init = V0[:, 2:].copy()
        W_init = W0[:, 2:].copy()
        
        def residuals2(params):
            sz_u, sz_v = n * m * r, m * p * r
            Uf = params[:sz_u].reshape(n * m, r)
            Vf = params[sz_u:sz_u + sz_v].reshape(m * p, r)
            Wf = params[sz_u + sz_v:].reshape(p * n, r)
            
            T = np.zeros((n * m, m * p, p * n))
            for c in range(r):
                T += np.einsum('a,b,c->abc', Uf[:, c], Vf[:, c], Wf[:, c])
            
            Tt = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Tt[i * m + j, j * p + k, k * n + i] = 1.0
            
            return (T - Tt).flatten()
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        result = least_squares(residuals2, x0, method='lm', max_nfev=5000,
                               xtol=1e-12, ftol=1e-12)
        
        if result.success or np.max(np.abs(result.fun)) < 1e-8:
            sz_u, sz_v = n * m * r, m * p * r
            Uf = result.x[:sz_u].reshape(n * m, r)
            Vf = result.x[sz_u:sz_u + sz_v].reshape(m * p, r)
            Wf = result.x[sz_u + sz_v:].reshape(p * n, r)
            
            if check(Uf, Vf, Wf):
                best_rank = r
                best_U, best_V, best_W = Uf, Vf, Wf
    
    # Round to half-integers
    best_U = round_half(best_U)
    best_V = round_half(best_V)
    best_W = round_half(best_W)
    
    if not check(best_U, best_V, best_W):
        best_U = np.round(2.0 * U0) * 0.5
        best_V = np.round(2.0 * V0) * 0.5
        best_W = np.round(2.0 * W0) * 0.5
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
