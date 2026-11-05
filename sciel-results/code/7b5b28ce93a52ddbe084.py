"""
Strategy: 
1. Base: 44-term decomposition (4 Strassen blocks for 2x2x8 + schoolbook for 2x1x8).
2. Operator-valued tensor approach: Model the multiplication as a linear operator
   on the 16-dim output space, decompose into Kronecker products, and search for
   a common low-rank basis to reduce tensor rank below 44.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Build 44-term decomposition
    rank_44 = 44
    U_44 = [[0] * rank_44 for _ in range(nm)]
    V_44 = [[0] * rank_44 for _ in range(mp)]
    W_44 = [[0] * rank_44 for _ in range(pn)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U_44[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V_44[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W_44[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    for i in range(n):
        for k in range(p):
            U_44[i * m + 2][col] = 1
            V_44[2 * p + k][col] = 1
            W_44[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Verify 44-term
    candidate_44 = {"U": U_44, "V": V_44, "W": W_44}
    if not _validate(candidate_44):
        return _schoolbook()

    best_rank = 44
    best_candidate = candidate_44

    # ===== RANK REDUCTION VIA SUBSPACE PROJECTION =====
    # The tensor T[i,a,k] = 1 for all (i,a,k) must be decomposed as:
    # T[i,a,k] = sum_c U[i*m+a, c] * V[a*p+k, c] * W[k*n+i, c]
    #
    # This is equivalent to: for each (i,a,k), the bilinear form equals 1.
    # We can model this as a constrained optimization:
    #   Minimize rank subject to: for all (i,a,k), sum_c U_ia,c * V_ak,c * W_ki,c = 1
    #
    # Key insight: The constraint matrix has structure. For fixed (a,k), the
    # constraint is sum_c U[i*m+a,c] * V[a*p+k,c] * W[k*n+i,c] = 1 for all i.
    # This means: for each (a,k), the vector [V[a*p+k,c] * W[k*n+i,c]]_c must
    # satisfy: sum_i U[i*m+a,c] * (V[a*p+k,c] * W[k*n+i,c]) = 1 for all i... 
    # Wait, that's not quite right. Let me re-examine.
    #
    # Actually, the constraint is: for each triple (i,a,k):
    #   sum_c U[i*m+a, c] * V[a*p+k, c] * W[k*n+i, c] = 1
    #
    # Let's define for each c:
    #   u_c = (U[0,c], U[1,c], ..., U[5,c])  (length 6)
    #   v_c = (V[0,c], V[1,c], ..., V[23,c])  (length 24)  
    #   w_c = (W[0,c], W[1,c], ..., W[15,c])  (length 16)
    #
    # The constraint for (i,a,k) is: u_c[i*m+a] * v_c[a*p+k] * w_c[k*n+i] summed over c = 1
    #
    # This is a system of 48 nonlinear equations in 48*R variables (R = rank).
    #
    # Alternative approach: Use the fact that the all-ones tensor has special structure.
    # The tensor T = ones(n,m,p) can be written as sum of rank-1 tensors.
    # 
    # For the all-ones tensor, we can use the following identity:
    #   1 = sum_{r=1}^{R} prod of factors
    #
    # Let's try a direct construction using the fact that:
    #   For any R, if we can find vectors u_r in R^n, v_r in R^m, w_r in R^p
    #   such that for all (i,a,k): sum_r u_r[i] * v_r[a] * w_r[k] = 1
    #
    # This is equivalent to: the all-ones tensor has rank R.
    #
    # Known results: The rank of the all-ones tensor [n,m,p] is at most n*m + n*p + m*p - n - m - p + 1
    # (by inclusion-exclusion / combinatorial argument).
    #
    # For [2,3,8]: upper bound = 2*3 + 2*8 + 3*8 - 2 - 3 - 8 + 1 = 6 + 16 + 24 - 13 + 1 = 34
    #
    # Let's try to construct a rank-34 decomposition!
    #
    # The construction: Use the fact that 1 = (sum_i alpha_i)(sum_a beta_a)(sum_k gamma_k)
    # is not quite right. Let me think differently.
    #
    # Actually, for the all-ones tensor, we can use:
    #   T[i,a,k] = 1 = sum_{S subset of {i,a,k}} (-1)^{|S|} * something
    #
    # Better approach: Use the polynomial identity.
    # Consider: 1 = (x_1 + ... + x_n)(y_1 + ... + y_m)(z_1 + ... + z_p) evaluated at x_i=y_a=z_k=1
    #
    # The rank of the all-ones tensor [n,m,p] is actually min(n,m,p) if we allow
    # arbitrary entries, because:
    #   T[i,a,k] = 1 = sum_{r=1}^{min(n,m,p)} (1/min(n,m,p)) * 1 * 1 * 1
    # No, that gives rank 1 only if all entries are 1/min, which they aren't.
    #
    # Let me just try numerical optimization to find a low-rank decomposition.
    
    # Numerical approach: Start with random initialization and use gradient descent
    # or alternating least squares to minimize the constraint violation.
    
    def _try_rank_reduction(target_rank, max_iter=500):
        """Try to find a decomposition of rank target_rank."""
        if not _elapsed_ok():
            return None
        
        # Initialize with small random values
        np.random.seed(42)
        U_num = np.random.randn(nm, target_rank) * 0.1
        V_num = np.random.randn(mp, target_rank) * 0.1
        W_num = np.random.randn(pn, target_rank) * 0.1
        
        # Build constraint matrix: for each (i,a,k), we need sum_c U_ia,c * V_ak,c * W_ki,c = 1
        # This is nonlinear. Use alternating minimization:
        # Fix V,W and solve for U (linear in U for each (i,a,k))
        # Fix U,W and solve for V
        # Fix U,V and solve for W
        
        for iteration in range(max_iter):
            if not _elapsed_ok():
                return None
            
            # Update U: for each i, solve linear system
            # For fixed a,k: sum_c U[i*m+a,c] * (V[a*p+k,c] * W[k*n+i,c]) = 1
            # This is: U[i*m+a,:] @ (V[a*p+k,:] * W[k*n+i,:]) = 1
            # But this couples across (a,k) for the same i.
            #
            # Actually, for each i, we have constraints for all (a,k):
            #   sum_c U[i*m+a,c] * V[a*p+k,c] * W[k*n+i,c] = 1 for all a,k
            #
            # This is a system of m*p equations in n*m unknowns (per i, that's m unknowns per a).
            # Wait, U has nm rows, so for fixed i, we have m rows of U to determine.
            # Each row has target_rank elements, so m*target_rank unknowns.
            # And we have m*p equations.
            #
            # This is underdetermined if m*target_rank > m*p, i.e., target_rank > p.
            # For target_rank=34 and p=8, we have 3*34=102 unknowns and 3*8=24 equations.
            # Underdetermined! So we can use least squares.
            
            # Build the linear system for U
            # For each i, a: U[i*m+a,:] @ diag(V[a*p+k,:]*W[k*n+i,:]) summed over k... 
            # No, let me re-index.
            #
            # Constraint: for (i,a,k): sum_c U[i*m+a,c] * V[a*p+k,c] * W[k*n+i,c] = 1
            #
            # For fixed (i,a), this is a vector equation over k:
            #   U[i*m+a,:] @ (V[a*p:,:] * W[k*n+i,:] for each k) ... 
            #
            # Let me define M[i,a] as a matrix of shape (p, target_rank) where
            # M[i,a][k,c] = V[a*p+k,c] * W[k*n+i,c]
            # Then the constraint is: U[i*m+a,:] @ M[i,a].T = ones(p)
            # i.e., M[i,a] @ U[i*m+a,:].T = ones(p)
            
            for i in range(n):
                for a in range(m):
                    # Build M[i,a] of shape (p, target_rank)
                    M_ia = np.zeros((p, target_rank))
                    for k in range(p):
                        M_ia[k, :] = V_num[a*p+k, :] * W_num[k*n+i, :]
                    # Solve M_ia @ x = ones(p) for x = U[i*m+a, :]
                    if M_ia.shape[0] >= M_ia.shape[1]:
                        # Overdetermined or square: use least squares
                        U_num[i*m+a, :], residuals, rank_M, sv = np.linalg.lstsq(M_ia, np.ones(p), rcond=None)
                    else:
                        # Underdetermined: use pseudoinverse
                        U_num[i*m+a, :] = np.linalg.pinv(M_ia) @ np.ones(p)
            
            # Update V: for each a, solve for V[a*p+k,:] for all k
            # Constraint: for (i,a,k): sum_c U[i*m+a,c] * V[a*p+k,c] * W[k*n+i,c] = 1
            # For fixed a: sum_k... no, for each (i,k): 
            #   V[a*p+k,:] @ (U[i*m+a,:] * W[k*n+i,:]) = 1 for all i
            #
            # For fixed a, define N_a as a matrix of shape (n*p, target_rank) where
            # N_a[i*p+k, c] = U[i*m+a,c] * W[k*n+i,c]
            # Then the constraint is: N_a @ V[a*p:,:].T = ones(n*p)
            # i.e., for each (i,k): V[a*p+k,:] @ N_a[i*p+k,:] = 1
            
            for a in range(m):
                N_a = np.zeros((n*p, target_rank))
                for i in range(n):
                    for k in range(p):
                        N_a[i*p+k, :] = U_num[i*m+a, :] * W_num[k*n+i, :]
                # Solve N_a @ X = ones(n*p) where X has shape (target_rank, n*p)
                # X[:, i*p+k] = V[a*p+k, :]
                if N_a.shape[0] >= N_a.shape[1]:
                    X, residuals, rank_N, sv = np.linalg.lstsq(N_a, np.ones(n*p), rcond=None)
                else:
                    X = np.linalg.pinv(N_a) @ np.ones(n*p)
                for i in range(n):
                    for k in range(p):
                        V_num[a*p+k, :] = X[:, i*p+k]
            
            # Update W: for each k, solve for W[k*n+i,:] for all i
            # Constraint: for (i,a,k): sum_c U[i*m+a,c] * V[a*p+k,c] * W[k*n+i,c] = 1
            # For fixed k: for each (i,a):
            #   W[k*n+i,:] @ (U[i*m+a,:] * V[a*p+k,:]) = 1 for all a
            #
            # For fixed k, define P_k as a matrix of shape (n*m, target_rank) where
            # P_k[i*m+a, c] = U[i*m+a,c] * V[a*p+k,c]
            # Then: P_k @ W[k*n:,:].T = ones(n*m)
            
            for k in range(p):
                P_k = np.zeros((n*m, target_rank))
                for i in range(n):
                    for a in range(m):
                        P_k[i*m+a, :] = U_num[i*m+a, :] * V_num[a*p+k, :]
                if P_k.shape[0] >= P_k.shape[1]:
                    Y, residuals, rank_P, sv = np.linalg.lstsq(P_k, np.ones(n*m), rcond=None)
                else:
                    Y = np.linalg.pinv(P_k) @ np.ones(n*m)
                for i in range(n):
                    for a in range(m):
                        W_num[k*n+i, :] = Y[:, i*m+a]
            
            # Check convergence: compute max constraint violation
            max_violation = 0.0
            for i in range(n):
                for a in range(m):
                    for k in range(p):
                        val = np.dot(U_num[i*m+a, :], V_num[a*p+k, :] * W_num[k*n+i, :])
                        violation = abs(val - 1.0)
                        if violation > max_violation:
                            max_violation = violation
            
            if max_violation < 1e-10:
                # Found a solution! Now try to round to half-integers.
                U_rounded = np.round(U_num * 2) / 2
                V_rounded = np.round(V_num * 2) / 2
                W_rounded = np.round(W_num * 2) / 2
                
                # Check if rounded solution still satisfies constraints
                max_violation_rounded = 0.0
                for i in range(n):
                    for a in range(m):
                        for k in range(p):
                            val = np.dot(U_rounded[i*m+a, :], V_rounded[a*p+k, :] * W_rounded[k*n+i, :])
                            violation = abs(val - 1.0)
                            if violation > max_violation_rounded:
                                max_violation_rounded = violation
                
                if max_violation_rounded < 1e-8:
                    # Convert to lists of integers/half-integers
                    U_list = [[int(round(2*x))/2 for x in U_rounded[row, :]] for row in range(nm)]
                    V_list = [[int(round(2*x))/2 for x in V_rounded[row, :]] for row in range(mp)]
                    W_list = [[int(round(2*x))/2 for x in W_rounded[row, :]] for row in range(pn)]
                    
                    candidate = {"U": U_list, "V": V_list, "W": W_list}
                    if _validate(candidate):
                        return candidate
                    else:
                        # Try with the unrounded values converted to fractions
                        # This won't work if they're not half-integers
                        return None
                else:
                    # Rounding failed, try to use the floating point values
                    # But the task requires integer/half-integer entries
                    return None
            
            if iteration % 50 == 0 and not _elapsed_ok():
                return None
        
        return None
    
    # Try to find lower-rank decompositions
    for target_rank in range(34, 44):
        if not _elapsed_ok():
            break
        result = _try_rank_reduction(target_rank)
        if result is not None:
            best_rank = target_rank
            best_candidate = result
            break  # Found the lowest rank, stop
    
    if not _elapsed_ok():
        return best_candidate
    
    return best_candidate

# EVOLVE-BLOCK-END
