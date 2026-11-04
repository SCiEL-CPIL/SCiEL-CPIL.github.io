import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = 4, 5, 5
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Build the matrix multiplication tensor T[u,v,w]
    # u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Schoolbook fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1
                    V[v, col] = 1
                    W[w, col] = 1
                    col += 1
        return U, V, W

    def check(U, V, W, tol=1e-9):
        Tc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
        return np.linalg.norm(Tc - T) < tol

    def snap_half(U, V, W):
        return np.round(U * 2) / 2.0, np.round(V * 2) / 2.0, np.round(W * 2) / 2.0

    def snap_int(U, V, W):
        return np.round(U), np.round(V), np.round(W)

    def rational_reconstruct(vals, max_den=64):
        out = np.zeros_like(vals)
        for idx, v in enumerate(vals.flat):
            best = None
            bestd = float('inf')
            for den in range(1, max_den + 1):
                num = int(round(v * den))
                fv = num / den
                d = abs(fv - v)
                if d < bestd:
                    bestd = d
                    best = fv
            out.flat[idx] = best
        return out

    # ---- Constructive block-Strassen for 4x5x5 ----
    # Split the 5 columns/rows into 3 + 2.
    # For the 3x3 part, use a rank-7 Strassen-like scheme.
    # For the 2x2 part, use rank-4 schoolbook (or rank 3 if we can).
    # The 4 rows of A: split into 2+2.
    # This gives a structured decomposition.

    # Let's try to build a rank-60 decomposition using block structure:
    # A is 4x5, B is 5x5, C is 4x5.
    # Split columns of A and rows of B: 3+2.
    # A = [A1 (4x3) | A2 (4x2)]
    # B = [B1 (3x5); B2 (2x5)]
    # C = A1*B1 + A2*B2
    # 
    # For A1*B1: 4x3 * 3x5. Use Strassen-like on the 3x3 core? 
    # Actually, let's use a simpler approach: 
    # Multiply 4x3 by 3x5. We can think of this as 4 independent 1x3 times 3x5.
    # Each 1x3 * 3x5 = 3x5 result. Rank 3 per row? No, that's schoolbook.
    #
    # Let's try a different constructive approach: 
    # Use the fact that 4x5x5 can be embedded in 5x5x5.
    # The rank of 5x5x5 is at most 5^3 = 125, but we know it's lower.
    # Actually, let's just try to find a rank-80 decomposition by 
    # removing redundant terms from schoolbook using linear dependencies.

    # New strategy: Use a "peasant" multiplication with cancellations.
    # For each output entry C[i,k], it's a dot product of row i of A and col k of B.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can group the j's. For 5 terms, we can use a scheme with fewer multiplications.
    # For example, for 2 terms: a*c + b*d = (a+b)(c+d) - a*c - b*d. That's 4 mults for 2 terms, worse.
    # Actually, (a+b)(c+d) = ac + ad + bc + bd. So 1 mult gives 4 terms.
    # For 5 terms, we can use a tree: 
    # (a+b+c)(d+e+f) = ad+ae+af+bd+be+bf+cd+ce+cf. That's 9 terms from 1 mult.
    # But we only need 5 specific terms.
    #
    # Let's try a different angle: 
    # The tensor has a lot of structure. Let's use a "slicing" approach.
    # Slice by j (the summation index).
    # For each j, we have a rank-1 tensor: A[:,j] * B[j,:] * I (output identity).
    # The total tensor is sum_j A[:,j] * B[j,:] * I.
    # This is the schoolbook decomposition with rank 5*4*5 = 100? No, rank 5 (for j) * 4 * 5? 
    # Actually, the schoolbook rank is n*m*p = 100.
    #
    # Let's try to use a "Strassen" on the j-index.
    # Group j into 3+2.
    # For the 3 j's: use a rank-7 scheme for 3x3x3? But our dimensions are 4x3x5 and 4x2x5.
    #
    # I think the most promising constructive approach is:
    # 1. Start with schoolbook (rank 100).
    # 2. Look for linear dependencies among the 100 rank-1 terms.
    # 3. If we can find 10 dependencies, we get rank 90.
    #
    # Let's try to find a rank-90 decomposition by solving a linear system.

    # Build the schoolbook decomposition
    U_sb, V_sb, W_sb = schoolbook()
    r_sb = n * m * p  # 100

    # The tensor T = sum_t U[:,t] * V[:,t] * W[:,t]
    # We want to find a linear combination of the 100 terms that sums to zero.
    # i.e., find c in R^100 such that sum_t c_t * U[:,t] * V[:,t] * W[:,t] = 0
    # This is a linear system in c.
    # The number of equations is n*m*p*n = 4*5*5*5 = 500? No, the tensor has n*m*p*n = 500 entries? 
    # T has shape (20, 25, 20) = 10000 entries.
    # The linear system is: for each (u,v,w), sum_t c_t * U[u,t]*V[v,t]*W[w,t] = T[u,v,w]
    # But we want sum_t c_t * U[u,t]*V[v,t]*W[w,t] = 0 for all (u,v,w).
    # This is a homogeneous system with 10000 equations and 100 unknowns.
    # It likely has only the trivial solution.
    #
    # Instead, let's try to find a rank-90 decomposition by 
    # replacing 10 of the schoolbook terms with a combination of the other 90.
    # This is equivalent to finding a rank-1 tensor in the span of the other 90.
    #
    # Let's try a more direct approach: 
    # Use a "structured" initialization based on the block decomposition
    # and then refine with a discrete coordinate descent.

    # Block decomposition:
    # Split j into 3+2.
    # A1 = A[:, :3] (4x3), A2 = A[:, 3:] (4x2)
    # B1 = B[:3, :] (3x5), B2 = B[3:, :] (2x5)
    # C = A1*B1 + A2*B2
    #
    # For A1*B1: 4x3 * 3x5. 
    # We can use a scheme with rank 4*3*5 = 60 (schoolbook) or less.
    # For A2*B2: 4x2 * 2x5. Rank 4*2*5 = 40 (schoolbook) or less.
    # Total schoolbook: 100.
    #
    # Let's try to use a Strassen-like scheme for the 3x3 part of A1*B1.
    # A1*B1 = sum_{j=0}^{2} A1[:,j] * B1[j,:]
    # This is a 4x3x5 tensor. We can view it as 4 independent 1x3x5 tensors.
    # For each row i, C1[i,:] = A1[i,:] * B1.
    # A1[i,:] is 1x3, B1 is 3x5. Result is 1x5.
    # This is a 3-term dot product. Rank 3 per row, total rank 4*3 = 12 for A1*B1? No, that's not right.
    # The rank of the 4x3x5 tensor is at most 4*3*5 = 60, but likely less.
    #
    # I think the best constructive approach is to use a "Kronecker" product structure.
    # 4x5x5 = (2x2) * (2x5x5) + ... no, that doesn't work.
    #
    # Let's try a completely different constructive approach:
    # Use the fact that the tensor is "commutative" in some sense.
    # T[i,j,k] = 1 if the indices match.
    # We can use a "Fourier" basis to diagonalize the tensor.
    # But the tensor is not a matrix, it's a 3D array.
    #
    # I'll try a "greedy" approach: 
    # Start with rank 100, and try to reduce by 1 at a time.
    # For each rank r from 90 to 100, try to find a rank-r decomposition
    # by solving a least-squares problem with r variables.

    # Let's implement a "rank reduction" approach.
    # Start with schoolbook (rank 100).
    # Try to find a rank-99 decomposition by setting one of the 100 terms to be a 
    # linear combination of the other 99.
    # This is equivalent to finding a vector c in R^99 such that
    # sum_{t=0}^{98} c_t * U[:,t]*V[:,t]*W[:,t] = U[:,99]*V[:,99]*W[:,99]
    # This is a linear system with 10000 equations and 99 unknowns.
    # It likely has no solution, but let's try.

    def try_rank_reduction(r_target):
        """Try to find a rank-r_target decomposition by reducing from schoolbook."""
        if time.time() > deadline - 10:
            return None
        
        # Start with schoolbook
        U = U_sb[:, :r_target].copy()
        V = V_sb[:, :r_target].copy()
        W = W_sb[:, :r_target].copy()
        
        # The remaining terms are from r_target to 100.
        # We want to express them as linear combinations of the first r_target terms.
        # This is a linear system:
        # For each (u,v,w): sum_{t=0}^{r_target-1} c_t * U[u,t]*V[v,t]*W[w,t] = T[u,v,w] - sum_{t=r_target}^{99} U[u,t]*V[v,t]*W[w,t]
        
        # Build the matrix A of size (10000, r_target)
        # A[u*v*w + (v*w + w), t] = U[u,t]*V[v,t]*W[w,t]
        # This is too large. Let's use a different approach.
        
        # Instead, let's use a "random projection" to reduce the number of equations.
        # Project the tensor onto a random subspace of dimension ~ r_target * 2.
        
        np.random.seed(42)
        num_projections = r_target * 3
        
        # Random projection matrices
        P1 = np.random.randn(n * m, num_projections)
        P2 = np.random.randn(m * p, num_projections)
        P3 = np.random.randn(p * n, num_projections)
        
        # Projected tensor: T_proj[a,b,c] = sum_{u,v,w} P1[u,a]*P2[v,b]*P3[w,c]*T[u,v,w]
        # = sum_{u,v,w} P1[u,a]*P2[v,b]*P3[w,c] * sum_t U[u,t]*V[v,t]*W[w,t]
        # = sum_t (sum_u P1[u,a]*U[u,t]) * (sum_v P2[v,b]*V[v,t]) * (sum_w P3[w,c]*W[w,t])
        # = sum_t U_proj[a,t] * V_proj[b,t] * W_proj[c,t]
        
        U_proj = P1.T @ U_sb  # (num_projections, 100)
        V_proj = P2.T @ V_sb  # (num_projections, 100)
        W_proj = P3.T @ W_sb  # (num_projections, 100)
        
        # The projected tensor is:
        # T_proj[a,b,c] = sum_t U_proj[a,t]*V_proj[b,t]*W_proj[c,t]
        # We want to find a rank-r_target decomposition of T_proj.
        # i.e., find U_p, V_p, W_p of size (num_projections, r_target) such that
        # sum_t U_p[a,t]*V_p[b,t]*W_p[c,t] = T_proj[a,b,c]
        
        # This is still a tensor decomposition problem, but with smaller dimensions.
        # num_projections = r_target * 3 ~ 270-300.
        # r_target ~ 90.
        # This is still hard.
        
        # Let's try a different approach: 
        # Use a "greedy" rank reduction.
        # Start with rank 100, and try to remove one term at a time.
        # For each term t, check if it can be expressed as a linear combination of the others.
        
        for t_remove in range(r_sb - 1, r_target - 1, -1):
            if time.time() > deadline - 10:
                return None
            
            # Check if term t_remove is in the span of the other terms.
            # Build the matrix of the other terms.
            idx = [i for i in range(r_sb) if i != t_remove]
            U_other = U_sb[:, idx]  # (20, 99)
            V_other = V_sb[:, idx]  # (25, 99)
            W_other = W_sb[:, idx]  # (20, 99)
            
            # The tensor of the other terms:
            # T_other[u,v,w] = sum_t U_other[u,t]*V_other[v,t]*W_other[w,t]
            # We want to check if T[u,v,w] - T_other[u,v,w] is a rank-1 tensor.
            # i.e., if there exist vectors a, b, c such that
            # (T - T_other)[u,v,w] = a[u]*b[v]*c[w]
            
            # Compute T - T_other
            T_other = np.einsum('ut,vt,wt->uvw', U_other, V_other, W_other, optimize=True)
            T_diff = T - T_other
            
            # Check if T_diff is rank-1.
            # A tensor is rank-1 if it can be written as a[u]*b[v]*c[w].
            # This is equivalent to checking if the matricizations are rank-1.
            # Matricization along u: shape (20, 25*20) = (20, 500)
            # Rank-1 iff the matrix has rank 1.
            
            T_diff_u = T_diff.reshape(n * m, m * p * p * n)
            rank_u = np.linalg.matrix_rank(T_diff_u, tol=1e-9)
            
            if rank_u == 1:
                # T_diff is rank-1. Extract the vectors.
                # a = T_diff_u[:, 0] / norm, etc.
                # Actually, we can just use the SVD.
                U_svd, S, Vt = np.linalg.svd(T_diff_u, full_matrices=False)
                if len(S) > 0 and S[0] > 1e-9:
                    a = U_svd[:, 0] * np.sqrt(S[0])
                    b_c = Vt[0, :] * np.sqrt(S[0])
                    b_c = b_c.reshape(m * p, p * n)
                    # Now b_c should be rank-1: b[v]*c[w]
                    U_b, S_b, Vt_c = np.linalg.svd(b_c, full_matrices=False)
                    if len(S_b) > 0 and S_b[0] > 1e-9:
                        b = U_b[:, 0] * np.sqrt(S_b[0])
                        c = Vt_c[0, :] * np.sqrt(S_b[0])
                        
                        # Now we have a, b, c such that T_diff = a*b*c.
                        # Add this as a new term.
                        U_new = np.column_stack([U_sb[:, idx], a])
                        V_new = np.column_stack([V_sb[:, idx], b])
                        W_new = np.column_stack([W_sb[:, idx], c])
                        
                        # Check if this gives a valid decomposition.
                        if check(U_new, V_new, W_new):
                            return U_new, V_new, W_new
        
        return None

    # Try to reduce from rank 100 to rank 90, 85, 80, etc.
    for r_target in [90, 85, 80, 75, 70]:
        if time.time() > deadline - 20:
            break
        
        result = try_rank_reduction(r_target)
        if result:
            U, V, W = result
            # Try to snap to half-integers or integers
            U_s, V_s, W_s = snap_half(U, V, W)
            if check(U_s, V_s, W_s):
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
            
            U_s, V_s, W_s = snap_int(U, V, W)
            if check(U_s, V_s, W_s):
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
            
            U_rat = rational_reconstruct(U.flatten()).reshape(U.shape)
            V_rat = rational_reconstruct(V.flatten()).reshape(V.shape)
            W_rat = rational_reconstruct(W.flatten()).reshape(W.shape)
            if check(U_rat, V_rat, W_rat):
                return {"U": U_rat.tolist(), "V": V_rat.tolist(), "W": W_rat.tolist()}
            
            # Return the non-snapped version if it's valid
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # If all else fails, return schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
