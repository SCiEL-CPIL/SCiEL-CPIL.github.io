"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.linalg import lstsq

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook_decomp(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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


def _build_tensor(n, m, p):
    """Build the flattened multiplication tensor of shape (n*m*p,).
    T[i*m+j, k*n+i] = 1 for all (i,j,k), zero elsewhere.
    We flatten as T_flat[i*m*p + j*p + k] = 1 if the (i,j,k) entry is 1.
    Actually, the tensor T has entries T[i,j,k] = 1 if (i,j,k) is a valid index.
    We need: C[i,k] = sum_j A[i,j]*B[j,k]
    So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    The flattening convention: we use index = i*(m*p) + j*p + k
    """
    T = np.zeros(n * m * p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * (m * p) + j * p + k
                T[idx] = 1.0
    return T


def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = U.shape[1]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
                expected = 1.0
                err = abs(val - expected)
                if err > max_err:
                    max_err = err
    return max_err < tol


def _try_rank_decomposition(n, m, p, rank, seed=0, max_iter=200):
    """Try to find a rank-r decomposition by fixing random U,V and solving for W,
    then iterating. Returns (U, V, W) if found, else None.
    
    The key identity: T must be in the span of {u_r ⊗ v_r ⊗ w_r : r=0..rank-1}
    For fixed U and V, we can solve for W using the linear system:
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is nonlinear in W. Instead, we use an alternating approach:
    Fix U and V, solve for W via least squares on the linear system in W entries.
    Then fix V and W, solve for U. Iterate.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with random factors
    U_arr = rng.randn(n * m, rank)
    V_arr = rng.randn(m * p, rank)
    W_arr = rng.randn(p * n, rank)
    
    # Normalize columns to help convergence
    for r in range(rank):
        norm_u = np.linalg.norm(U_arr[:, r])
        norm_v = np.linalg.norm(V_arr[:, r])
        norm_w = np.linalg.norm(W_arr[:, r])
        if norm_u < 1e-12 or norm_v < 1e-12 or norm_w < 1e-12:
            continue
        U_arr[:, r] /= norm_u
        V_arr[:, r] /= norm_v
        W_arr[:, r] /= norm_w
    
    for it in range(max_iter):
        # Solve for W given U, V
        # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # Group by w_index = k*n+i
        # For a fixed w_index (k,i), the equation is:
        #   sum over (j) and r: U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # But W[k*n+i,r] appears multiplied by U[i*m+j,r]*V[j*p+k,r]
        # So for fixed (k,i), we have one equation per j:
        #   sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
        # This gives m equations in rank unknowns W[k*n+i,0..rank-1]
        # We can solve this as a least squares problem.
        
        W_new = np.zeros((p * n, rank))
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                # Build the system: for each j, sum_r A[j,r] * W[w_idx,r] = 1
                # where A[j,r] = U[i*m+j, r] * V[j*p+k, r]
                A = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    for r in range(rank):
                        A[j, r] = U_arr[i * m + j, r] * V_arr[j * p + k, r]
                if rank <= m:
                    # Underdetermined or square: use lstsq
                    W_new[w_idx, :], residuals, rank_ls, sv = lstsq(A, b, cond=None)
                else:
                    # Overdetermined: use lstsq
                    W_new[w_idx, :], residuals, rank_ls, sv = lstsq(A, b, cond=None)
        
        W_arr = W_new
        
        # Solve for U given V, W
        # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # For fixed u_index = i*m+j, the equation is:
        #   sum over (k) and r: U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
        # This gives p equations in rank unknowns.
        
        U_new = np.zeros((n * m, rank))
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                A = np.zeros((p, rank))
                b = np.ones(p)
                for k in range(p):
                    for r in range(rank):
                        A[k, r] = V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if rank <= p:
                    U_new[u_idx, :], residuals, rank_ls, sv = lstsq(A, b, cond=None)
                else:
                    U_new[u_idx, :], residuals, rank_ls, sv = lstsq(A, b, cond=None)
        
        U_arr = U_new
        
        # Solve for V given U, W
        # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # For fixed v_index = j*p+k, the equation is:
        #   sum over (i) and r: V[j*p+k,r] * (U[i*m+j,r] * W[k*n+i,r]) = 1
        # This gives n equations in rank unknowns.
        
        V_new = np.zeros((m * p, rank))
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                A = np.zeros((n, rank))
                b = np.ones(n)
                for i in range(n):
                    for r in range(rank):
                        A[i, r] = U_arr[i * m + j, r] * W_arr[k * n + i, r]
                if rank <= n:
                    V_new[v_idx, :], residuals, rank_ls, sv = lstsq(A, b, cond=None)
                else:
                    V_new[v_idx, :], residuals, rank_ls, sv = lstsq(A, b, cond=None)
        
        V_arr = V_new
        
        # Check convergence
        if _verify_decomp(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), n, m, p, tol=1e-7):
            # Convert to list of lists with rounded values
            U_list = [[float(U_arr[i, r]) for r in range(rank)] for i in range(n * m)]
            V_list = [[float(V_arr[i, r]) for r in range(rank)] for i in range(m * p)]
            W_list = [[float(W_arr[i, r]) for r in range(rank)] for i in range(p * n)]
            return U_list, V_list, W_list
    
    return None


def construct():
    n, m, p = 3, 3, 5
    start_time = time.time()
    deadline = 120.0 - 5.0  # 5 second safety margin
    
    # Start with schoolbook
    best = _schoolbook_decomp(n, m, p)
    best_rank = n * m * p  # 45
    
    # Use a structured rank-40 construction based on the following:
    # 
    # We need: C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k] for i in 0..2, k in 0..4
    # 
    # Construction idea: use 40 terms where each term is a product
    # (linear combo of A entries) * (linear combo of B entries)
    # and contributes to specific (i,k) outputs via the W factor.
    #
    # Let me use a known construction. The rank of M(3,3,5) is at most 40.
    # I'll use a construction that achieves rank 40.
    #
    # Based on the "substitution" method:
    # - Compute 10 products of the form (sum a_i A[i,j]) * (sum b_j B[j,k])
    # - Each such product contributes to multiple (i,k) entries
    #
    # Actually, let me use a direct verified construction.
    # I'll use the following 40-term construction:
    
    # Term structure: 40 terms
    # For t = 0..39:
    #   U[t] has nonzero entries in specific (i,j) positions
    #   V[t] has nonzero entries in specific (j,k) positions  
    #   W[t] has nonzero entries in specific (k,i) positions
    #
    # This is hard to construct manually. Let me use a computational search
    # with a better strategy: use the fact that we can fix U and V and solve for W.
    
    # For fixed U and V, the equation is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    #
    # This is linear in W! For each (k,i), we have m equations:
    # sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1 for j=0,1,2
    #
    # So we have a 3 x rank linear system for W[k*n+i, :].
    # If rank >= 3, this is solvable (may be underdetermined).
    #
    # So the strategy is:
    # 1. Fix U and V randomly
    # 2. Solve for W using the linear system
    # 3. Verify the full decomposition
    # 4. If it doesn't work, try different U, V
    
    # The issue: the linear system for W may not have a solution that also
    # satisfies the equations for all (i,j,k) simultaneously, because the
    # W entries are shared across different j values.
    
    # Wait, let me re-examine. For fixed U, V:
    # For each (k,i), the equations are:
    #   For j=0: sum_r U[i*m+0, r] * V[0*p+k, r] * W[k*n+i, r] = 1
    #   For j=1: sum_r U[i*m+1, r] * V[1*p+k, r] * W[k*n+i, r] = 1
    #   For j=2: sum_r U[i*m+2, r] * V[2*p+k, r] * W[k*n+i, r] = 1
    #
    # This is a 3 x rank system. If rank > 3, it's underdetermined and has
    # solutions. If rank = 3, it's square and has a unique solution (if invertible).
    #
    # So for rank >= 3, we can always find W that satisfies these equations!
    # But we need to verify that the resulting W also satisfies the equations
    # when we consider the full trilinear form.
    
    # Actually, the equations ARE the full trilinear form. If we solve for W
    # such that for all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1,
    # then we have a valid decomposition.
    
    # So the algorithm is:
    # 1. For a given rank r, randomly initialize U and V
    # 2. For each (k,i), solve the 3 x r linear system for W[k*n+i, :]
    # 3. Verify the full decomposition
    # 4. If verification fails, the issue is that the linear systems are
    #    inconsistent (which shouldn't happen if we solve each independently)
    
    # Wait, each (k,i) gives an independent system for W[k*n+i, :]. So if we
    # solve each system, we get a W that satisfies all equations by construction!
    
    # Let me verify this logic:
    # The equation for (i,j,k) is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # For fixed (k,i), this is a system in W[k*n+i, 0..r-1]:
    #   sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1 for j=0,1,2
    #
    # This is a 3 x r system. If r >= 3, it has a solution (by linear algebra,
    # as long as the rows are linearly independent, or we use least squares).
    #
    # So for ANY U, V, and rank r >= 3, we can find W that satisfies all equations!
    #
    # But wait, that can't be right. The rank of the tensor is a fixed number.
    # 
    # The issue: the W we find must be consistent. But W[k*n+i, :] is determined
    # independently for each (k,i). There's no coupling between different (k,i)
    # groups in the equations. So yes, for any U, V with rank >= 3, we can find W.
    #
    # This means ANY rank r >= 3 works?! That's clearly wrong.
    #
    # Let me re-examine. The tensor T has entries T[i,j,k] = 1 for all (i,j,k).
    # A rank-r decomposition is: T[i,j,k] = sum_{r=0}^{r-1} U[i,j,r] * V[j,k,r] * W[k,i,r]
    #
    # For fixed U, V, we need to find W such that:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
    #
    # For fixed (k,i), this is: for j=0,1,2:
    #   sum_r (U[i,j,r] * V[j,k,r]) * W[k,i,r] = 1
    #
    # This is a 3 x r linear system. If r >= 3, it has a solution.
    #
    # So indeed, for any U, V, and r >= 3, we can find W. This means the rank
    # is at most 3?! That's clearly wrong.
    #
    # The error: I'm confusing the tensor structure. Let me re-read the problem.
    #
    # "The output factor follows the official trace convention W[k*n+i,r]"
    #
    # So W is indexed by (k, i) where k is the column of B (and C) and i is the
    # row of A (and C). The tensor entry is T[i,j,k] where i is row of A/C,
    # j is column of A / row of B, k is column of B/C.
    #
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # This is correct. And my analysis shows that for any U, V, we can solve for W.
    #
    # But this can't be right. Let me test with a small example.
    #
    # Take n=m=p=1. T[0,0,0] = 1.
    # U[0,0] * V[0,0] * W[0,0] = 1.
    # For fixed U, V, W[0,0] = 1/(U[0,0]*V[0,0]).
    # So rank 1 works. Correct.
    #
    # Take n=m=p=2. T[i,j,k] = 1 for all i,j,k in {0,1}.
    # For fixed U (2x2), V (2x2), solve for W (2x2).
    # For (k=0,i=0): 
    #   j=0: U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[1,0]*W[0,1] = 1
    #   j=1: U[1,0]*V[0,0]*W[0,0] + U[1,1]*V[1,0]*W[0,1] = 1
    # For (k=0,i=1):
    #   j=0: U[0,0]*V[0,0]*W[1,0] + U[0,1]*V[1,0]*W[1,1] = 1
    #   j=1: U[1,0]*V[0,0]*W[1,0] + U[1,1]*V[1,0]*W[1,1] = 1
    # Etc.
    #
    # Each (k,i) gives a 2x2 system. If the matrix is invertible, we get a unique W.
    # So for any U, V, we can find W. This means rank 2 works for 2x2x2?!
    #
    # But rank(2,2,2) = 7 (known result). So something is wrong.
    #
    # AH! I see the error. The W entries are NOT independent across different (k,i)
    # pairs in the way I'm thinking. Let me re-examine.
    #
    # W is a (p*n) x r matrix. W[k*n+i, r] is the entry. For fixed r, W[k*n+i, r]
    # is a single number. The equations for different (k,i) involve different W entries.
    #
    # For (k=0, i=0): involves W[0, r] for all r. (since k*n+i = 0*2+0 = 0)
    # For (k=0, i=1): involves W[1, r] for all r. (since k*n+i = 0*2+1 = 1)
    # For (k=1, i=0): involves W[2, r] for all r. (since k*n+i = 1*2+0 = 2)
    # For (k=1, i=1): involves W[3, r] for all r. (since k*n+i = 1*2+1 = 3)
    #
    # So each (k,i) pair involves a DIFFERENT set of W entries. They are independent!
    #
    # So for n=m=p=2, rank 2 should work? But that contradicts the known result.
    #
    # Let me actually test this numerically.
    
    # Test: n=m=p=2, rank=2
    n_test, m_test, p_test = 2, 2, 2
    rank_test = 2
    U_test = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)  # (n*m, rank) = (4, 2)
    V_test = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)  # (m*p, rank) = (4, 2)
    
    # Solve for W
    W_test = np.zeros((p_test * n_test, rank_test))
    for k in range(p_test):
        for i in range(n_test):
            w_idx = k * n_test + i
            A_mat = np.zeros((m_test, rank_test))
            b_vec = np.ones(m_test)
            for j in range(m_test):
                for r in range(rank_test):
                    A_mat[j, r] = U_test[i * m_test + j, r] * V_test[j * p_test + k, r]
            # Solve A_mat @ W_test[w_idx, :] = b_vec
            # A_mat is 2x2, W_test[w_idx, :] is 2x1
            try:
                W_test[w_idx, :] = np.linalg.solve(A_mat, b_vec)
            except np.linalg.LinAlgError:
                W_test[w_idx, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
    
    # Verify
    max_err = 0
    for i in range(n_test):
        for j in range(m_test):
            for k in range(p_test):
                val = sum(U_test[i*m_test+j, r] * V_test[j*p_test+k, r] * W_test[k*n_test+i, r] for r in range(rank_test))
                err = abs(val - 1.0)
                max_err = max(max_err, err)
    
    # If max_err is small, rank 2 works for 2x2x2, which contradicts known results.
    # Let me print the result.
    
    # Actually, I realize the issue: the problem is that U and V are FIXED, and we're
    # solving for W. But the rank of the tensor is the MINIMUM r such that there EXIST
    # U, V, W of rank r. My analysis shows that for ANY U, V, we can find W if r >= max(n, m, p)
    # (or something like that). But this means the rank is at most max(n,m,p) for any
    # matrix multiplication, which is clearly wrong.
    #
    # The error must be in my understanding. Let me re-read the tensor decomposition.
    #
    # Ah, I think I see. The tensor T is NOT T[i,j,k] = 1 for all (i,j,k). 
    # The multiplication tensor has T[i,j,k] = 1 if the (i,j,k) entry is "active",
    # which is always the case for matrix multiplication. But the DECOMPOSITION
    # requires that the trilinear form MATCHES the tensor, not just that each
    # equation is satisfied independently.
    #
    # Wait, that's what I'm doing. Each equation IS the tensor entry.
    #
    # I'm confused. Let me just run the numerical test.
    
    # Use a rank-42 construction:
    # 24 terms for 3x3x3 + 18 terms for 3x3x2
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 24-term decomposition for 3x3x3 (terms 0-23)
    # This is a known construction. I'll use a specific set of 24 terms.
    # Based on the "border rank" construction or a simplified Strassen.
    
    # For simplicity, I'll use a rank-24 decomposition that I can construct:
    # Use 27 schoolbook terms but merge 3 pairs.
    
    # Actually, let me use a different approach: use the schoolbook 27 terms
    # for 3x3x3 and 18 terms for 3x3x2, giving 45. That's not an improvement.
    
    # Let me use a verified rank-40 construction from a reliable source.
    # I'll use the following construction:
    
    # The key insight: for 3x3x5, we can use a rank-40 decomposition.
    # I'll construct it as follows:
    # - 20 terms that handle the "core" 3x3x3 multiplication with some sharing
    # - 20 terms that handle the extra 2 columns
    
    # This is getting too complex without a verified source.
    
    # Return the schoolbook decomposition as a safe fallback
    return best
# EVOLVE-BLOCK-END
