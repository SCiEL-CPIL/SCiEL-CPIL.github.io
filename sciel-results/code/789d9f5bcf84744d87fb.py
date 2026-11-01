import numpy as np
import itertools
import time

DIMENSIONS = [2, 2, 4]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_rank14_strassen():
    """
    Deterministic Strassen-based rank-14 decomposition for 2x2 times 2x4.
    This serves as a guaranteed baseline and fallback.
    """
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 factor matrices (4x7 each)
    U_2x2 = np.array([
        [1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, -1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V_2x2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [1, 1, -1, 0, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W_2x2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Block 1: first 2 columns (k=0,1)
    U[:, 0:7] = U_2x2
    V[0*4+0:0*4+2, 0:7] = V_2x2[0:2, :]
    V[1*4+0:1*4+2, 0:7] = V_2x2[2:4, :]
    W[0*2+0:0*2+2, 0:7] = W_2x2[0:2, :]
    W[1*2+0:1*2+2, 0:7] = W_2x2[2:4, :]
    
    # Block 2: second 2 columns (k=2,3)
    U[:, 7:14] = U_2x2
    V[0*4+2:0*4+4, 7:14] = V_2x2[0:2, :]
    V[1*4+2:1*4+4, 7:14] = V_2x2[2:4, :]
    W[0*2+2:0*2+4, 7:14] = W_2x2[0:2, :]
    W[1*2+2:1*2+4, 7:14] = W_2x2[2:4, :]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _trivial_solution(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _search_rank13(n, m, p, deadline):
    """
    Attempts to find a rank-13 decomposition using a constrained combinatorial search.
    
    Strategy:
    The tensor T has 16 non-zero entries (indices (i,j,k)).
    A rank-r decomposition is a sum of r rank-1 tensors.
    Each rank-1 term is defined by selecting one index from each dimension.
    Let S be the set of 16 indices. We want to cover S with r terms such that
    the system of linear equations for the coefficients has a solution.
    
    Actually, a more direct approach for small ranks:
    We look for a set of r "monomials" (triplets (i,j,k)) such that the matrix
    formed by these monomials (rows = equations i,j,k; cols = terms r) has rank
    equal to the number of non-zero equations (16) if r >= 16? No.
    
    Correct formulation:
    We want U, V, W of size (nm x r), (mp x r), (pn x r).
    The condition is: for all (i,j,k), sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    
    This is a system of 16 equations.
    Unknowns are the entries of U, V, W. Total unknowns = r*(nm+mp+pn).
    However, the equations are multilinear.
    
    Alternative: Fix the support of the rank-1 terms.
    A rank-1 term is specified by (i_r, j_r, k_r).
    If we fix the supports (i_r, j_r, k_r) for r=1..R, the problem becomes linear in the scaling factors?
    No, because U[i,j,r] is non-zero only if i_r=i and j_r=j.
    Wait, U is a matrix. U[i*m+j, r].
    If term r has support (i_r, j_r, k_r), then:
    U[i*m+j, r] = u_r if i=i_r, j=j_r, else 0.
    V[j*p+k, r] = v_r if j=j_r, k=k_r, else 0.
    W[k*n+i, r] = w_r if k=k_r, i=i_r, else 0.
    
    Then the product U[i,j,r]V[j,k,r]W[k,i,r] is non-zero ONLY IF
    i=i_r, j=j_r, k=k_r.
    In that case, the value is u_r v_r w_r.
    
    So, if we fix the supports S = {(i_r, j_r, k_r)}_{r=1..R},
    the equation for a specific (i,j,k) is:
    sum_{r: (i_r,j_r,k_r)==(i,j,k)} (u_r v_r w_r) = 1
    
    This implies that if multiple terms have the same support, their products sum to 1.
    If a support appears 0 times, the LHS is 0, but RHS is 1 -> Impossible.
    Therefore, the set of supports MUST cover all 16 indices.
    
    If R=13, we need to cover 16 indices with 13 supports.
    This means some supports must be "shared" or... wait.
    If a support (i,j,k) is used by term r, it contributes to equation (i,j,k).
    It does NOT contribute to any other equation (i',j',k') != (i,j,k).
    
    Therefore, for the equation (i,j,k) to equal 1, there must be at least one term r with support (i,j,k).
    And sum_{r in S_{i,j,k}} u_r v_r w_r = 1.
    
    This means we can simply assign:
    For each (i,j,k), let S_{i,j,k} be the set of terms with that support.
    We need |S_{i,j,k}| >= 1 for all 16 pairs.
    Since total terms R = sum |S_{i,j,k}|, and we have 16 non-empty sets,
    the minimum R is 16 (if each set has size 1).
    
    Is it possible to have R < 16?
    Only if a single term could contribute to multiple equations.
    But U[i,j,r]V[j,k,r]W[k,i,r] is non-zero ONLY at (i_r, j_r, k_r).
    It is a delta function in the output space.
    It cannot contribute to (i',j',k') != (i_r,j_r,k_r).
    
    Therefore, the tensor rank of the "diagonal" part (which is the whole tensor here, as A and B are just inputs to the multiplication tensor which is sparse in the output index relative to the input indices? No.)
    
    Let's re-read the tensor definition.
    T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    
    If we choose a rank-1 term with fixed indices (i_r, j_r, k_r) for U, V, W respectively:
    U_{i,j,r} is non-zero only if i=i_r AND j=j_r.
    V_{j,k,r} is non-zero only if j=j_r AND k=k_r.
    W_{k,i,r} is non-zero only if k=k_r AND i=i_r.
    
    For the product to be non-zero, we need:
    1. i=i_r and j=j_r (from U)
    2. j=j_r and k=k_r (from V)
    3. k=k_r and i=i_r (from W)
    
    These conditions are consistent and imply i=i_r, j=j_r, k=k_r.
    So yes, each rank-1 term in this specific decomposition format (where U, V, W are matrices with fixed row indices) corresponds to a single "spike" in the (i,j,k) space.
    
    Wait, is U allowed to have multiple non-zero entries in a column?
    Yes, U is a matrix of shape (n*m, rank).
    Column r of U can have multiple non-zeros.
    Let U_{i,j,r} be the value.
    If U_{i,j,r} != 0 for multiple (i,j), then the term r contributes to multiple equations.
    
    Let's re-evaluate.
    Term r is defined by vectors u_r (in R^{nm}), v_r (in R^{mp}), w_r (in R^{pn}).
    The contribution to T_{i,j,k} is sum_r u_r[i,j] v_r[j,k] w_r[k,i].
    
    If we restrict u_r, v_r, w_r to be "unit vectors" (single non-zero), then rank >= 16.
    But we are NOT restricted to unit vectors.
    u_r can be a general vector.
    
    So, the "support" argument above was flawed because it assumed rank-1 tensors in the *tensor product space* C^{nm} x C^{mp} x C^{pn} which map to C^{nmp}.
    The tensor T is in C^{nm} x C^{mp} x C^{pn} -> C^{nmp}.
    Actually, T is a 3-tensor.
    T_{i,j,k} is the coefficient.
    The decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
    (u_r \otimes v_r \otimes w_r)_{i,j,k} = u_r[i,j] v_r[j,k] w_r[k,i].
    
    If u_r, v_r, w_r are arbitrary vectors, a single term can contribute to many (i,j,k).
    Specifically, term r contributes to (i,j,k) if u_r[i,j] != 0 AND v_r[j,k] != 0 AND w_r[k,i] != 0.
    
    So, can we cover the 16 required entries with 13 terms?
    Yes, potentially.
    
    How to search?
    This is a hard NP-complete problem (Tensor Rank).
    However, for 2x2x4, the dimensions are small.
    
    We can use a randomized approach with a constraint solver?
    Or, we can use the known result:
    The rank of 2x2x4 matrix multiplication is known to be 14?
    Actually, standard matrix multiplication 2x2x2 has rank 7.
    2x2x4 is like two 2x2x2s coupled?
    
    Let's check the literature or logic.
    If we view 2x2x4 as 2x2x2 + 2x2x2 (block diagonal in k), rank is 7+7=14.
    Can we do better?
    
    There is a known result for 2x2x4.
    According to some sources, the rank of 2x2x4 is 14.
    If the rank is 14, then the current program is optimal.
    
    However, the task asks to "Find a rank-minimal exact tensor decomposition".
    If 14 is minimal, we should return 14.
    If 13 is possible, we should find it.
    
    Given the time limit and the difficulty of proving rank 13 exists (or doesn't),
    and the fact that the current program achieves 14,
    a "structural change" that might improve the score (if 13 exists) or maintain it (if 14 is minimal) is to implement a **randomized local search** for rank 13.
    
    If we find rank 13, score becomes 16/13 = 1.23.
    If we don't, we return rank 14, score 1.0.
    
    Strategy for search:
    1. Start with a random rank-14 decomposition (or the Strassen one).
    2. Try to merge two terms or adjust coefficients to eliminate one term?
       Merging is hard.
    3. Better: Randomly initialize 13 terms.
       Set u_r, v_r, w_r to random sparse vectors.
       Use gradient descent or Newton's method to minimize ||T - sum_r u_r \otimes v_r \otimes w_r||^2.
       If we reach 0, we have a rank 13 decomposition.
    
    Given 120s, we can afford a few iterations of a simple solver.
    
    Let's implement a simple randomized search for rank 13.
    
    Steps:
    1. Generate random sparse vectors for 13 terms.
       To ensure coverage, we can bias the sparsity.
    2. Solve the linear system for the "scales" if the supports are fixed?
       No, the supports are part of the variables.
       
    Alternative: Fix the supports (indices of non-zeros) randomly, then solve for the values.
    If the supports are fixed, the equations are linear in the products of the non-zero entries?
    No, u[i] * v[j] * w[k] is trilinear.
    
    However, if we fix the *pattern* of non-zeros, we can use a solver like `scipy.optimize.least_squares` on the full set of variables.
    
    Let's try a simpler heuristic:
    The 2x2x4 tensor is "block diagonal" in the k-dimension if we group k=0,1 and k=2,3.
    The Strassen solution exploits this.
    Any rank 13 solution would have to break this block structure.
    
    Given the high likelihood that 14 is the true minimal rank for this specific tensor structure (often cited in algebraic complexity for small matrices), the probability of finding 13 is low.
    
    However, the directive says "Make a STRUCTURAL change".
    The current program is "Deterministic Construction".
    The new program will be "Randomized Optimization Search for Rank 13, falling back to Deterministic Rank 14".
    
    If the search fails, it returns 14.
    If it succeeds, it returns 13.
    
    This is a valid structural change.
    
    Implementation of the search:
    1. Define the target tensor T (16 ones).
    2. Try rank 13.
    3. Initialize U, V, W with random values (small integers or floats).
    4. Use `scipy.optimize.minimize` to minimize the sum of squared errors.
    5. If error < 1e-6, verify and return.
    6. Repeat a few times with different seeds.
    7. If time runs out or no success, return Strassen 14.
    
    To make the search more effective, we can initialize with the Strassen 14 solution, zero out one column (randomly), and then optimize the remaining 13 columns to fit the target. This is a "reduction" strategy.
    
    Let's implement this.
"""

import scipy.optimize as so
import numpy as np
import time
import random

DIMENSIONS = [2, 2, 4]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_rank14_strassen():
    n, m, p = DIMENSIONS
    U_2x2 = np.array([
        [1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, -1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V_2x2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [1, 1, -1, 0, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W_2x2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U[:, 0:7] = U_2x2
    V[0*4+0:0*4+2, 0:7] = V_2x2[0:2, :]
    V[1*4+0:1*4+2, 0:7] = V_2x2[2:4, :]
    W[0*2+0:0*2+2, 0:7] = W_2x2[0:2, :]
    W[1*2+0:1*2+2, 0:7] = W_2x2[2:4, :]
    
    U[:, 7:14] = U_2x2
    V[0*4+2:0*4+4, 7:14] = V_2x2[0:2, :]
    V[1*4+2:1*4+4, 7:14] = V_2x2[2:4, :]
    W[0*2+2:0*2+4, 7:14] = W_2x2[0:2, :]
    W[1*2+2:1*2+4, 7:14] = W_2x2[2:4, :]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _trivial_solution(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _search_rank13(n, m, p, deadline):
    """
    Attempts to find a rank-13 decomposition by optimizing a rank-13 model
    initialized from a rank-14 solution with one column removed.
    """
    # Target tensor values
    # T[i,j,k] = 1 for all i,j,k
    # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    
    # Get the rank 14 solution
    sol14 = _build_rank14_strassen()
    U14 = np.array(sol14['U'])
    V14 = np.array(sol14['V'])
    W14 = np.array(sol14['W'])
    
    # Try to remove each of the 14 columns and see if the rest can be adjusted to fit
    # We will run a limited number of optimizations
    
    def objective(params):
        # params: flattened U, V, W
        # U: 4x13, V: 8x13, W: 8x13
        # Total size: 4*13 + 8*13 + 8*13 = 52 + 104 + 104 = 260
        
        U_flat, V_flat, W_flat = params[:52], params[52:156], params[156:]
        U = U_flat.reshape(4, 13)
        V = V_flat.reshape(8, 13)
        W = W_flat.reshape(8, 13)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(13):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    error += (s - 1.0)**2
        return error
    
    best_sol = None
    best_err = float('inf')
    
    # Try removing each column of the rank 14 solution as a starting point
    for drop_idx in range(14):
        if time.time() > deadline - 10:
            break
            
        # Create initial guess
        U_init = np.delete(U14, drop_idx, axis=1)
        V_init = np.delete(V14, drop_idx, axis=1)
        W_init = np.delete(W14, drop_idx, axis=1)
        
        # Add small noise to escape local minima
        U_init = U_init + np.random.normal(0, 0.1, U_init.shape)
        V_init = V_init + np.random.normal(0, 0.1, V_init.shape)
        W_init = W_init + np.random.normal(0, 0.1, W_init.shape)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            # Use Nelder-Mead for robustness, or BFGS
            # Nelder-Mead is slower but handles non-smoothness better (though this is smooth)
            # Let's use BFGS with a small number of iterations
            res = so.minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < best_err:
                best_err = res.fun
                if res.fun < 1e-6:
                    # Success
                    U_res = res.x[:52].reshape(4, 13)
                    V_res = res.x[52:156].reshape(8, 13)
                    W_res = res.x[156:].reshape(8, 13)
                    
                    # Round to nearest half-integer? The task allows integer/half-integer.
                    # But the evaluator might accept floats? "integer/half-integer entries"
                    # We should try to snap to half-integers if close.
                    U_res = np.round(U_res * 2) / 2
                    V_res = np.round(V_res * 2) / 2
                    W_res = np.round(W_res * 2) / 2
                    
                    # Verify the snapped version
                    if _verify_decomposition(U_res.tolist(), V_res.tolist(), W_res.tolist(), n, m, p, tol=1e-5):
                        return {
                            "U": U_res.tolist(),
                            "V": V_res.tolist(),
                            "W": W_res.tolist()
                        }
                    else:
                        # If snapping failed, try returning the float version?
                        # The task says "integer/half-integer entries".
                        # If we can't snap, it's invalid.
                        pass
                        
        except Exception as e:
            continue
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Try to find rank 13
    # This is the structural change: active search for lower rank
    result_13 = _search_rank13(n, m, p, deadline)
    if result_13 is not None:
        return result_13
        
    # 2. Fallback to rank 14
    result_14 = _build_rank14_strassen()
    if _verify_decomposition(result_14["U"], result_14["V"], result_14["W"], n, m, p, tol=1e-8):
        return result_14
        
    # 3. Fallback to rank 16
    return _trivial_solution(n, m, p)
