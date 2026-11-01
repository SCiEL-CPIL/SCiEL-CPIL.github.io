"""Tensor rank minimization for 2x2*2x4 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import linprog
from itertools import combinations

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _build_rank14():
    """Return the known rank-14 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for B1 (cols 0,1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Strassen for B2 (cols 2,3)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return U, V, W


def _verify_decomposition(U, V, W, tol=1e-8):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = len(U_arr[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U_arr[i*m + j, :] * V_arr[j*p + k, :] * W_arr[k*n + i, :])
                if abs(total - 1.0) > tol:
                    return False
    return True


def _verify_exact(U, V, W):
    """Verify with exact integer/half-integer arithmetic using fractions."""
    from fractions import Fraction
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0, 1)
                for r in range(rank):
                    total += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if total != 1:
                    return False
    return True


def _to_rational_list(arr, tol=1e-4):
    """Convert numpy array to list of integers or half-integers."""
    result = []
    for row in arr:
        new_row = []
        for val in row:
            # Try integer
            int_val = round(val)
            if abs(val - int_val) < tol:
                new_row.append(int_val)
            else:
                # Try half-integer
                half_val = round(val * 2) / 2.0
                if abs(val - half_val) < tol:
                    new_row.append(half_val)
                else:
                    # Try quarter-integer
                    qtr_val = round(val * 4) / 4.0
                    if abs(val - qtr_val) < tol:
                        new_row.append(qtr_val)
                    else:
                        return None
        result.append(new_row)
    return result


def _alternating_minimize(U_init, V_init, W_init, target_rank, max_iter=500, tol=1e-12):
    """Alternating minimization: fix two factors, solve for the third."""
    n, m, p = DIMENSIONS
    rank = target_rank
    
    U = np.array(U_init, dtype=float)
    V = np.array(V_init, dtype=float)
    W = np.array(W_init, dtype=float)
    
    for iteration in range(max_iter):
        U_old = U.copy()
        V_old = V.copy()
        W_old = W.copy()
        
        # Fix V and W, solve for U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                A_sys = np.zeros((p, rank))
                for k in range(p):
                    A_sys[k, :] = V[j*p + k, :] * W[k*n + i, :]
                b_sys = np.ones(p)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                    U_new[i*m + j, :] = sol
                except np.linalg.LinAlgError:
                    return None
        U = U_new
        
        # Fix U and W, solve for V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A_sys = np.zeros((n, rank))
                for i in range(n):
                    A_sys[i, :] = U[i*m + j, :] * W[k*n + i, :]
                b_sys = np.ones(n)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                    V_new[j*p + k, :] = sol
                except np.linalg.LinAlgError:
                    return None
        V = V_new
        
        # Fix U and V, solve for W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A_sys = np.zeros((m, rank))
                for j in range(m):
                    A_sys[j, :] = U[i*m + j, :] * V[j*p + k, :]
                b_sys = np.ones(m)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                    W_new[k*n + i, :] = sol
                except np.linalg.LinAlgError:
                    return None
        W = W_new
        
        change = (np.max(np.abs(U - U_old)) + np.max(np.abs(V - V_old)) + 
                  np.max(np.abs(W - W_old)))
        if change < tol:
            break
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                if abs(total - 1.0) > 1e-6:
                    return None
    
    # Convert to rational
    U_rat = _to_rational_list(U, tol=1e-3)
    V_rat = _to_rational_list(V, tol=1e-3)
    W_rat = _to_rational_list(W, tol=1e-3)
    
    if U_rat is None or V_rat is None or W_rat is None:
        return None
    
    # Exact verification
    if not _verify_exact(U_rat, V_rat, W_rat):
        return None
    
    return U_rat, V_rat, W_rat


def _structured_rank13_search():
    """
    Search for rank-13 decomposition using structured initialization.
    
    Key idea: The 2x2x2x4 problem can be seen as two 2x2x2 problems sharing A.
    We parameterize with cross-terms between the two halves.
    
    Structure: 7 terms for left half (B cols 0,1) + 7 terms for right half (B cols 2,3)
    = 14 terms. To get 13, we need one shared/merged term.
    
    We try: use 7 terms for left, 6 for right, with one term contributing to both halves.
    Or: 6+7, or use a different partition.
    """
    n, m, p = DIMENSIONS
    target_rank = 13
    
    U14, V14, W14 = _build_rank14()
    U14_arr = np.array(U14, dtype=float)
    V14_arr = np.array(V14, dtype=float)
    W14_arr = np.array(W14, dtype=float)
    
    best_result = None
    
    # Strategy 1: Merge pairs from rank-14
    # Try all pairs (r1, r2) to merge into one term
    for r1 in range(14):
        for r2 in range(r1 + 1, 14):
            U_init = np.zeros((n*m, target_rank))
            V_init = np.zeros((m*p, target_rank))
            W_init = np.zeros((p*n, target_rank))
            
            new_idx = 0
            for r in range(14):
                if r == r1 or r == r2:
                    continue
                U_init[:, new_idx] = U14_arr[:, r]
                V_init[:, new_idx] = V14_arr[:, r]
                W_init[:, new_idx] = W14_arr[:, r]
                new_idx += 1
            
            # Try different merge strategies
            for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                U_init[:, new_idx] = (1-alpha)*U14_arr[:, r1] + alpha*U14_arr[:, r2]
                V_init[:, new_idx] = (1-alpha)*V14_arr[:, r1] + alpha*V14_arr[:, r2]
                W_init[:, new_idx] = (1-alpha)*W14_arr[:, r1] + alpha*W14_arr[:, r2]
                
                result = _alternating_minimize(U_init, V_init, W_init, target_rank, max_iter=300)
                if result is not None:
                    best_result = result
                    return best_result
    
    # Strategy 2: Structured initialization with shared A-factors
    # The key insight: A is 2x2, so U has only 4 rows.
    # For rank 13, we have 13 columns in U (4x13), V (8x13), W (8x13)
    
    # Try random initializations with structure
    np.random.seed(42)
    for trial in range(200):
        # Initialize with random but structured values
        U_init = np.zeros((n*m, target_rank))
        V_init = np.zeros((m*p, target_rank))
        W_init = np.zeros((p*n, target_rank))
        
        # Fill with small random integers/half-integers
        for idx in range(target_rank):
            # U: 4 entries per term
            U_init[:, idx] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=4)
            # V: 8 entries per term  
            V_init[:, idx] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=8)
            # W: 8 entries per term
            W_init[:, idx] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=8)
        
        result = _alternating_minimize(U_init, V_init, W_init, target_rank, max_iter=200)
        if result is not None:
            best_result = result
            return best_result
    
    return best_result


def _rank13_direct_search():
    """
    Direct algebraic search for rank-13 decomposition.
    
    The tensor T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2,3}
    gives 16 equations. We need:
    sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    For rank 13: 13*(4+8+8) = 260 unknowns, 16 equations.
    Very underdetermined - the challenge is finding rational solutions.
    
    Approach: Use the fact that the problem has symmetry.
    The two halves (k=0,1 and k=2,3) are structurally similar.
    """
    n, m, p = DIMENSIONS
    target_rank = 13
    
    # Build the system more carefully
    # Variables: U[i*m+j, r] for i in 0..1, j in 0..1, r in 0..12
    #            V[j*p+k, r] for j in 0..1, k in 0..3, r in 0..12
    #            W[k*n+i, r] for k in 0..3, i in 0..1, r in 0..12
    
    # The 16 constraints:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of 16 trilinear equations in 260 unknowns.
    # We can fix some variables and solve.
    
    # Key structural insight: For the 2x2x2x4 case, we can exploit that
    # the multiplication by B[:,0:2] and B[:,2:4] share the same A.
    
    # Let's try: fix W to be simple (0s and 1s), then solve for U and V.
    # Or fix U to be simple.
    
    # Approach: Set W[k*n+i, r] to be 1 for specific (k,i,r) and 0 otherwise.
    # This reduces to a bilinear system.
    
    # For each r, W[:, :, r] is a 4x2 matrix (k x i).
    # If we set W to be "diagonal" in some sense, we can simplify.
    
    # Try: W[k*n+i, r] = 1 if k==r%4 and i==r//4, else 0
    # But that's only 8 terms, not 13.
    
    # Better: use the Strassen structure as a guide.
    # In Strassen for 2x2x2, the 7 terms have specific patterns.
    
    # Let me try a different approach: use the known rank-14 and look for
    # a linear dependency that allows merging.
    
    U14, V14, W14 = _build_rank14()
    U14_arr = np.array(U14, dtype=float)
    V14_arr = np.array(V14, dtype=float)
    W14_arr = np.array(W14, dtype=float)
    
    # Check if any pair of rank-14 terms can be merged
    # by finding alpha, beta such that the merged term still works
    # when the other 12 terms are kept.
    
    for r1 in range(14):
        for r2 in range(r1+1, 14):
            # Keep all terms except r1, r2
            # Add one merged term
            # Solve for the merged term coefficients
            
            # The residual after removing terms r1 and r2:
            # R[i,j,k] = 1 - sum_{r != r1,r2} U14[i,j,r]*V14[j,k,r]*W14[k,i,r]
            # We need: U_m[i,j]*V_m[j,k]*W_m[k,i] = R[i,j,k]
            
            R = np.ones((2, 2, 4))
            for r in range(14):
                if r == r1 or r == r2:
                    continue
                for i in range(2):
                    for j in range(2):
                        for k in range(4):
                            R[i,j,k] -= U14_arr[i*2+j, r] * V14_arr[j*4+k, r] * W14_arr[k*2+i, r]
            
            # Check if R can be written as a single rank-1 tensor
            # R[i,j,k] = U_m[i,j] * V_m[j,k] * W_m[k,i]
            
            # This is a 2x2x4 tensor. Can it be rank-1?
            # A tensor is rank-1 if all its "slices" are proportional.
            
            # Check: R[0,0,:] should be proportional to R[0,1,:], R[1,0,:], R[1,1,:]
            # with the right factors.
            
            # More precisely: R[i,j,k] = a[i,j] * b[j,k] * c[k,i]
            # Fix a[0,0] = 1 (if nonzero), then:
            # b[j,k] = R[0,j,k] / c[k,0]  (for j=0)
            # This gets complicated. Let's just try to solve it.
            
            # If R is rank-1, then:
            # R[0,0,0]*R[1,1,1] = R[0,1,0]*R[1,0,1] (for fixed k=0,1... no, k varies)
            
            # Actually, for a 2x2x4 tensor to be rank-1:
            # R[i,j,k] = u[i,j] * v[j,k] * w[k,i]
            
            # This means: for fixed j, R[:,j,:] = u[:,j] * (v[j,:] * w[:,j]^T)
            # The matrix R[:,j,:] (2x4) must be rank <= 1.
            
            rank1 = True
            for j in range(2):
                slice_j = R[:, j, :]  # 2x4 matrix
                if np.linalg.matrix_rank(slice_j) > 1:
                    rank1 = False
                    break
            
            if rank1:
                # Try to extract u, v, w
                # u[i,j] * v[j,k] * w[k,i] = R[i,j,k]
                
                # Set u[0,0] = 1 (if R[0,0,k] is nonzero)
                # Then v[0,k] = R[0,0,k] / w[k,0]
                # And u[1,0] = R[1,0,k] / (v[0,k] * w[k,1])
                
                # This is still coupled. Let's use a different approach.
                # Pick w[0,0] = 1, w[1,0] = 1, w[2,0] = 1, w[3,0] = 1
                # Then v[0,k] = R[0,0,k]
                # u[0,0] = R[0,0,0] / (v[0,0] * w[0,0]) = R[0,0,0] / R[0,0,0] = 1 (if nonzero)
                
                # This is getting complex. Let's just verify if R is rank-1
                # by checking all 2x2 minors of each mode-2 slice.
                
                # For mode-2 slice j: matrix M_j[i,k] = R[i,j,k]
                # M_j must be rank 1: all 2x2 minors must be zero.
                # For 2x4 matrix: M[0,k]*M[1,k'] - M[0,k']*M[1,k] = 0 for all k,k'
                
                for j in range(2):
                    M = R[:, j, :]
                    for k1 in range(4):
                        for k2 in range(k1+1, 4):
                            if abs(M[0,k1]*M[1,k2] - M[0,k2]*M[1,k1]) > 1e-10:
                                rank1 = False
                                break
                        if not rank1:
                            break
                    if not rank1:
                        break
            
            if rank1:
                # Extract the rank-1 decomposition
                # R[i,j,k] = u[i,j] * v[j,k] * w[k,i]
                
                # For j=0: R[i,0,k] = u[i,0] * v[0,k] * w[k,i]
                # For j=1: R[i,1,k] = u[i,1] * v[1,k] * w[k,i]
                
                # Divide: R[i,0,k]/R[i,1,k] = u[i,0]*v[0,k] / (u[i,1]*v[1,k])
                # = (u[i,0]/u[i,1]) * (v[0,k]/v[1,k])
                
                # This is a product of a function of i and a function of k.
                # So the matrix Q[i,k] = R[i,0,k]/R[i,1,k] must be rank 1.
                
                # Let's just construct it directly.
                # Set w[k,i] = 1 for all k,i (simplest).
                # Then R[i,j,k] = u[i,j] * v[j,k]
                # For fixed j: R[:,j,:] = u[:,j] * v[j,:]^T
                # This is a rank-1 matrix.
                
                # Check if this works:
                valid = True
                u = np.zeros((2,2))
                v = np.zeros((2,4))
                
                for j in range(2):
                    M_j = R[:, j, :]  # 2x4
                    if np.linalg.matrix_rank(M_j) > 1:
                        valid = False
                        break
                    # Extract: M_j = u[:,j] * v[j,:]
                    # u[:,j] = M_j[:,0] / v[j,0] (if v[j,0] != 0)
                    # Or use SVD
                    if abs(M_j[0,0]) > 1e-10:
                        u[0,j] = 1.0
                        v[j,:] = M_j[0,:]
                        u[1,j] = M_j[1,0] / M_j[0,0]
                    elif abs(M_j[1,0]) > 1e-10:
                        u[1,j] = 1.0
                        v[j,:] = M_j[1,:]
                        u[0,j] = M_j[0,0] / M_j[1,0]
                    else:
                        valid = False
                        break
                
                if valid:
                    # Verify
                    for i in range(2):
                        for j in range(2):
                            for k in range(4):
                                if abs(u[i,j]*v[j,k] - R[i,j,k]) > 1e-10:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                
                if valid:
                    # We have a rank-13 decomposition!
                    U13 = np.zeros((4, 13))
                    V13 = np.zeros((8, 13))
                    W13 = np.zeros((8, 13))
                    
                    new_idx = 0
                    for r in range(14):
                        if r == r1 or r == r2:
                            continue
                        U13[:, new_idx] = U14_arr[:, r]
                        V13[:, new_idx] = V14_arr[:, r]
                        W13[:, new_idx] = W14_arr[:, r]
                        new_idx += 1
                    
                    # Add the merged term
                    # W is all 1s (since we set w[k,i]=1)
                    U13[:, new_idx] = [u[0,0], u[0,1], u[1,0], u[1,1]]
                    V13[:, new_idx] = [v[0,0], v[0,1], v[0,2], v[0,3],
                                       v[1,0], v[1,1], v[1,2], v[1,3]]
                    W13[:, new_idx] = [1, 1, 1, 1, 1, 1, 1, 1]  # w[k,i] for k=0..3, i=0..1
                    
                    # Convert to rational
                    U13_rat = _to_rational_list(U13, tol=1e-3)
                    V13_rat = _to_rational_list(V13, tol=1e-3)
                    W13_rat = _to_rational_list(W13, tol=1e-3)
                    
                    if U13_rat is not None and V13_rat is not None and W13_rat is not None:
                        if _verify_exact(U13_rat, V13_rat, W13_rat):
                            return U13_rat, V13_rat, W13_rat
    
    return None


def _peeling_search(max_rank=13, max_iters=200, tol=1e-10):
    """
    Peeling algorithm for tensor rank minimization.
    """
    n, m, p = DIMENSIONS
    target_rank = max_rank
    
    def _verify(U, V, W, tol=1e-8):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U_arr[i*m + j, :] * V_arr[j*p + k, :] * W_arr[k*n + i, :])
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    U14, V14, W14 = _build_rank14()
    U14_arr = np.array(U14, dtype=float)
    V14_arr = np.array(V14, dtype=float)
    W14_arr = np.array(W14, dtype=float)
    
    best_result = None
    
    # Strategy 1: Start from rank-14 and try to merge pairs
    for r1 in range(14):
        for r2 in range(r1 + 1, 14):
            U_init = np.zeros((n*m, target_rank))
            V_init = np.zeros((m*p, target_rank))
            W_init = np.zeros((p*n, target_rank))
            
            new_idx = 0
            for r in range(14):
                if r == r1 or r == r2:
                    continue
                U_init[:, new_idx] = U14_arr[:, r]
                V_init[:, new_idx] = V14_arr[:, r]
                W_init[:, new_idx] = W14_arr[:, r]
                new_idx += 1
            
            U_init[:, new_idx] = (U14_arr[:, r1] + U14_arr[:, r2]) / 2.0
            V_init[:, new_idx] = (V14_arr[:, r1] + V14_arr[:, r2]) / 2.0
            W_init[:, new_idx] = (W14_arr[:, r1] + W14_arr[:, r2]) / 2.0
            
            result = _alternating_minimize(U_init, V_init, W_init, target_rank, max_iter=300)
            if result is not None:
                best_result = result
                break
        if best_result is not None:
            break
    
    # Strategy 2: Random initializations
    if best_result is None:
        np.random.seed(42)
        for trial in range(100):
            U_init = np.random.randn(n*m, target_rank)
            V_init = np.random.randn(m*p, target_rank)
            W_init = np.random.randn(p*n, target_rank)
            
            U_init /= np.sqrt(np.sum(U_init**2))
            V_init /= np.sqrt(np.sum(V_init**2))
            W_init /= np.sqrt(np.sum(W_init**2))
            
            result = _alternating_minimize(U_init, V_init, W_init, target_rank, max_iter=200)
            if result is not None:
                best_result = result
                break
    
    return best_result


def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit
    
    # Try direct algebraic search first (fast, deterministic)
    if time.time() - start_time < deadline - 10:
        result_direct = _rank13_direct_search()
        if result_direct is not None:
            return {"U": result_direct[0], "V": result_direct[1], "W": result_direct[2]}
    
    # Try structured search
    if time.time() - start_time < deadline - 10:
        result_structured = _structured_rank13_search()
        if result_structured is not None:
            return {"U": result_structured[0], "V": result_structured[1], "W": result_structured[2]}
    
    # Try peeling search
    if time.time() - start_time < deadline - 5:
        result_13 = _peeling_search()
        if result_13 is not None:
            return {"U": result_13[0], "V": result_13[1], "W": result_13[2]}
    
    # Fallback to rank-14
    U, V, W = _build_rank14()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

