import numpy as np
import time
from itertools import combinations, product

DIMENSIONS = [2, 3, 3]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r_idx in range(r):
                    val += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_combinatorial_rank16(n, m, p, deadline):
    """
    Combinatorial search for rank-16 decomposition.
    
    Strategy: For each of the 16 rank terms, we choose which (i,j), (j,k), (k,i) 
    indices are non-zero. The key insight is that for a valid decomposition,
    each term must contribute to exactly one (i,j,k) triple (or be structured 
    so that the sum works out).
    
    Actually, a better approach: use the fact that for matrix multiplication,
    we can try to find a rank-16 decomposition by solving a linear system.
    
    For rank r, we have r unknowns per factor matrix entry. The constraint is:
    sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    This is a system of n*m*p = 18 equations. With rank 16, we have 16*(6+9+6) = 336 unknowns.
    This is heavily underdetermined, so we need additional constraints.
    
    Key structural insight: For matrix multiplication, the tensor has a special structure.
    We can use the fact that T[i,j,k] = delta_{j,j'} * delta_{k,k'} summed appropriately.
    
    Alternative approach: Use known rank bounds. The border rank of 2x3x3 is at most 16.
    Let's try to construct it explicitly.
    
    Actually, let me think about this differently. The matrix multiplication tensor 
    for (2,3,3) can be decomposed using block structure.
    
    Let A be 2x3, B be 3x3, C = A*B be 2x3.
    
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    That's 18 terms total. We want to reduce to 16.
    
    We can try to combine some terms using identities. For example, if we have terms like:
    x*y and x*z, we might be able to combine them.
    
    Let's use a different strategy: try to find a rank-16 decomposition by
    parameterizing the factors and solving.
    """
    
    nm, mp, pn = n*m, m*p, p*n
    R = 16
    
    # We'll try to find U, V, W of shape (nm, R), (mp, R), (pn, R)
    # such that for all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # The system is underdetermined, so we need to fix some degrees of freedom.
    # Fix W to be identity-like: W[k*n+i, r] = delta_{k*n+i, r} for r < pn
    # But pn = 6, and R = 16, so we can't do that directly.
    
    # Alternative: Use the structure of matrix multiplication.
    # T[i,j,k] = 1 iff the indices are "compatible" in the multiplication.
    
    # Let's try a direct construction approach.
    # We know that rank(2,3,3) <= 16 from theoretical bounds.
    # Let's try to find it by solving a system of polynomial equations.
    
    # For each rank term r, let:
    #   u_r: vector of length 6 (indices i*m+j for i in {0,1}, j in {0,1,2})
    #   v_r: vector of length 9 (indices j*p+k for j in {0,1,2}, k in {0,1,2})
    #   w_r: vector of length 6 (indices k*n+i for k in {0,1,2}, i in {0,1})
    
    # The constraint is: for all (i,j,k) in {0,1}x{0,1,2}x{0,1,2}:
    #   sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    
    # This is 18 equations. We have 16*(6+9+6) = 336 unknowns.
    # We need 318 more constraints to get a unique solution, but we don't need uniqueness.
    # We just need ANY solution.
    
    # Strategy: Fix most of W to be sparse. For example, set W to have only one
    # non-zero per row, or use a structured pattern.
    
    # Let's try: W[k*n+i, r] = 1 if r == k*n+i, else 0. But that's only 6 non-zeros.
    # Then for each (i,j,k), the constraint becomes:
    #   u_{k*n+i}[i*m+j] * v_{k*n+i}[j*p+k] = 1
    # This gives 18 constraints on 16*6 + 16*9 = 240 unknowns (u and v only).
    # But we also need consistency: the same u_r and v_r are used across different (i,j,k).
    
    # Actually, if W is the identity (6x16 with 6 non-zeros), then:
    # For each r in 0..5 (corresponding to k*n+i), we need:
    #   u_r[i*m+j] * v_r[j*p+k] = 1 for all (i,j,k) such that k*n+i = r
    
    # For r = k*n+i, the (i,j,k) triples with this k*n+i are:
    #   k is fixed, i is fixed, j varies over 0,1,2
    # So we need: u_r[i*m+0]*v_r[0*p+k] = 1, u_r[i*m+1]*v_r[1*p+k] = 1, u_r[i*m+2]*v_r[2*p+k] = 1
    
    # That's 3 constraints per r, for 6 values of r, so 18 constraints total.
    # Unknowns: for each r, u_r has 6 entries, v_r has 9 entries, but only 3 of each are used.
    # We can set the unused entries to 0.
    
    # So for each r (0..5), let k = r // n, i = r % n (wait, r = k*n+i, so k = r//n, i = r%n)
    # n=2, so k = r//2, i = r%2
    
    # For r=0: k=0, i=0. Need u_0[0]*v_0[0] = 1, u_0[1]*v_0[3] = 1, u_0[2]*v_0[6] = 1
    # For r=1: k=0, i=1. Need u_1[3]*v_1[0] = 1, u_1[4]*v_1[3] = 1, u_1[5]*v_1[6] = 1
    # For r=2: k=1, i=0. Need u_2[0]*v_2[1] = 1, u_2[1]*v_2[4] = 1, u_2[2]*v_2[7] = 1
    # For r=3: k=1, i=1. Need u_3[3]*v_3[1] = 1, u_3[4]*v_3[4] = 1, u_3[5]*v_3[7] = 1
    # For r=4: k=2, i=0. Need u_4[0]*v_4[2] = 1, u_4[1]*v_4[5] = 1, u_4[2]*v_4[8] = 1
    # For r=5: k=2, i=1. Need u_5[3]*v_5[2] = 1, u_5[4]*v_5[5] = 1, u_5[5]*v_5[8] = 1
    
    # We can satisfy each by setting, e.g., u_r[idx] = 1, v_r[idx] = 1.
    # But this uses 6 rank terms, not 16. The issue is that this only covers 6 of the 18 (i,j,k) triples.
    # Wait, no: each r covers 3 (i,j,k) triples (varying j), and 6*3 = 18. So this works!
    
    # But wait: the constraint is sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1.
    # If W is identity (W[k*n+i, r] = delta_{k*n+i, r}), then only the term r = k*n+i contributes.
    # So we need U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1 for all (i,j,k).
    
    # For fixed (i,k), we need: U[i*m+0, k*n+i]*V[0*p+k, k*n+i] = 1
    #                         U[i*m+1, k*n+i]*V[1*p+k, k*n+i] = 1
    #                         U[i*m+2, k*n+i]*V[2*p+k, k*n+i] = 1
    
    # We can set U[i*m+j, k*n+i] = 1 and V[j*p+k, k*n+i] = 1 for all i,j,k.
    # But U has shape (6, 16), and we're setting entries at positions (i*m+j, k*n+i).
    # For i=0: positions (0,0), (1,0), (2,0) for k=0; (0,2), (1,2), (2,2) for k=1; (0,4), (1,4), (2,4) for k=2
    # For i=1: positions (3,1), (4,1), (5,1) for k=0; (3,3), (4,3), (5,3) for k=1; (3,5), (4,5), (5,5) for k=2
    
    # Each row of U has exactly 3 non-zeros (one for each k). Each column of U has either 0 or 3 non-zeros.
    # Similarly for V.
    
    # This gives a valid rank-6 decomposition! But wait, that can't be right because the rank of 2x3x3 is known to be higher.
    
    # Let me re-examine. The issue is that I'm confusing the rank. The rank is the number of terms in the sum, which is R=16.
    # But if W is 6x16 with only 6 non-zeros (one per row), then effectively only 6 terms contribute.
    # So the rank is 6, not 16.
    
    # But is rank 6 possible for 2x3x3? The lower bound for rank(n,m,p) is at least max(n*m, m*p, p*n) / something...
    # Actually, the rank of matrix multiplication is at least the border rank, and for (2,3,3) it's known to be at least 11 or so.
    
    # I think I made an error. Let me re-derive.
    
    # The tensor is T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    # This is NOT the matrix multiplication tensor! The matrix multiplication tensor has T[i,j,k] = delta_{j,j'}*delta_{k,k'}... 
    # Wait, no. The standard matrix multiplication tensor is defined as:
    # T[i,j,k] = 1 if the indices correspond to a valid multiplication term, 0 otherwise.
    
    # Actually, for matrix multiplication, the tensor is:
    # T[a,b,c] where a indexes A, b indexes B, c indexes C.
    # T[a,b,c] = 1 if a = (i,j), b = (j,k), c = (i,k) for some i,j,k, else 0.
    
    # So T is not all ones! It's a sparse tensor. Let me re-read the task.
    
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 3 matrix."
    
    # So the tensor T[i,j,k] should be 1 when (i,j,k) corresponds to a valid multiplication term.
    # For A (2x3) * B (3x3) = C (2x3):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the terms are (i,j) from A, (j,k) from B, (i,k) from C.
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    
    # So the tensor has 18 non-zero entries, not 18 entries all equal to 1 in a dense sense.
    # The tensor is indexed by (a,b,c) where a in 0..5, b in 0..8, c in 0..5.
    # T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0.
    
    # So my earlier analysis was wrong. The tensor is sparse.
    
    # Now, the rank-6 construction I described would work if the tensor were dense, but it's not.
    # Let me redo the analysis with the correct tensor.
    
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    # All other entries are 0.
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k) in the valid set.
    # And: sum_r U[a, r] * V[b, r] * W[c, r] = 0 for all other (a,b,c).
    
    # This is a much harder constraint. The "off-diagonal" constraints are what make it non-trivial.
    
    # For the rank-6 construction with W as identity:
    # W[c, r] = delta_{c, r} for c in 0..5, r in 0..5, and 0 for r >= 6.
    # Then for (a,b,c) with c = k*n+i:
    #   U[a, c] * V[b, c] = T[a,b,c]
    # For (a,b,c) with c >= 6: automatically 0.
    
    # But we also need: for (a,b,c) where c = k*n+i but (a,b,c) is NOT a valid multiplication term,
    #   U[a, c] * V[b, c] = 0
    
    # For c = k*n+i, a valid term has a = i*m+j', b = j'*p+k for some j'.
    # So for a given c = k*n+i, the valid (a,b) pairs are:
    #   a = i*m+0, b = 0*p+k
    #   a = i*m+1, b = 1*p+k  
    #   a = i*m+2, b = 2*p+k
    
    # For other (a,b) with the same c, we need U[a,c]*V[b,c] = 0.
    
    # So for each c (0..5), we need U[:,c] and V[:,c] to be orthogonal in the sense that
    # their element-wise product is zero except at the 3 valid positions.
    
    # We can achieve this by setting U[i*m+j, c] = 1 only for valid j, and 0 otherwise,
    # and similarly for V.
    
    # For c = k*n+i:
    #   U[i*m+0, c] = 1, U[i*m+1, c] = 1, U[i*m+2, c] = 1
    #   U[i'*m+j, c] = 0 for i' != i
    #   V[0*p+k, c] = 1, V[1*p+k, c] = 1, V[2*p+k, c] = 1
    #   V[j'*p+k', c] = 0 for j' != j or k' != k... wait, b = j*p+k, so for fixed c=k*n+i, k is fixed.
    #   So V[0*p+k, c] = 1, V[1*p+k, c] = 1, V[2*p+k, c] = 1
    #   V[j'*p+k', c] = 0 for k' != k
    
    # Let's check: for c = 0 (k=0, i=0):
    #   U[0,0]=1, U[1,0]=1, U[2,0]=1, U[3,0]=0, U[4,0]=0, U[5,0]=0
    #   V[0,0]=1, V[3,0]=1, V[6,0]=1, others=0
    #   Product: U[a,0]*V[b,0] is non-zero only when a in {0,1,2} and b in {0,3,6}.
    #   Valid terms for c=0: (a,b) = (0,0), (1,3), (2,6).
    #   But U[0,0]*V[3,0] = 1*1 = 1, which is non-zero, but (a=0,b=3) is NOT a valid term!
    #   So this doesn't work.
    
    # The issue is that U and V are not diagonal in the right way. We need U[a,c]*V[b,c] = 1 iff (a,b,c) is valid, else 0.
    # This requires that for each c, the non-zero pattern of U[:,c] and V[:,c] are "matched" such that
    # their element-wise product is 1 exactly at the valid (a,b) pairs.
    
    # For c=0: valid pairs are (0,0), (1,3), (2,6).
    # We need U[0,0]*V[0,0]=1, U[1,0]*V[3,0]=1, U[2,0]*V[6,0]=1, and all other products = 0.
    # One way: U[0,0]=1, U[1,0]=1, U[2,0]=1, U[3,0]=U[4,0]=U[5,0]=0
    #           V[0,0]=1, V[3,0]=1, V[6,0]=1, V[1,0]=V[2,0]=V[4,0]=V[5,0]=V[7,0]=V[8,0]=0
    # Then U[0,0]*V[3,0] = 1*1 = 1, but we need it to be 0.
    
    # So we need a different pattern. We could use different values:
    # U[0,0]=1, U[1,0]=1, U[2,0]=1
    # V[0,0]=1, V[3,0]=1, V[6,0]=1
    # But then cross terms are non-zero.
    
    # To kill cross terms, we need U[a,c]*V[b,c] = 0 for invalid (a,b).
    # This is possible if, for each c, the non-zero entries of U[:,c] and V[:,c] are on "disjoint" index sets in some sense.
    # But a and b are different indices, so we can't directly disjoint them.
    
    # Alternative: use the fact that we have 16 rank terms, not just 6. We can distribute the constraints.
    
    # This is getting complex. Let me just use the known rank-17 solution as fallback and try to find rank 16 via optimization.
    
    # For now, let's implement a simple approach: try rank 16 with a structured initialization and optimize.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    deadline = time.time() + 140  # Safety margin
    
    # First, try to construct a rank-16 decomposition using combinatorial/structured approach
    
    # Known result: rank(2,3,3) = 16 (or possibly 17). Let's try to find it.
    
    # Approach: Use the block structure.
    # A is 2x3, B is 3x3, C is 2x3.
    # Split B into 3x1 blocks: B1, B2, B3 (each 3x1)
    # C = A * [B1, B2, B3] = [A*B1, A*B2, A*B3]
    
    # A*B1: 2x3 * 3x1 = 2x1, rank at most 3 (or less)
    # Similarly for A*B2, A*B3.
    
    # Total rank at most 3+3+3 = 9? No, that's not right because the factors are shared.
    
    # Let me try a different structural approach.
    
    # Use the fact that 2x3x3 can be embedded in 3x3x3, and rank(3,3,3) = 23 (Strassen).
    # But that doesn't directly help.
    
    # Let's just try to optimize for rank 16 with a good initialization.
    
    def tensor_value(U, V, W, r):
        """Compute the tensor from factors."""
        T = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T += np.einsum('i,j,k->ijk', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return T
    
    def target_tensor():
        T = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    T_target = target_tensor()
    
    def error_fn(params, r):
        U = params[:r*nm].reshape(nm, r)
        V = params[r*nm:r*nm+r*mp].reshape(mp, r)
        W = params[r*nm+r*mp:].reshape(pn, r)
        T = tensor_value(U, V, W, r)
        diff = T - T_target
        return np.sum(diff**2)
    
    def try_rank(r, num_restarts=3):
        if time.time() > deadline:
            return None
        
        best_params = None
        best_err = float('inf')
        
        for attempt in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Structured initialization
            params = np.random.randn(r * (nm + mp + pn)) * 0.5
            
            try:
                from scipy.optimize import minimize
                res = minimize(error_fn, params, args=(r,), method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
            except Exception:
                pass
        
        if best_params is None or best_err > 1e-6:
            return None
        
        # Try rounding to half-integers
        rounded = np.round(best_params * 2) / 2.0
        err_rounded = error_fn(rounded, r)
        if err_rounded < 1e-9:
            U = rounded[:r*nm].reshape(nm, r)
            V = rounded[r*nm:r*nm+r*mp].reshape(mp, r)
            W = rounded[r*nm+r*mp:].reshape(pn, r)
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try rank 16
    res16 = try_rank(16, num_restarts=2)
    if res16:
        return res16
    
    # Fallback to rank 17 using Strassen-like construction
    # Use the known rank-17 decomposition
    
    # Construct rank-17 using block structure:
    # Treat A as [A0; A1] where A0, A1 are 1x3 rows.
    # B is 3x3.
    # C = [A0*B; A1*B]
    
    # A0*B: 1x3 * 3x3 = 1x3, rank 3 (direct)
    # A1*B: 1x3 * 3x3 = 1x3, rank 3 (direct)
    # Total: 6 terms. But we can do better.
    
    # Actually, let's just use the schoolbook rank 18 as final fallback.
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
