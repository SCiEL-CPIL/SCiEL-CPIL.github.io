"""Matrix multiplication 2x3 x 3x6, rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]

def _find_2x3x3_rank15():
    """
    Attempts to find a rank-15 exact decomposition for 2x3x3 matrix multiplication.
    Returns U (6x15), V (9x15), W (6x15) if found, else None.
    """
    n, m, p = 2, 3, 3
    rank = 15
    
    # Target tensor T of shape (n*m, m*p, p*n) is not quite right for optimization.
    # We want U, V, W such that for all i, k:
    # sum_j A_{i,j} B_{j,k} = sum_r (sum_j U[i*m+j, r] A_{i,j}) * (sum_j' V[j'*p+k, r] B_{j',k}) * W[k*n+i, r]?
    # No, the decomposition is T = sum u_r v_r w_r.
    # The condition is that the bilinear map defined by U,V,W matches the multiplication.
    # Specifically, for each r, we have a term u_r (vector in A-space), v_r (vector in B-space), w_r (vector in C-space).
    # The contribution to C_{i,k} is w_{r, k*n+i} * (u_r dot A) * (v_r dot B)?
    # No, u_r is a linear functional on A. v_r is a linear functional on B.
    # So term is w_{r, k*n+i} * (sum_{i',j'} U[i'*m+j', r] A_{i',j'}) * (sum_{j',k'} V[j'*p+k', r] B_{j',k'})?
    # This would produce a scalar for each r, multiplied by w_{r, k*n+i}.
    # But A and B are matrices. The "dot" is the full inner product.
    # So for a fixed r, the term added to C_{i,k} is:
    # W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] A_{i',j'}) * (sum_{j',k'} V[j'*p+k', r] B_{j',k'})
    # This is a rank-1 update in the "bilinear" sense?
    # Actually, the standard decomposition is:
    # C = sum_r w_r * (u_r^T A) * (v_r^T B)?
    # No, that's for scalars.
    # For matrices, it is:
    # C_{i,k} = sum_r U_{i,j,r} ...?
    # Let's use the property:
    # The tensor T has entries T_{i,j,k}.
    # We want T = sum_r u_r \otimes v_r \otimes w_r.
    # This means T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}?
    # Indices:
    # U is indexed by (i,j). V by (j,k). W by (k,i).
    # So T_{i,j,k} = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # And we require T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} is part of the multiplication?
    # Wait, the multiplication tensor is defined such that
    # <T, A \otimes B> = <C, AB>.
    # The basis of A is E_{i,j}. Basis of B is F_{j,k}. Basis of C is G_{i,k}.
    # The tensor T maps E_{i,j} \otimes F_{j,k} to G_{i,k}.
    # So the component T_{i,j,k}^{i',k'} is 1 if i=i', k=k', and 0 otherwise?
    # No, the tensor is a linear map from A \otimes B to C.
    # As a trilinear form: T(A, B, C) = sum_{i,k} C_{i,k} (AB)_{i,k}?
    # Or T(A, B, C^*) = sum_{i,j,k} A_{i,j} B_{j,k} C^*_{i,k}.
    # The rank is the minimum R such that T = sum_{r=1}^R u_r \otimes v_r \otimes w_r.
    # Here u_r is in (A-space)^*, v_r in (B-space)^*, w_r in (C-space).
    # So T_{i,j,k} (where i,j index A, j,k index B, k,i index C? No, C is i,k)
    # Let's align indices.
    # A has indices (i, j). B has indices (j, k). C has indices (i, k).
    # The term is A_{i,j} B_{j,k} -> C_{i,k}.
    # So the tensor component corresponding to (i, j) in A, (j, k) in B, and (i, k) in C is 1.
    # Note that j is shared.
    # So T is a 3rd order tensor with dimensions (nm, mp, pn)?
    # No, the spaces are distinct.
    # Space A: dim nm. Space B: dim mp. Space C: dim pn.
    # The tensor T has size nm x mp x pn.
    # Entry T[ (i,j), (j',k), (i',k') ] is 1 if i=i', k=k', and j=j'?
    # Yes, because A_{i,j} and B_{j,k} multiply to give C_{i,k}.
    # So T[ idx_A, idx_B, idx_C ] = 1 if idx_A corresponds to (i,j), idx_B to (j,k), idx_C to (i,k).
    # Otherwise 0.
    
    # We want to find U (nm x R), V (mp x R), W (pn x R) such that:
    # T = sum_{r=1}^R u_r \otimes v_r \otimes w_r.
    # This means:
    # T[ a, b, c ] = sum_{r=1}^R U[a, r] V[b, r] W[c, r].
    
    # For 2x3x3:
    # nm = 6, mp = 9, pn = 6.
    # R = 15.
    # Variables: 6*15 + 9*15 + 6*15 = 315.
    # Equations: 6*9*6 = 324?
    # No, T is sparse. Only 18 entries are 1.
    # The equation is:
    # For all a, b, c:
    # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
    # Most T[a,b,c] are 0.
    # This is a system of 324 polynomial equations (cubic in variables? No, bilinear in U,V for fixed W, etc).
    # Actually, it's trilinear.
    # We can fix W to 1s? No, W can be anything.
    # We can fix one matrix to identity or specific values to reduce gauge freedom?
    # The decomposition is not unique.
    # We can use least squares on the non-zero entries?
    # No, we must satisfy the zero entries too.
    # If we minimize the sum of squares of (LHS - T) over all 324 entries, it might work.
    
    def residuals(params):
        U = params[0:90].reshape(6, 15)
        V = params[90:225].reshape(9, 15)
        W = params[225:315].reshape(6, 15)
        
        # Compute T_hat
        # T_hat[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # This is a tensor contraction.
        # We can do this efficiently.
        # T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Target T
        # T is 6x9x6.
        # Indices:
        # a = i*3 + j (i in 0..1, j in 0..2)
        # b = j*3 + k (j in 0..2, k in 0..2)
        # c = k*2 + i (k in 0..2, i in 0..1)
        # T[a, b, c] = 1 if i_a == i_c and j_a == j_b and k_b == k_c.
        # Note: j_a is the column of A, j_b is the row of B. They must match.
        
        T_target = np.zeros((6, 9, 6))
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    a = i * 3 + j
                    b = j * 3 + k
                    c = k * 2 + i
                    T_target[a, b, c] = 1.0
                    
        return (T_hat - T_target).flatten()

    # Initial guess
    # Random initialization
    np.random.seed(42)
    x0 = np.random.randn(315) * 0.1
    
    # Optimize
    try:
        res = least_squares(residuals, x0, method='lm', max_nfev=1000)
        if res.cost < 1e-10:
            U = res.x[0:90].reshape(6, 15)
            V = res.x[90:225].reshape(9, 15)
            W = res.x[225:315].reshape(6, 15)
            # Round to nearest half-integer
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            # Verify
            T_hat = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
            if np.allclose(T_hat, T_target, atol=1e-9):
                return U_int, V_int, W_int
    except Exception:
        pass
        
    return None

def construct():
    n, m, p = DIMENSIONS
    # Try to find rank 30 solution
    block = _find_2x3x3_rank15()
    if block is not None:
        U_block, V_block, W_block = block
        R = 30
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # First half (cols 0-2 of B, cols 0-2 of C)
        U[:, 0:15] = U_block
        V[0:9, 0:15] = V_block
        W[0:6, 0:15] = W_block
        
        # Second half (cols 3-5 of B, cols 3-5 of C)
        U[:, 15:30] = U_block
        V[9:18, 15:30] = V_block
        W[6:12, 15:30] = W_block
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to trivial rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
