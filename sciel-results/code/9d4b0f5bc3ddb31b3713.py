"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _snap_to_rational(arr, max_denom=100):
    """Snap a numpy array of floats to nearest rational with denominator <= max_denom."""
    result = np.zeros_like(arr, dtype=object)
    for idx in np.ndindex(arr.shape):
        val = float(arr[idx])
        if abs(val) < 1e-9:
            result[idx] = Fraction(0, 1)
        else:
            # Try to find a simple fraction
            frac = Fraction(val).limit_denominator(max_denom)
            if abs(float(frac) - val) < 1e-6:
                result[idx] = frac
            else:
                # Try half-integers
                for d in [2, 4, 8, 16]:
                    num = round(val * d)
                    if abs(float(num) / d - val) < 1e-6:
                        result[idx] = Fraction(num, d)
                        break
                else:
                    result[idx] = Fraction(val).limit_denominator(max_denom)
    return result

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] for (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The coefficient of A[i,j]*B[j,k] in (A*B)[i,k] is 1
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
                # Check that other entries are zero
                # Actually we need to check the full tensor equality
    # More thorough check: build the full tensor and compare
    T_reconstructed = np.zeros((n, m, p, n, p))  # This is too big, let's do it differently
    
    # The multiplication tensor T has T[i,j,k,l] where (A*B)[l,?] ... 
    # Actually the standard form: T is a 4-index tensor T[i,j,k,l] such that
    # (A*B)[i,l] = sum_{j,k} A[i,j] * B[j,k] * T[i,j,k,l] where T[i,j,k,l] = delta_{i,l}... no
    # 
    # The tensor for matrix multiplication: T[i,j,k] where the bilinear form is
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # As a tensor: T is 3-tensor of shape (n*m, m*p, n*p) with T[i*m+j, j*p+k, k*n+i]... 
    # Wait, the standard decomposition: 
    # (A*B)[i,k] = sum_c U[i*m+j?, ...]
    # 
    # Let me re-read the convention. The task says:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W[k*n+i, r]
    # 
    # The tensor equality is:
    # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for all other combinations, it should be 0.
    # 
    # Actually the full tensor has entries indexed by (i,j) for U, (j,k) for V, (k,i) for W
    # The product U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should give the tensor entry.
    # The tensor T[i,j,k] (for the multiplication) is 1 if this corresponds to a valid
    # multiplication term, i.e., T[i,j,k] = 1 for all i,j,k since (A*B)[i,k] = sum_j A[i,j]B[j,k]
    # 
    # But we need to check that non-diagonal terms are zero. Let's think of it as:
    # The full multiplication tensor M has shape (n*m, m*p, n*p) where
    # M[a,b,c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i' with i=i', j=j', k=k'... 
    # No, that's not right either.
    # 
    # The correct interpretation: The tensor T is such that
    # (A*B)[i,k] = sum_{j} A[i,j] * B[j,k]
    # In tensor form, if we flatten A to vector of length n*m, B to length m*p, 
    # and (A*B) to length n*p, then (A*B) = T(A, B) where T is a 3-way tensor.
    # T[a, b, c] where a indexes A, b indexes B, c indexes (A*B).
    # T[a,b,c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i' with j=j' and i=i' and k=k'... 
    # Wait no. (A*B)[i,k] = sum_j A[i,j] B[j,k]. So the output index is (i,k), 
    # which in the W convention is k*n+i. So c = k*n+i.
    # a = i*m+j, b = j*p+k, c = k*n+i.
    # For the sum to work, we need j from A to match j from B, and i from A to match i in output,
    # and k from B to match k in output.
    # So T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    # This means T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, and 0 otherwise.
    # 
    # The decomposition says: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # So we need: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k
    # And: sum_r U[a, r] V[b, r] W[c, r] = 0 for all (a,b,c) not of the form above.
    
    # Let's do a full check
    T_target = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Reconstruct T from U, V, W
    # T_reconstructed[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # This is equivalent to: for each r, add outer(U[:,r], V[:,r], W[:,r])
    T_rec = np.zeros((n*m, m*p, n*p))
    for r in range(rank):
        T_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    if not np.allclose(T_rec, T_target, atol=tol):
        return False
    return True

def _try_rank(rank, n, m, p, time_limit=100):
    """
    Try to find a rank-`rank` decomposition using a heuristic approach.
    We use a simple iterative method: start with random initialization and use
    gradient-free updates, or use the fact that for small ranks we can set up
    a system of equations.
    """
    start_time = time.time()
    
    # For rank < n*m*p, we need a non-trivial decomposition.
    # The known lower bound for 3x3x7 is at least max(9, 21, 21) = 21? No.
    # The border rank lower bound: for multiplying (n x m) by (m x p), 
    # the rank is at least max(n*m, m*p, n*p) in some cases, but that's not tight.
    # Actually, the trivial lower bound is max(n*m, m*p, n*p) is wrong.
    # The rank must be at least max(n*m, m*p, n*p) only if... no.
    # 
    # For 3x3x7: n*m=9, m*p=21, n*p=21. The rank is at least 21? 
    # No, the rank is at least max(n*m, m*p, n*p) is NOT a valid lower bound.
    # The valid lower bound is: rank >= max(n*m, m*p, n*p) is false.
    # Actually, each of the three modes has dimensions n*m, m*p, n*p.
    # The rank R must satisfy that the tensor fits in R "terms".
    # A simple lower bound: R >= max(n*m, m*p, n*p) is not correct.
    # The correct simple lower bound: R >= n*m (since the U matrix is n*m x R, 
    # and the tensor has n*m*p non-zero entries in a specific pattern).
    # 
    # Let's just try to find a decomposition with rank < 63.
    # Known results: 3x3x3 has rank 23 (Landsberg). 3x3x7 is less well-studied.
    # The schoolbook rank is 3*3*7 = 63.
    # Can we do better? For square matrices, Strassen's algorithm gives rank 7 for 2x2x2.
    # For 3x3, the best known is 23 (Landsberg 1989).
    # For 3x3x7, we might be able to do better than 63.
    
    # Strategy: Use a random initialization and try to solve the system.
    # The system is: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # And for all other (a,b,c), the sum = 0.
    
    # This is a system of (n*m)*(m*p)*(n*p) - (n*m*p) equations... that's huge.
    # 9*21*21 = 3969 total entries, 63 are 1, rest are 0.
    # That's 3906 equations in 3*rank*... variables.
    
    # Alternative: Use the fact that we can fix one factor and solve for the others.
    # Let's try a simpler approach: use known algorithms or patterns.
    
    # For now, let's try a randomized approach with local search.
    # Initialize U, V, W randomly and try to reduce the error.
    
    np.random.seed(42)
    best_error = np.inf
    best_U = None
    best_V = None
    best_W = None
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try a few random initializations
    for trial in range(50):
        if time.time() - start_time > time_limit:
            break
        
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(n*p, rank) * 0.1
        
        # Simple iterative improvement: update one factor at a time
        for iteration in range(200):
            if time.time() - start_time > time_limit:
                break
            
            # Update U: for each row a, find U[a,:] that minimizes error
            # This is a linear system if V and W are fixed
            # error = ||T_target - sum_r U[:,r] outer(V[:,r], W[:,r])||^2
            # For fixed V, W, the optimal U is:
            # U = T_target * (V ⊙ W)^+ where ⊙ is Khatri-Rao product
            # M = V ⊙ W has shape (m*p * n*p, rank)
            # T_target reshaped: T_target.reshape(n*m, m*p*n*p)
            # U = T_target.reshape(n*m, -1) @ M @ pinv(M.T @ M)
            
            M = np.kron(V, W)  # This is not quite right; Khatri-Rao is column-wise Kronecker
            # Khatri-Rao product: M[:,r] = V[:,r] ⊗ W[:,r]
            M = np.zeros((m*p * n*p, rank))
            for r in range(rank):
                M[:, r] = np.kron(V[:, r], W[:, r])
            
            T_flat = T_target.reshape(n*m, -1)
            # Solve: T_flat ≈ U @ M.T
            # U = T_flat @ M @ (M.T @ M)^-1
            try:
                MM = M.T @ M
                U_new = T_flat @ M @ np.linalg.pinv(MM)
            except:
                continue
            
            # Similarly update V and W
            M2 = np.kron(U, W)
            for r in range(rank):
                M2[:, r] = np.kron(U[:, r], W[:, r])
            T_flat2 = T_target.transpose(1, 0, 2).reshape(m*p, -1)
            try:
                MM2 = M2.T @ M2
                V_new = T_flat2 @ M2 @ np.linalg.pinv(MM2)
            except:
                continue
            
            M3 = np.kron(U, V)
            for r in range(rank):
                M3[:, r] = np.kron(U[:, r], V[:, r])
            T_flat3 = T_target.transpose(2, 0, 1).reshape(n*p, -1)
            try:
                MM3 = M3.T @ M3
                W_new = T_flat3 @ M3 @ np.linalg.pinv(MM3)
            except:
                continue
            
            U, V, W = U_new, V_new, W_new
        
        # Compute final error
        T_rec = np.zeros((n*m, m*p, n*p))
        for r in range(rank):
            T_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        error = np.linalg.norm(T_rec - T_target)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_error < 1e-8:
            break
    
    if best_error < 1e-8:
        # Snap to rational
        U_frac = _snap_to_rational(best_U)
        V_frac = _snap_to_rational(best_V)
        W_frac = _snap_to_rational(best_W)
        
        # Verify
        if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p, tol=1e-9):
            return U_frac, V_frac, W_frac
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix
    by a 3 by 7 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # The schoolbook rank is n*m*p = 63
    schoolbook_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # Known: for 3x3x3, best known rank is 23 (vs schoolbook 27)
    # For 3x3x7, we don't know the exact minimum, but it should be < 63.
    # Let's try ranks from a reasonable lower bound up to 63.
    
    # Lower bound: at least max(n*m, m*p, n*p) = 21? No, that's not a valid lower bound.
    # A valid lower bound: the rank must be at least the matrix rank of any unfolding.
    # The (n*m) x (m*p) unfolding has rank at most n*p, but that doesn't give a lower bound on R.
    # Actually, the rank R of the tensor must satisfy R >= matrix rank of any matricization.
    # The matricization along mode 1 has shape (n*m) x (m*p * n*p) and has rank <= min(n*m, m*p*n*p).
    # The actual matrix rank of the multiplication tensor matricization is n*m = 9 (since each row
    # corresponds to a unique (i,j) pair and has support only on entries involving that (i,j)).
    # So the lower bound from matricization is 9, which is weak.
    
    # Let's just try to find any decomposition with rank < 63.
    # Start from rank 21 (a guess) and go up.
    
    # Actually, let's first check if we can find a decomposition with rank < 63
    # using the iterative method. If not, return schoolbook.
    
    # Try a few ranks
    for rank in range(21, schoolbook_rank):
        if time.time() > deadline - 30:
            break
        result = _try_rank(rank, n, m, p, time_limit=min(20, deadline - time.time() - 10))
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook decomposition
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
