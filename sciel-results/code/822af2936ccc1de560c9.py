import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify the tensor decomposition is exact using numpy."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    try:
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        # result[i, j, k] = sum_s U[i, j, s] * V[j, k, s] * W[k, i, s]
        # Note: W is indexed by k (first dim), i (second dim) in the tensor product
        # W shape is (p, n, r), so W[k, i, s]
        result = np.einsum('ijs,jks,ks->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        return np.allclose(result, 1.0, atol=1e-9)
    except Exception:
        return False

def _rank11_solution():
    n, m, p = DIMENSIONS
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def set_u(i, j, r, val):
        U[i * m + j, r] = val

    def set_v(j, k, r, val):
        V[j * p + k, r] = val

    def set_w(k, i, r, val):
        W[k * n + i, r] = val

    # Strassen for 2x2 block (columns 0,1 of B)
    set_u(0,0,0,1); set_u(1,1,0,1)
    set_v(0,0,0,1); set_v(1,1,0,1)
    set_w(0,0,0,1); set_w(1,1,0,1)

    set_u(1,0,1,1); set_u(1,1,1,1)
    set_v(0,0,1,1)
    set_w(0,1,1,1); set_w(1,1,1,-1)

    set_u(0,0,2,1)
    set_v(0,1,2,1); set_v(1,1,2,-1)
    set_w(1,0,2,1); set_w(1,1,2,1)

    set_u(1,1,3,1)
    set_v(1,0,3,1); set_v(0,0,3,-1)
    set_w(0,0,3,1); set_w(0,1,3,1)

    set_u(0,0,4,1); set_u(0,1,4,1)
    set_v(1,1,4,1)
    set_w(0,0,4,-1); set_w(1,0,4,1)

    set_u(1,0,5,1); set_u(0,0,5,-1)
    set_v(0,0,5,1); set_v(0,1,5,1)
    set_w(1,1,5,1)

    set_u(0,1,6,1); set_u(1,1,6,-1)
    set_v(1,0,6,1); set_v(1,1,6,1)
    set_w(0,0,6,1)

    # Schoolbook for column 2
    set_u(0,0,7,1); set_v(0,2,7,1); set_w(2,0,7,1)
    set_u(0,1,8,1); set_v(1,2,8,1); set_w(2,0,8,1)
    set_u(1,0,9,1); set_v(0,2,9,1); set_w(2,1,9,1)
    set_u(1,1,10,1); set_v(1,2,10,1); set_w(2,1,10,1)

    return U, V, W

def _try_rank10_search(deadline):
    """
    Search for a rank-10 solution by looking for linear dependencies
    among the 11 rank-1 terms. We model this as: can we find a non-trivial
    linear combination of the 11 terms that equals zero in the tensor?
    
    The tensor T = sum_{s=1}^{11} U_s ⊗ V_s ⊗ W_s
    We look for coefficients c_1,...,c_11 (not all zero) such that
    sum_s c_s * (U_s ⊗ V_s ⊗ W_s) = 0
    
    This is a syzygy. If such a syzygy exists, we can eliminate one term.
    """
    n, m, p = DIMENSIONS
    U, V, W = _rank11_solution()
    
    # The 11 rank-1 tensors: U_s (4-dim), V_s (6-dim), W_s (6-dim)
    # Flatten each rank-1 tensor to a vector of size 4*6*6 = 144
    # Then check if these 11 vectors are linearly dependent
    
    tensors = []
    for s in range(11):
        u_s = U[:, s]  # (4,)
        v_s = V[:, s]  # (6,)
        w_s = W[:, s]  # (6,)
        # Outer product flattened
        t_s = np.outer(np.outer(u_s, v_s), w_s).flatten()  # (144,)
        tensors.append(t_s)
    
    T = np.array(tensors).T  # (144, 11)
    
    # Check rank
    rank = np.linalg.matrix_rank(T, tol=1e-9)
    
    if rank < 11:
        # Linear dependence exists! Find the null space vector
        # Use SVD
        U_svd, s_vals, Vt = np.linalg.svd(T, full_matrices=False)
        # The last column of Vt corresponds to the null space
        c = Vt[-1]  # coefficients for the linear combination
        
        # Verify: sum_s c_s * tensor_s ≈ 0
        combo = T @ c
        if np.allclose(combo, 0, atol=1e-9):
            # Find which coefficient has the largest magnitude
            idx = np.argmax(np.abs(c))
            
            # We can eliminate term idx. The new decomposition uses the other 10 terms
            # but with modified coefficients. Actually, we need to be more careful.
            # 
            # The syzygy says: sum_s c_s * T_s = 0
            # So T_idx = -(1/c_idx) * sum_{s != idx} c_s * T_s
            # 
            # This means the original decomposition T = sum_{s=1}^{11} T_s can be
            # rewritten. But we need to express it as a sum of 10 rank-1 terms.
            #
            # Actually, the syzygy tells us the 11 rank-1 tensors are linearly dependent,
            # but that doesn't directly give us a rank-10 decomposition of the original
            # multiplication tensor. We need a different approach.
            #
            # Let me think differently. The multiplication tensor is:
            # M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * delta_{i,j,k}
            # where for 2x2x3, the "delta" picks out the right terms.
            #
            # Actually, let me just try a direct rank-10 search using the structure.
            pass
    
    # Alternative approach: directly search for rank-10 by parameterizing
    # the decomposition with the Strassen structure for the first 2 columns
    # and allowing the 3rd column to be computed from linear combinations
    
    # For 2x2x3, the multiplication tensor has 2*2*3 = 12 nonzero entries.
    # A rank-r decomposition gives r*4*6*6 = 144r parameters, but with constraints.
    
    # Let me try a more direct approach: assume the rank-10 solution has
    # the form where the first 7 terms are Strassen-like for columns 0,1
    # and the remaining 3 terms handle column 2 with some sharing.
    
    # Actually, the simplest rank-10 candidate: use Strassen for 2x2x2 (rank 7)
    # and add 3 more terms for the 3rd column, but allow those 3 terms to
    # also contribute to columns 0,1 in a way that's consistent.
    
    # Let me try: rank = 7 (Strassen for cols 0,1) + 3 (for col 2) = 10
    # But the 3 terms for col 2 might be able to be "absorbed" if they
    # have the right structure.
    
    # Actually, I think the key insight is: for 2x2x3, the border rank might be 10
    # but the exact rank is 11. Let me verify by trying to find a rank-10 solution
    # using a more systematic search.
    
    # Try: can we find a rank-10 solution where the W matrix has a special structure?
    # W is (p*n, rank) = (6, 10). For column 2 of output, W[2*2:2*2+2, :] = W[4:6, :]
    # These 2 rows correspond to output entries C[0,2] and C[1,2].
    
    # Let me just try a random search with the constraint that the solution must
    # be exact.
    
    start = time.time()
    np.random.seed(42)
    
    # Try structured rank-10: 7 Strassen terms + 3 additional terms
    # The 3 additional terms should primarily handle column 2
    # but can have small contributions to columns 0,1 that cancel out
    
    best_solution = None
    
    for attempt in range(1000):
        if time.time() > deadline:
            break
        
        # Start with Strassen for 2x2x2
        U = np.zeros((4, 10))
        V = np.zeros((6, 10))
        W = np.zeros((6, 10))
        
        # Strassen terms (0-6)
        # Term 0: (A00+A11)(B00+B11) -> C00 +=, C11 +=
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1
        
        # Term 1: (A10+A11)B00 -> C10 +=, C11 -=
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[1,1]=1; W[3,1]=-1
        
        # Term 2: A00(B10-B00) -> C01 +=, C11 +=
        U[0,2]=1
        V[2,2]=1; V[1,2]=-1
        W[2,2]=1; W[3,2]=1
        
        # Term 3: A11(B01-B10) -> C00 +=, C01 +=
        U[3,3]=1
        V[2,3]=1; V[1,3]=-1
        W[0,3]=1; W[1,3]=1
        
        # Term 4: (A00+A01)B11 -> C00 -=, C10 +=
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1; W[2,4]=1
        
        # Term 5: (A10-A00)(B00+B01) -> C11 +=
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1
        
        # Term 6: (A01-A11)(B10+B11) -> C00 +=
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1
        
        # Now terms 7-9 for column 2. These should primarily compute:
        # C[0,2] = A00*B02 + A01*B12
        # C[1,2] = A10*B02 + A11*B12
        
        # Try different structures for the last 3 terms
        # Option: use 3 terms that compute the 4 products with some sharing
        
        # Let's try: 
        # Term 7: A00 * B02
        # Term 8: A01 * B12  
        # Term 9: (A10+A11) * B02, and we need A11*B12 too...
        
        # Actually, we need 4 products but only have 3 terms.
        # Let's try to combine:
        # Term 7: A00 * B02 -> contributes to C[0,2]
        # Term 8: A01 * B12 -> contributes to C[0,2]
        # Term 9: (A10 + a*A11) * B02 + b * A11 * B12
        # This is getting complicated. Let me just try random coefficients.
        
        # Initialize last 3 terms with random values, then optimize
        U[0:4, 7:10] = np.random.randn(4, 3)
        V[0:6, 7:10] = np.random.randn(6, 3)
        W[0:6, 7:10] = np.random.randn(6, 3)
        
        # Check if this is close to a solution
        if _verify(U, V, W):
            best_solution = (U.copy(), V.copy(), W.copy())
            break
        
        # Try a few iterations of gradient descent to fix the last 3 terms
        # while keeping the first 7 fixed
        
    if best_solution is not None:
        return best_solution
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    U, V, W = _rank11_solution()
    
    if _verify(U, V, W):
        # Try to find a rank-10 solution
        rank10 = _try_rank10_search(deadline)
        if rank10 is not None:
            U10, V10, W10 = rank10
            if _verify(U10, V10, W10):
                return {"U": U10.tolist(), "V": V10.tolist(), "W": W10.tolist()}
    
    # Fallback to rank 11
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
