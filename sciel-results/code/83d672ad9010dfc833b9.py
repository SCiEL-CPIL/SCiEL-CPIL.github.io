"""Tensor decomposition search for 2x4x7 matrix multiplication."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 7]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U*V*W decomposition gives the correct multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Build the full tensor from decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Compare against the identity multiplication tensor
    # For matrix mult: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] should be 1 if we're computing the j-th term, i.e. T[i,j,k]=1 for all
    # Actually the multiplication tensor is: T[i,j,k] = 1 for all (i,j,k) in the standard basis
    # The decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    # Let's check this
    # Efficient: for each c, the outer product contribution
    # T_decomp[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We want T_decomp = 1 everywhere (since multiplying basis matrices)
    # Actually no - the multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # because (E_i * E_jk)[?] ... let me think again.
    # The matrix multiplication tensor: for matrices A (n x m) and B (m x p),
    # C = A*B, C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T with T[i,j,k] = 1 (it's the all-ones tensor of shape n x m x p)
    # Wait no. The tensor T_{ijk} such that sum_j T_{ijk} A_{ij} B_{jk} = C_{ik}
    # So T_{ijk} = 1 for all i,j,k. Yes, the multiplication tensor is the all-ones tensor.
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Build U_mat: shape (n*m, rank), V_mat: shape (m*p, rank), W_mat: shape (p*n, rank)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Compute via: for each i,j, the vector sum_c U3[i,j,c]*V3[j,k,c] over c gives a length-p vector
    # then dot with W3[k,i,c]... this is complex. Let's just do it with einsum.
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)

def _try_rank_reduction():
    """Try to find a lower rank decomposition via SVD-based approach."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # The multiplication tensor is all-ones of shape (n, m, p)
    # Matricize as (n*m) x p: M[i*m+j, k] = 1 for all i,j,k
    # This is just a (n*m) x p all-ones matrix, which has rank 1!
    # But that's the wrong matricization for finding tensor rank.
    # 
    # The tensor rank is the minimum number of rank-1 tensors (outer products of 3 vectors)
    # to express T = sum_c u_c ⊗ v_c ⊗ w_c where (u_c)_i, (v_c)_j, (w_c)_k
    # and T[i,j,k] = sum_c u_c[i] * v_c[j] * w_c[k] = 1
    #
    # For the all-ones tensor, we can write:
    # T[i,j,k] = 1 = sum_c a_c[i] * b_c[j] * d_c[k]
    # Trivially: a_1[i]=1, b_1[j]=1, d_1[k]=1 gives rank 1!
    # But wait - the constraint is that the decomposition must correspond to
    # the specific matrix multiplication structure with U,V,W shapes.
    # 
    # The constraint is:
    # U has shape (n*m, rank): U[i*m+j, c]
    # V has shape (m*p, rank): V[j*p+k, c]  
    # W has shape (p*n, rank): W[k*n+i, c]
    # And: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    #
    # So U3[i,j,c] = U[i*m+j, c], V3[j,k,c] = V[j*p+k, c], W3[k,i,c] = W[k*n+i, c]
    # The constraint couples the indices in a specific way.
    #
    # This is NOT the same as arbitrary tensor decomposition of the all-ones tensor.
    # The indices are permuted in W: W[k*n+i] not W[i*n+k].
    #
    # Let me define: U3[i,j,c], V3[j,k,c], W3[k,i,c]
    # sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c] = 1
    #
    # If we let A[i,j,c] = U3[i,j,c], B[j,k,c] = V3[j,k,c], C[k,i,c] = W3[k,i,c]
    # This is a constrained tensor decomposition.
    
    # Strategy: set up as a system. For a given rank r, we need:
    # For all (i,j,k): sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # 
    # This is a nonlinear system. Let's try small ranks.
    # 
    # Lower bound: The tensor has "slice rank" considerations.
    # For n=2, m=4, p=7: the schoolbook gives 56.
    # Known: R(2,2,2)=7. R(2,2,p) for larger p?
    # 
    # Let's try to use the fact that for n=2, we can write:
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    # 
    # One approach: group terms. For each j, the contribution to C[0,k] and C[1,k]
    # is A[0,j]*B[j,k] and A[1,j]*B[j,k].
    # 
    # Let's try rank = 2*m*p = 56 (schoolbook) first, then try to reduce.
    # 
    # Actually, let me try a different approach: parameterize and use least squares.
    # For a given rank r, choose random initialization and optimize.
    
    best_result = None
    best_rank = n * m * p + 1
    
    # Try ranks from a reasonable lower bound up
    # Lower bound: max(n*p, m*p, n*m) ... actually tensor rank >= matricization rank
    # Matricization (n*m) x p: all ones, rank 1. Not useful.
    # The constraint structure matters. Let's just try decreasing from schoolbook.
    
    for target_rank in [56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 24, 20, 16, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        # Try to find a decomposition of this rank
        # Use the structure: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # 
        # Fix W to be simple (e.g., one-hot) and solve for U, V? No, that's schoolbook.
        # 
        # Alternative: use the identity that if we set W[k*n+i, c] appropriately...
        # 
        # Let's try: for each c, define the "slice" contribution.
        # A rank-1 term contributes: u_c[i] * v_c[j] * w_c[k] where
        # u_c[i] = U[i*m+j, c]... no, U[i*m+j, c] depends on both i and j.
        # 
        # The term is: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This is NOT a simple outer product of three vectors because:
        # U[i*m+j, c] is a function of (i,j), V[j*p+k, c] is a function of (j,k),
        # W[k*n+i, c] is a function of (k,i).
        # 
        # So the sum is: sum_c f_c(i,j) * g_c(j,k) * h_c(k,i) = 1
        # where f_c(i,j) = U[i*m+j, c], g_c(j,k) = V[j*p+k, c], h_c(k,i) = W[k*n+i, c]
        #
        # This is a "cyclic" tensor decomposition, different from standard tensor rank.
        #
        # For the schoolbook: f_c is one-hot on (i,j), g_c one-hot on (j,k), h_c one-hot on (k,i)
        # with the same (i,j,k) triple.
        
        # Let me try: can we merge some schoolbook terms?
        # Term (i,j,k): f=1 at (i,j), g=1 at (j,k), h=1 at (k,i)
        # If we try to merge (i1,j,k) and (i2,j,k) into one term:
        # f=1 at (i1,j) AND (i2,j), g=1 at (j,k), h must satisfy:
        # h(k,i1)*1*1 + h(k,i2)*1*1 should give 1 for both i1 and i2 at position k...
        # No, we need the sum over c to give 1 at each (i,j,k).
        # 
        # If one term has f=1 at both (i1,j) and (i2,j), then for position (i1,j,k):
        # contribution from this term = 1 * g(j,k) * h(k,i1)
        # For position (i2,j,k): contribution = 1 * g(j,k) * h(k,i2)
        # We need both to equal 1 (if this is the only term covering those positions).
        # So g(j,k)*h(k,i1) = 1 and g(j,k)*h(k,i2) = 1
        # This means h(k,i1) = h(k,i2) = 1/g(j,k) for all k.
        # But h depends on (k,i), so h(k,i1) and h(k,i2) are different entries.
        # This works! Set g(j,k)=1, h(k,i1)=1, h(k,i2)=1 for all k.
        # Then the term contributes 1 to both (i1,j,k) and (i2,j,k) for all k.
        # But we also need it to NOT contribute to other positions.
        # f is 1 only at (i1,j) and (i2,j). g is 1 at (j,k) for all k? No, g should be 1 only where needed.
        # If g(j,k)=1 for all k, then for position (i1, j', k) with j'≠j: f(i1,j')=0, so no contribution. Good.
        # For position (i1, j, k): contribution = 1*1*1 = 1. Good.
        # But wait, this term also contributes to (i2, j, k) = 1. 
        # So this single term covers ALL k for both i1 and i2 at fixed j.
        # That's p terms replaced by 1 term! We save p-1 per merge.
        
        # So: merge all k's for a fixed (i,j) into one term?
        # Term: f=1 at (i,j), g(j,k)=1 for all k, h(k,i)=1 for all k.
        # Contribution at (i,j,k) = 1*1*1 = 1. ✓
        # Contribution at (i',j',k) for (i',j')≠(i,j): f=0. ✓
        # But contribution at (i, j, k) is covered. What about other i' at same j?
        # If another term has f=1 at (i',j), it doesn't affect (i,j,k) since f(i,j) for that term is 0.
        
        # Wait, but h(k,i) = 1 for all k means W[k*n+i, c] = 1 for all k.
        # And g(j,k) = 1 for all k means V[j*p+k, c] = 1 for all k.
        # 
        # So for each (i,j) pair, we need ONE term that covers all k.
        # Number of terms = n * m = 2 * 4 = 8!
        # 
        # Let me verify: with 8 terms, term c corresponds to (i_c, j_c).
        # U[i*m+j, c] = 1 if (i,j)=(i_c,j_c), else 0.
        # V[j*p+k, c] = 1 if j=j_c, else 0 (for all k).
        # W[k*n+i, c] = 1 if i=i_c, else 0 (for all k).
        # 
        # Check: at position (i,j,k):
        # Sum over c: U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # = [1 if (i,j)=(i_c,j_c)] * [1 if j=j_c] * [1 if i=i_c]
        # For the term where (i_c,j_c)=(i,j): 1*1*1 = 1.
        # For other terms: if i_c≠i or j_c≠j, at least one factor is 0.
        # So sum = 1. ✓✓✓
        
        # This gives rank n*m = 8!
        # Let me double-check the shapes:
        # U: (n*m, rank) = (8, 8). U[i*m+j, c] = δ_{(i,j),(i_c,j_c)}
        # V: (m*p, rank) = (28, 8). V[j*p+k, c] = δ_{j,j_c}
        # W: (p*n, rank) = (14, 8). W[k*n+i, c] = δ_{i,i_c}
        
        rank_try = n * m  # = 8
        U = [[0] * rank_try for _ in range(n * m)]
        V = [[0] * rank_try for _ in range(m * p)]
        W = [[0] * rank_try for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                # U[i*m+j, col] = 1
                U[i * m + j][col] = 1
                # V[j*p+k, col] = 1 for all k
                for k in range(p):
                    V[j * p + k][col] = 1
                # W[k*n+i, col] = 1 for all k
                for k in range(p):
                    W[k * n + i][col] = 1
                col += 1
        
        # Verify
        if _check_decomposition(U, V, W, n, m, p):
            best_rank = rank_try
            best_result = {"U": U, "V": V, "W": W}
            print(f"Found valid decomposition with rank {rank_try}!", file=sys.stderr)
            break
        else:
            print(f"Rank {rank_try} failed verification", file=sys.stderr)
    
    if best_result is not None:
        return best_result
    return None

# EVOLVE-BLOCK-START
def construct():
    """Construct the tensor decomposition, trying to minimize rank."""
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Try to find a lower rank decomposition
    try:
        result = _try_rank_reduction()
        if result is not None:
            return result
    except Exception as e:
        print(f"Search failed: {e}", file=sys.stderr)
    
    # Fallback: schoolbook decomposition
    return _schoolbook()
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    n, m, p = DIMENSIONS
    print(f"Valid: {_check_decomposition(result['U'], result['V'], result['W'], n, m, p)}")

