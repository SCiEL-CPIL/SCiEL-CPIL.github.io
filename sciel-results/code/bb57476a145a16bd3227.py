"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_from_terms(all_terms, n, m, p):
    """Build U, V, W matrices from a list of (u_vec, v_vec, w_vec) terms."""
    rank = len(all_terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r, (u_vec, v_vec, w_vec) in enumerate(all_terms):
        U[:, r] = u_vec
        V[:, r] = v_vec
        W[:, r] = w_vec
    return U, V, W

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    # M1 = (a11+a22)(b11+b22)
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (a21+a22)b11
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 1]))
    # M3 = a11(b12-b22)
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    # M4 = a22(b21-b11)
    terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]))
    # M5 = (a11+a12)b22
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]))
    # M6 = (a21-a11)(b11+b12)
    terms.append(([1, 0, -1, 0], [1, 1, 0, 0], [0, 1, 0, 0]))
    # M7 = (a12-a22)(b21+b22)
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]))
    return terms

def _make_schoolbook_terms(n, m, p, k_start=0, k_end=None):
    """Make schoolbook terms for columns k_start to k_end-1."""
    if k_end is None:
        k_end = p
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(k_start, k_end):
                u_vec = [0.0] * (n * m)
                u_vec[i * m + j] = 1.0
                v_vec = [0.0] * (m * p)
                v_vec[j * p + k] = 1.0
                w_vec = [0.0] * (p * n)
                w_vec[k * n + i] = 1.0
                terms.append((u_vec, v_vec, w_vec))
    return terms

def _extend_2x2_term_to_2x5(u_vec, v_vec_2x2, w_vec_2x2, n=2, m=2, p=5):
    """Extend a 2x2x2 term to a 2x2x5 term (only uses k=0,1)."""
    u_full = list(u_vec)  # already length 4
    v_full = [0.0] * (m * p)
    for j in range(m):
        for k in range(2):
            v_full[j * p + k] = v_vec_2x2[j * 2 + k]
    w_full = [0.0] * (p * n)
    for k in range(2):
        for i in range(n):
            w_full[k * n + i] = w_vec_2x2[k * 2 + i]
    return u_full, v_full, w_full

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # Build initial rank-13 decomposition: Strassen(2x2x2) + schoolbook(2x2x3)
    strassen_terms = _strassen_2x2_terms()
    all_terms = []
    
    for u_vec, v_vec, w_vec in strassen_terms:
        u_full, v_full, w_full = _extend_2x2_term_to_2x5(u_vec, v_vec, w_vec, n, m, p)
        all_terms.append((u_full, v_full, w_full))
    
    schoolbook_terms = _make_schoolbook_terms(n, m, p, k_start=2, k_end=5)
    all_terms.extend(schoolbook_terms)
    
    U, V, W = _build_from_terms(all_terms, n, m, p)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to full schoolbook
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try to reduce rank by finding a lower-rank decomposition
    # Strategy: try ranks from best_rank-1 down to 10
    # For each rank, try to solve the tensor decomposition problem
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try a structured approach for lower ranks
    # Key insight: we can try to use the fact that the 2x2x5 tensor
    # might have a decomposition that shares structure between the
    # 2x2x2 block and the 2x2x3 block
    
    # Try rank 12: use 7 Strassen-like terms + 5 additional terms
    # where the 5 terms cover the 2x2x3 block more efficiently
    
    # Try rank 11 and 10 as well
    
    for target_rank in range(best_rank - 1, 9, -1):
        if time.time() > deadline - 15:
            break
        
        # Try to find a decomposition of the given rank
        # Use a structured approach:
        # For the 2x2x5 problem, we can think of it as:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # One approach: use the fact that for fixed i, the equation
        # sum_j A[i,j]*B[j,k] = C[i,k] must hold for all k
        # This gives us p=5 equations per (i,j) pair
        
        # Try a parametric approach:
        # Assume certain structure in U, V, W and solve for the free parameters
        
        # For now, try a simpler heuristic:
        # Check if we can merge two terms by finding a linear combination
        
        # Try: for each pair of terms (r1, r2), check if we can find
        # scalars a, b such that a*term_r1 + b*term_r2 = 0 in some sense
        # This is complex, so let's try a different approach
        
        # Alternative: use the known result that 2x2x5 has rank at most 12
        # by using a specific construction
        
        # Let me try to construct a rank-12 decomposition explicitly
        if target_rank == 12:
            # Try: 7 terms for the 2x2x2 part (modified) + 5 terms for 2x2x3
            # The 5 terms for 2x2x3: use a "partial Strassen" for 2x2x3
            # 2x2x3 schoolbook is 6, can we do 5?
            # Lower bound for 2x2x3: max(4, 6, 6) = 6, so no.
            # But when combined with the 2x2x2 part, we might share.
            
            # Let me try a different construction:
            # Use 6 terms that cover both the 2x2x2 and part of 2x2x3
            # and 6 more terms for the rest
            
            # Actually, let's try to solve the system directly for rank 12
            # with a structured ansatz
            
            # Ansatz: 
            # U = [U1 | U2] where U1 is 4x7 (Strassen-like) and U2 is 4x5
            # V = [V1 | V2] where V1 is 10x7 and V2 is 10x5
            # W = [W1 | W2] where W1 is 10x7 and W2 is 10x5
            
            # The 7 Strassen terms handle k=0,1
            # The 5 additional terms need to handle k=2,3,4
            # But 5 terms for 2x2x3 is below the lower bound of 6...
            # Unless the Strassen terms also contribute to k=2,3,4
            
            # Let me modify the Strassen terms to also contribute to k=2,3,4
            # by extending V and W vectors
            
            # New approach: design 12 terms from scratch
            # Term structure:
            # - 4 terms that are "diagonal" for the 2x2 part
            # - 3 terms that are "off-diagonal" for the 2x2 part  
            # - 5 terms for the extended k=2,3,4 part
            
            # Let me try a direct construction:
            # For k=0,1,2: use a 2x2x3 decomposition (rank 6)
            # For k=3,4: use 2x2x2 decomposition (rank 4)
            # Total: 10, but with overlap we might get 12 or less
            
            # Actually, let's just try to verify if rank 10, 11, or 12 is achievable
            # by trying random initializations and gradient descent
            # But that might be too slow...
            
            # For now, let's try a specific rank-12 construction:
            # Use the 7 Strassen terms extended to all 5 k-values
            # and add 5 more terms to fix the k=2,3,4 contributions
            
            # Modified Strassen terms that contribute to all k:
            new_terms = []
            
            # Term 1: (a11+a22)(b11+b22+b23+b24) -> contributes to c11, c22 for all k
            # U: [1,0,0,1], V: [1,0,0,0,0, 0,0,0,0,0] wait, V is j*p+k
            # V for b11+b22: j=0,k=0 and j=1,k=1 -> [1,0,0,0,0, 0,0,1,0,0]
            # But we want b11+b22+b23+b24 for all k... that's not a rank-1 term
            
            # Let me think differently. A rank-1 term has the form:
            # u_i * v_jk * w_ki
            # where u_i is a linear form in A entries, v_jk in B entries, w_ki in C entries
            
            # For the 2x2x5 case:
            # u has 4 components (a11, a12, a21, a22)
            # v has 10 components (b11, b12, b13, b14, b15, b21, b22, b23, b24, b25)
            # w has 10 components (c11, c12, c13, c14, c15, c21, c22, c23, c24, c25)
            
            # Let me try to find 10, 11, or 12 rank-1 terms that sum to the tensor
            
            # This is a hard problem. Let me try a simpler heuristic:
            # Start with the rank-13 solution and try to eliminate one term
            # by expressing it as a combination of others
            
            U_cur = U.copy()
            V_cur = V.copy()
            W_cur = W.copy()
            rank_cur = U_cur.shape[1]
            
            # Try to eliminate term r by solving for coefficients
            # such that term_r = sum_{s != r} alpha_s * term_s
            # This requires: U[:,r] = sum_{s!=r} alpha_s * U[:,s]
            # AND V[:,r] = sum_{s!=r} alpha_s * V[:,s]  
            # AND W[:,r] = sum_{s!=r} alpha_s * W[:,s]
            # AND the tensor product condition
            
            # This is generally not possible, but let's check if any term is
            # linearly dependent in a useful way
            
            for r_elim in range(rank_cur):
                if time.time() > deadline - 10:
                    break
                
                # Check if U[:,r_elim] is in the span of other U columns
                U_others = np.delete(U_cur, r_elim, axis=1)
                try:
                    coeffs_U, residuals_U, rank_U, sv = np.linalg.lstsq(U_others, U_cur[:, r_elim], rcond=None)
                    if np.allclose(U_others @ coeffs_U, U_cur[:, r_elim], atol=1e-8):
                        # U is dependent, check V and W
                        V_others = np.delete(V_cur, r_elim, axis=1)
                        coeffs_V, _, _, _ = np.linalg.lstsq(V_others, V_cur[:, r_elim], rcond=None)
                        if np.allclose(V_others @ coeffs_V, V_cur[:, r_elim], atol=1e-8):
                            W_others = np.delete(W_cur, r_elim, axis=1)
                            coeffs_W, _, _, _ = np.linalg.lstsq(W_others, W_cur[:, r_elim], rcond=None)
                            if np.allclose(W_others @ coeffs_W, W_cur[:, r_elim], atol=1e-8):
                                # All three are dependent, but we need the tensor product
                                # to still work. This is a sufficient but not necessary condition.
                                # Try removing this term and verifying
                                U_test = np.delete(U_cur, r_elim, axis=1)
                                V_test = np.delete(V_cur, r_elim, axis=1)
                                W_test = np.delete(W_cur, r_elim, axis=1)
                                if _verify_decomposition(U_test, V_test, W_test, n, m, p):
                                    U_cur = U_test
                                    V_cur = V_test
                                    W_cur = W_test
                                    rank_cur -= 1
                                    break
                except:
                    pass
            
            if rank_cur < best_rank:
                best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                best_rank = rank_cur
        
        # If we haven't found a better solution, try a different approach
        # for this target rank
        if best_rank > target_rank and time.time() < deadline - 20:
            # Try to construct a solution from scratch for this rank
            # using a structured ansatz
            
            # For rank 10: try a "tensor train" like approach
            # For rank 11: try partial sharing
            # For rank 12: try the construction above
            
            pass
    
    # Final: return the best decomposition found
    # Clean up: convert near-integers to exact integers
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx, val in enumerate(row):
                if abs(val - round(val)) < 1e-10:
                    row[idx] = round(val)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
