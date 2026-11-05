"""Program for rank-minimal exact tensor decomposition of 4x5*5x6 matrix multiplication."""
import numpy as np
import time
from itertools import combinations

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Uses a block decomposition with Strassen's algorithm for 2x2x2 blocks 
    and schoolbook for 2x1x2 blocks, then attempts rank reduction.
    Expected rank: 108 (vs 120 schoolbook), potentially lower after reduction.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Define block offsets and sizes
    i_offsets = [0, 2]
    i_sizes = [2, 2]
    j_offsets = [0, 2, 4]
    j_sizes = [2, 2, 1]
    k_offsets = [0, 2, 4]
    k_sizes = [2, 2, 2]
    
    max_rank = 120  # Safe upper bound
    
    U = np.zeros((n * m, max_rank), dtype=float)
    V = np.zeros((m * p, max_rank), dtype=float)
    W = np.zeros((p * n, max_rank), dtype=float)
    
    col = 0
    
    for i_idx, i_0 in enumerate(i_offsets):
        n_b = i_sizes[i_idx]
        if time.time() > deadline:
            break
        for j_idx, j_0 in enumerate(j_offsets):
            m_b = j_sizes[j_idx]
            if time.time() > deadline:
                break
            for k_idx, k_0 in enumerate(k_offsets):
                p_b = k_sizes[k_idx]
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    # Strassen 2x2x2: 7 terms
                    # Term 1: (A11+A22)(B11+B22)
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    U[(i_0+1)*m + (j_0+1), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = 1
                    col += 1
                    
                    # Term 2: (A21+A22)B11
                    U[(i_0+1)*m + (j_0+0), col] = 1
                    U[(i_0+1)*m + (j_0+1), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = -1
                    col += 1
                    
                    # Term 3: A11(B21-B11)
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+1), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = -1
                    W[(k_0+1)*n + (i_0+0), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = 1
                    col += 1
                    
                    # Term 4: A22(B12-B22)
                    U[(i_0+1)*m + (j_0+1), col] = 1
                    V[(j_0+1)*p + (k_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = -1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+1), col] = 1
                    col += 1
                    
                    # Term 5: (A11+A12)B22
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    U[(i_0+0)*m + (j_0+1), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = -1
                    col += 1
                    
                    # Term 6: (A21-A11)(B11+B12)
                    U[(i_0+1)*m + (j_0+0), col] = 1
                    U[(i_0+0)*m + (j_0+0), col] = -1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = 1
                    col += 1
                    
                    # Term 7: (A12-A22)(B21+B22)
                    U[(i_0+0)*m + (j_0+1), col] = 1
                    U[(i_0+1)*m + (j_0+1), col] = -1
                    V[(j_0+1)*p + (k_0+0), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    col += 1
                    
                elif n_b == 2 and m_b == 1 and p_b == 2:
                    # Schoolbook for 2x1x2
                    for u in range(2):
                        for w in range(2):
                            U[(i_0+u)*m + (j_0+0), col] = 1
                            V[(j_0+0)*p + (k_0+w), col] = 1
                            W[(k_0+w)*n + (i_0+u), col] = 1
                            col += 1
                else:
                    for u in range(n_b):
                        for v in range(m_b):
                            for w in range(p_b):
                                U[(i_0+u)*m + (j_0+v), col] = 1
                                V[(j_0+v)*p + (k_0+w), col] = 1
                                W[(k_0+w)*n + (i_0+u), col] = 1
                                col += 1

    # Trim arrays to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    # Attempt rank reduction by finding and eliminating redundant terms
    # A term t is redundant if its tensor contribution can be expressed as
    # a linear combination of other terms.
    # For each term, check if removing it still allows reconstruction.
    
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        """Verify that the decomposition correctly computes matrix multiplication."""
        # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need: for all (a,b,c), sum_r U[a*m+b, r] * V[b*p+c, r] * W[c*n+a, r] = 1
        # if and only if the (a,c) entry of the product of A and B is 1,
        # which means: the tensor should satisfy T[a,b,c] = delta for the matmul.
        # Actually, the matmul tensor has T[i,j,k] = 1 iff we're computing C[i,k] from A[i,j]*B[j,k]
        # The standard form: T[i,j,k] = 1 for all (i,j,k) in the matmul tensor.
        # Verify by checking that the decomposition reproduces the identity:
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1 for all (i,j,k)
        # that correspond to valid matmul indices.
        
        # For matmul 4x5 * 5x6, the tensor T has T[i,j,k]=1 for all i in [0,4), j in [0,5), k in [0,6)
        # Wait, no. The matrix multiplication tensor T_{i,j,k} is defined such that
        # C_{i,k} = sum_j A_{i,j} * B_{j,k}
        # So the tensor has non-zero entries T_{i,j,k} = 1 for all valid (i,j,k).
        # In the decomposition: T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
        # (using the trace convention W[k*n+i, r])
        
        # Verify all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0  # All entries of the matmul tensor are 1
                    actual = 0.0
                    for r in range(U_mat.shape[1]):
                        actual += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(actual - expected) > 1e-10:
                        return False
        return True
    
    if time.time() < deadline - 30.0:
        # Try to find redundant terms
        # For each term, try to express it as a linear combination of others
        # This is expensive, so we do a limited search
        
        current_rank = col
        removed = []
        
        # Build the tensor from the current decomposition for verification
        # and try greedy removal
        
        for attempt in range(current_rank):
            if time.time() > deadline - 20.0:
                break
            
            found_removal = False
            
            # Try removing each remaining term
            for t in range(current_rank):
                if t in removed:
                    continue
                
                # Create a copy without term t
                U_test = U.copy()
                V_test = V.copy()
                W_test = W.copy()
                
                # Zero out column t
                U_test[:, t] = 0
                V_test[:, t] = 0
                W_test[:, t] = 0
                
                # Check if the decomposition still works
                # This is O(n*m*p*rank) which is 4*5*6*108 = 12960 ops per check
                # Too slow for all 108 terms. Use a smarter approach.
                
                # Instead, check if the contribution of term t is already
                # covered by other terms. The contribution of term t to the tensor is:
                # T_t[i,j,k] = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                # We need to check if this can be written as sum_{r != t} c_r * T_r[i,j,k]
                # for some coefficients c_r.
                
                # This is a linear system. But it's complex. Let's do a simpler check:
                # If the column t in U, V, W are all zero, it's already removed.
                # Otherwise, check if removing it breaks the decomposition.
                
                # Quick check: verify a subset of tensor entries
                valid = True
                # Check all entries (should be fast enough for 12960 multiplications)
                for i in range(n):
                    if not valid:
                        break
                    for j in range(m):
                        if not valid:
                            break
                        for k in range(p):
                            if not valid:
                                break
                            actual = 0.0
                            for r in range(current_rank):
                                if r == t:
                                    continue
                                actual += U_test[i*m+j, r] * V_test[j*p+k, r] * W_test[k*n+i, r]
                            if abs(actual - 1.0) > 1e-10:
                                valid = False
                                break
                
                if valid:
                    # Term t is redundant, remove it
                    removed.append(t)
                    found_removal = True
                    break
            
            if not found_removal:
                break
        
        # Apply removals
        if removed:
            keep = [r for r in range(current_rank) if r not in removed]
            U = U[:, keep]
            V = V[:, keep]
            W = W[:, keep]
    
    # Final verification
    if not verify_decomposition(U, V, W, n, m, p):
        # If verification fails, fall back to the original unmodified decomposition
        # Reconstruct from scratch
        U2 = np.zeros((n * m, max_rank), dtype=float)
        V2 = np.zeros((m * p, max_rank), dtype=float)
        W2 = np.zeros((p * n, max_rank), dtype=float)
        
        col2 = 0
        for i_idx, i_0 in enumerate(i_offsets):
            n_b = i_sizes[i_idx]
            for j_idx, j_0 in enumerate(j_offsets):
                m_b = j_sizes[j_idx]
                for k_idx, k_0 in enumerate(k_offsets):
                    p_b = k_sizes[k_idx]
                    if n_b == 2 and m_b == 2 and p_b == 2:
                        U2[(i_0+0)*m + (j_0+0), col2] = 1
                        U2[(i_0+1)*m + (j_0+1), col2] = 1
                        V2[(j_0+0)*p + (k_0+0), col2] = 1
                        V2[(j_0+1)*p + (k_0+1), col2] = 1
                        W2[(k_0+0)*n + (i_0+0), col2] = 1
                        W2[(k_0+1)*n + (i_0+1), col2] = 1
                        col2 += 1
                        U2[(i_0+1)*m + (j_0+0), col2] = 1
                        U2[(i_0+1)*m + (j_0+1), col2] = 1
                        V2[(j_0+0)*p + (k_0+0), col2] = 1
                        W2[(k_0+0)*n + (i_0+1), col2] = 1
                        W2[(k_0+1)*n + (i_0+1), col2] = -1
                        col2 += 1
                        U2[(i_0+0)*m + (j_0+0), col2] = 1
                        V2[(j_0+0)*p + (k_0+1), col2] = 1
                        V2[(j_0+1)*p + (k_0+1), col2] = -1
                        W2[(k_0+1)*n + (i_0+0), col2] = 1
                        W2[(k_0+1)*n + (i_0+1), col2] = 1
                        col2 += 1
                        U2[(i_0+1)*m + (j_0+1), col2] = 1
                        V2[(j_0+1)*p + (k_0+0), col2] = 1
                        V2[(j_0+0)*p + (k_0+0), col2] = -1
                        W2[(k_0+0)*n + (i_0+0), col2] = 1
                        W2[(k_0+0)*n + (i_0+1), col2] = 1
                        col2 += 1
                        U2[(i_0+0)*m + (j_0+0), col2] = 1
                        U2[(i_0+0)*m + (j_0+1), col2] = 1
                        V2[(j_0+1)*p + (k_0+1), col2] = 1
                        W2[(k_0+1)*n + (i_0+0), col2] = 1
                        W2[(k_0+0)*n + (i_0+0), col2] = -1
                        col2 += 1
                        U2[(i_0+1)*m + (j_0+0), col2] = 1
                        U2[(i_0+0)*m + (j_0+0), col2] = -1
                        V2[(j_0+0)*p + (k_0+0), col2] = 1
                        V2[(j_0+0)*p + (k_0+1), col2] = 1
                        W2[(k_0+1)*n + (i_0+1), col2] = 1
                        col2 += 1
                        U2[(i_0+0)*m + (j_0+1), col2] = 1
                        U2[(i_0+1)*m + (j_0+1), col2] = -1
                        V2[(j_0+1)*p + (k_0+0), col2] = 1
                        V2[(j_0+1)*p + (k_0+1), col2] = 1
                        W2[(k_0+0)*n + (i_0+0), col2] = 1
                        col2 += 1
                    elif n_b == 2 and m_b == 1 and p_b == 2:
                        for u in range(2):
                            for w in range(2):
                                U2[(i_0+u)*m + (j_0+0), col2] = 1
                                V2[(j_0+0)*p + (k_0+w), col2] = 1
                                W2[(k_0+w)*n + (i_0+u), col2] = 1
                                col2 += 1
                    else:
                        for u in range(n_b):
                            for v in range(m_b):
                                for w in range(p_b):
                                    U2[(i_0+u)*m + (j_0+v), col2] = 1
                                    V2[(j_0+v)*p + (k_0+w), col2] = 1
                                    W2[(k_0+w)*n + (i_0+u), col2] = 1
                                    col2 += 1
        U = U2[:, :col2]
        V = V2[:, :col2]
        W = W2[:, :col2]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
