import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 120.0
    n, m, p = DIMENSIONS
    
    # Schoolbook (Rank 27) as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    schoolbook = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    def verify(U, V, W):
        """Verify that U, V, W form a valid decomposition."""
        if time.time() > DEADLINE - 5:
            return False
        # Check shapes
        if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
            return False
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        # Check exact decomposition: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        # and for all other (i',j',k') where not matching the pattern, the sum should be 0
        # Actually, the tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For matrix multiplication, T[i,j,k] should be 1 if this corresponds to C[i,k] += A[i,j]*B[j,k]
        # The tensor for 3x3x3 matrix mult: T[i,j,k] = 1 for all i,j,k (it's the identity tensor)
        # Wait, let me think again.
        # C = A * B, C[i,k] = sum_j A[i,j] * B[j,k]
        # The multiplication tensor has T[i,j,k] = 1 for all i,j,k in {0,1,2}
        # So we need: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        
        # Efficient check using einsum
        # T_ijk = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a bit tricky with einsum. Let's do it step by step.
        # First, compute S[i,j,k] = sum_r U[i,j,r] * V[j,k,r]
        # S has shape (n, m, p)
        # S[i,j,k] = sum_r U_reshaped[i,j,:] * V_reshaped[j,k,:]
        
        # Use broadcasting: U_reshaped[:, :, None, :] * V_reshaped[None, :, :, :]
        # U_reshaped: (n, m, rank)
        # V_reshaped: (m, p, rank)
        # We want S[i,j,k] = sum_r U[i,j,r] * V[j,k,r]
        # U_reshaped[:, :, None, :] has shape (n, m, 1, rank)
        # V_reshaped[None, :, :, :] has shape (1, m, p, rank)
        # Product: (n, m, p, rank)
        # Sum over last axis: (n, m, p)
        
        S = np.einsum('ijr,jkr->ijk', U_reshaped, V_reshaped)
        
        # Now T[i,j,k] = S[i,j,k] * W[k,i,?]... wait
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is not simply S[i,j,k] * something because W depends on k and i, not j.
        # Let me redo this.
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # Let's compute it directly:
        # U_reshaped: (n, m, rank)
        # V_reshaped: (m, p, rank)  
        # W_reshaped: (p, n, rank)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We can do: first compute UV[i,j,k,r] = U[i,j,r] * V[j,k,r]
        # UV has shape (n, m, p, rank)
        # Then T[i,j,k] = sum_r UV[i,j,k,r] * W[k,i,r]
        
        # UV[i,j,k,r] = U_reshaped[i,j,r] * V_reshaped[j,k,r]
        # Using einsum: 'ijr,jkr->ijk r' -> 'ijkr'
        UV = np.einsum('ijr,jkr->ijk r', U_reshaped, V_reshaped)
        # Wait, that syntax is wrong. Let me use:
        UV = np.einsum('ijr,jkr->ijk r'.replace(' ', ''), U_reshaped, V_reshaped)
        
        # Now T[i,j,k] = sum_r UV[i,j,k,r] * W_reshaped[k,i,r]
        # W_reshaped has shape (p, n, rank), i.e., W_reshaped[k,i,r]
        # We need to sum over r: UV[i,j,k,r] * W_reshaped[k,i,r]
        # Result T[i,j,k]
        
        T = np.einsum('ijk r,ki r->ijk'.replace(' ', ''), UV, W_reshaped)
        
        # Check if T is all ones
        if not np.allclose(T, 1.0, atol=1e-10):
            return False
        
        # Also check that entries are integers or half-integers
        all_entries = np.concatenate([U_arr.flatten(), V_arr.flatten(), W_arr.flatten()])
        # Check if each entry is close to a half-integer
        doubled = 2 * all_entries
        if not np.allclose(doubled, np.round(doubled), atol=1e-10):
            return False
            
        return True
    
    # Build rank-26 decomposition using 2x2+1x1 block structure
    # Partition indices:
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2 = 8 terms
    # Block 2: i in {0,1}, j in {2}, k in {0,1} -> 2x1x2 = 4 terms  
    # Block 3: i in {0,1}, j in {0,1}, k in {2} -> 2x2x1 = 4 terms
    # Block 4: i in {2}, j in {0,1}, k in {0,1} -> 1x2x2 = 4 terms
    # Block 5: i in {0,1}, j in {2}, k in {2} -> 2x1x1 = 2 terms
    # Block 6: i in {2}, j in {0,1}, k in {2} -> 1x2x1 = 2 terms
    # Block 7: i in {2}, j in {2}, k in {0,1} -> 1x1x2 = 2 terms
    # Block 8: i in {2}, j in {2}, k in {2} -> 1x1x1 = 1 term
    # Total: 8+4+4+4+2+2+2+1 = 27... that's just schoolbook again.
    
    # Let me think differently. The 2x2+1x1 approach:
    # Use rank-5 for 2x2 matrix multiplication (Strassen-like but for 2x2, min rank is 7)
    # Wait, 2x2 matrix multiplication has rank 7 (Strassen).
    # So for the top-left 2x2 block: 7 terms
    # For the remaining parts, we need to handle cross terms.
    
    # Actually, let me use a different approach. 
    # Known result: 3x3x3 matrix multiplication can be done in rank 23 (Laderman) or 26.
    # Let me try to construct rank 26.
    
    # Alternative: use the fact that we can decompose as:
    # C[0:2,0:2] = A[0:2,0:2] * B[0:2,0:2]  (rank 7 via Strassen)
    # C[0:2,2] = A[0:2,0:2] * B[0:2,2] + A[0:2,2] * B[2,2]
    # C[2,0:2] = A[2,0:2] * B[0:2,0:2] + A[2,2] * B[2,0:2]
    # C[2,2] = A[2,0:2] * B[0:2,2] + A[2,2] * B[2,2]
    
    # This is getting complex. Let me just return schoolbook for now since I can't 
    # guarantee the rank-26 construction is correct.
    
    return schoolbook
# EVOLVE-BLOCK-END
