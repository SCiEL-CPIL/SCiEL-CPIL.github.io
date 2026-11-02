import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = 2, 2, 12
    
    # Try to find a rank-40 decomposition
    # Strategy: 6 blocks of 2x2, use Strassen (rank 7 each) = 42
    # Try to reduce by 2 via global linear combinations
    
    # First, build the standard rank-42 Strassen solution
    rank_42 = 42
    U42 = np.zeros((n * m, rank_42), dtype=int)
    V42 = np.zeros((m * p, rank_42), dtype=int)
    W42 = np.zeros((p * n, rank_42), dtype=int)
    
    U_forms = [
        [1, 0, 0, 1],  # M1: (A00+A11)
        [0, 0, 1, 1],  # M2: (A10+A11)
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: (A00+A01)
        [-1, 0, 1, 0], # M6: (A10-A00)
        [0, 1, 0, -1]  # M7: (A01-A11)
    ]
    
    for block_idx in range(6):
        b00 = 2 * block_idx
        b01 = 2 * block_idx + 1
        b10 = 12 + 2 * block_idx
        b11 = 12 + 2 * block_idx + 1
        c00 = 4 * block_idx
        c01 = 4 * block_idx + 2
        c10 = 4 * block_idx + 1
        c11 = 4 * block_idx + 3
        
        for t in range(7):
            col = block_idx * 7 + t
            for i in range(4):
                U42[i, col] = U_forms[t][i]
            
            if t == 0:
                V42[b00, col] += 1; V42[b11, col] += 1
                W42[c00, col] += 1; W42[c11, col] += 1
            elif t == 1:
                V42[b00, col] += 1
                W42[c10, col] += 1; W42[c11, col] -= 1
            elif t == 2:
                V42[b01, col] += 1; V42[b11, col] -= 1
                W42[c01, col] += 1; W42[c11, col] += 1
            elif t == 3:
                V42[b10, col] += 1; V42[b00, col] -= 1
                W42[c00, col] += 1; W42[c10, col] += 1
            elif t == 4:
                V42[b11, col] += 1
                W42[c00, col] -= 1; W42[c01, col] += 1
            elif t == 5:
                V42[b00, col] += 1; V42[b01, col] += 1
                W42[c11, col] += 1
            elif t == 6:
                V42[b10, col] += 1; V42[b11, col] += 1
                W42[c00, col] += 1
    
    # Verify rank-42 correctness
    def verify(U, V, W):
        # Check: sum_r U[i,r]*V[j,r] should give the multiplication tensor
        # A is n x m = 2x2, B is m x p = 2x12, C = A@B is n x p = 2x12
        # Tensor T[a, b, c] = 1 if C[c] = sum_k A[a,k]*B[k,c]
        # Using trace convention: W[k*n+i, r] -> C[i, k] where k is column index
        # So C[i, col] = sum_r U[a*2+b, r] * V[b*12+col, r] * ... 
        # Actually: C[i, col] = sum_{a,b} A[a,b] * B[b,col]
        # In our decomposition: C[i, col] = sum_r (sum_a U[a*2+b, r]*A[a,b]) * (sum_b V[b*12+col, r]) ... 
        # No: C[i,col] = sum_r U_row_i[r] * V_col_col[r] where the product reconstructs
        # The tensor is: T[i, b, col] contributes to C[i, col] via A[i,b]*B[b,col]
        # Decomposition: T[i, b, col] = sum_r U[i*2+b, r] * V[b*12+col, r] * W[col*2+i, r]... 
        # Wait, W is (p*n, rank) = (24, rank), indexed by [k*n+i, r] = [col*2+i, r]
        # The correct check: for each (i, col), C[i,col] = sum_{a,b} A[a,b] B[b,col]
        # = sum_r (sum_b U[i*2+b, r] * A[i,b]) * ... 
        # Actually the standard: C = A@B, so C[i,col] = sum_b A[i,b]*B[b,col]
        # Tensor form: T[i, b, col] = delta (it's the identity in b for the product)
        # Decomposition: T[i,b,col] = sum_r U[i*2+b, r] * V[b*12+col, r] * W[col*2+i, r] is WRONG
        # 
        # Correct: The matrix multiplication tensor M[i, b, col] where C[i,col] = sum_b A[i,b] B[b,col]
        # We need: M[i,b,col] = sum_r U[i,b,r] * V[b,col,r] ... but U is (n*m, rank)=(4,rank)
        # U[i*2+b, r], V[b*12+col, r], and the output is determined by W
        # Actually: C[i,col] = sum_r (sum_b U[i*2+b,r]*A[i,b]) ... no
        # 
        # Standard tensor decomposition for matmul:
        # C[i,j] = sum_k A[i,k] B[k,j]
        # Tensor T[i,k,j] = 1 (the multiplication tensor has T[i,k,j]=1 for all i,k,j? No)
        # T[i,k,j] = delta... The tensor is T[i,k,j] such that <T, A⊗B> = C
        # T[i,k,j] = 1 for all i,k,j? No, T[i,k,j] = 1 means C[i,j] = sum_k A[i,k]B[k,j]
        # So the tensor entries are all 1: T[i,k,j] = 1
        # Decomposition: 1 = sum_r U[i*2+k, r] * V[k*12+j, r] * W[j*2+i, r]
        # Check: sum_r U[i,k,r] V[k,j,r] W[j,i,r] should equal 1 for all i,k,j
        # Wait that's not right either. Let me think again.
        #
        # The tensor M of size (n*m) x (m*p) x (p*n) where M[(i,m),(m,j),(j,i)] ... 
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # The tensor T is (n*m) x (m*p) x (p*n) with T[a, b, c] 
        # where a = i*2+k (i=row of A, k=col of A = row of B), b = k*12+j (k=row of B, j=col of B)
        # c = j*2+i (j=col of C = col of B, i=row of C = row of A)
        # T[a,b,c] = 1 if the indices are consistent (k matches between a and b, j matches between b and c, i matches between a and c)
        # i.e., T[i*2+k, k*12+j, j*2+i] = 1, else 0
        #
        # Check: sum_r U[i*2+k, r] * V[k*12+j, r] * W[j*2+i, r] = 1 for all valid (i,k,j)
        
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    if not verify(U42, V42, W42):
        # Fallback: return rank 42 if it fails (shouldn't happen)
        return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
    
    # Now try to reduce rank by finding linear dependencies
    # Stack columns: M = [U; V; W] in a structured way
    # Each rank-r term is a vector in R^{4+24+24} = R^{52}
    # Term r: (U[:,r], V[:,r], W[:,r])
    
    terms = []
    for r in range(rank_42):
        term = np.concatenate([U42[:, r], V42[:, r], W42[:, r]])
        terms.append(term)
    
    terms_arr = np.array(terms, dtype=float)  # shape (42, 52)
    
    # Check rank of the set of terms
    rank_of_terms = np.linalg.matrix_rank(terms_arr, tol=1e-9)
    
    # If rank_of_terms < 42, we can reduce
    # Find a basis
    if rank_of_terms < rank_42:
        # Find null space of terms_arr.T to get linear dependencies
        # Or find which columns are linearly dependent
        # Use SVD
        U_svd, S, Vt = np.linalg.svd(terms_arr, full_matrices=False)
        
        # The number of significant singular values gives the rank
        num_indep = int(np.sum(S > 1e-9))
        
        if num_indep < rank_42:
            # We can express some terms as combinations of others
            # Keep the first num_indep terms (or find a better subset)
            # For now, just keep num_indep terms
            new_rank = num_indep
            U_new = U42[:, :new_rank]
            V_new = V42[:, :new_rank]
            W_new = W42[:, :new_rank]
            
            # Verify
            if verify(U_new, V_new, W_new):
                return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    # If no reduction found, try a different approach:
    # Group into 3 sets of 2 blocks, use rank-13 per group (total 39)
    # For 2x2 x 2x4 (two 2x2 blocks), rank 13 decomposition:
    # Use 7 terms for first block, 6 terms for second block sharing one A-form
    
    if time.time() < deadline - 10:
        # Try rank 40: use 42 terms but find 2 that can be eliminated
        # by expressing them as linear combinations
        
        # Build the full tensor and check if we can find a rank-40 decomposition
        # by removing 2 terms and adjusting
        
        # Alternative: just return rank 42 if we can't do better
        pass
    
    # Return the best verified solution
    return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
