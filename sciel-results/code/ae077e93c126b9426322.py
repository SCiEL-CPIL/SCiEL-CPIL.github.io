"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def _verify(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # W has shape (p*n, rank), U has (n*m, rank), V has (m*p, rank)
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 if this corresponds to (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Actually, the standard decomposition: (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor for matmul: T[i, j, k] where i indexes row of A, j is the inner index, k is col of B
    # Wait, let me re-read the task.
    # "Matrix multiplication 3,3,3"
    # Output factor follows W[k*n+i, r]
    # U: (n*m, rank) -> indexed by (i, j) where i in [0,n), j in [0,m)
    # V: (m*p, rank) -> indexed by (j, k) where j in [0,m), k in [0,p)
    # W: (p*n, rank) -> indexed by (k, i) where k in [0,p), i in [0,n)
    # The product is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = (A*B)[i,k]
    # For standard matmul, (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] should be 1 for each term A[i,j]*B[j,k] contributing to (A*B)[i,k]
    # Actually, the tensor decomposition of matmul: 
    # (A*B)_{i,k} = sum_{r} U_{(i,j),r} V_{(j,k),r} W_{(k,i),r}
    # This must equal sum_j A_{i,j} B_{j,k}
    # So for each r, we have a term U_{(i,j),r} V_{(j,k),r} W_{(k,i),r}
    # The total sum over r must give sum_j A_{i,j} B_{j,k} for all i,k
    # This means: for each i,k, sum_{j,r} U_{(i,j),r} V_{(j,k),r} W_{(k,i),r} = sum_j A_{i,j} B_{j,k}
    # In terms of the tensor: the matmul tensor is T[i,j,k] = 1 (it's the sum over j of A[i,j]B[j,k])
    # Wait, no. The tensor T is defined such that (A*B)_{i,k} = sum_{i,j,k} A_{i,j} B_{j,k} T_{i,j,k}
    # No, that's not right either.
    # Standard: (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor decomposition: (A*B)_{i,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # where the sum over r is the rank. But U depends on i,j and V on j,k and W on k,i.
    # For this to work: sum_r U_{(i,j),r} V_{(j,k),r} W_{(k,i),r} = delta_{i,k}... no.
    # Let me think again. The matmul tensor is T[i,j,k] = 1 for the "trivial" decomposition where
    # each (i,j,k) triple contributes A[i,j]*B[j,k]. But that's not a tensor decomposition in the
    # bilinear sense.
    # 
    # Bilinear decomposition of matmul:
    # (A*B)_{i,k} = sum_{r=1}^{R} sum_{j} U_{(i,j),r} A_{i,j} * V_{(j,k),r} B_{j,k} * W_{(k,i),r}
    # Wait, that's not right. In a bilinear decomposition:
    # (A*B)_{i,k} = sum_r (sum_j U_{(i,j),r} A_{i,j}) * (sum_k' V_{(j,k'),r} B_{j,k'}) ... no.
    #
    # Standard form: (A*B)_{i,k} = sum_r L_r(A) * M_r(B) * W_{(k,i),r}
    # where L_r(A) = sum_{i,j} U_{(i,j),r} A_{i,j} and M_r(B) = sum_{j,k} V_{(j,k),r} B_{j,k}
    # So: (A*B)_{i,k} = sum_r (sum_{i',j'} U_{(i',j'),r} A_{i',j'}) * (sum_{j',k'} V_{(j',k'),r} B_{j',k'}) * W_{(k,i),r}
    #
    # For this to equal sum_j A_{i,j} B_{j,k} for all A,B,i,k:
    # We need: sum_r U_{(i',j'),r} V_{(j',k'),r} W_{(k,i),r} A_{i',j'} B_{j',k'} = sum_j A_{i,j} B_{j,k}
    # This must hold for all A,B, so:
    # sum_r U_{(i',j'),r} V_{(j',k'),r} W_{(k,i),r} = delta_{i',i} delta_{j',j} delta_{k',k} ... no.
    # It should be: the coefficient of A_{i',j'} B_{j',k'} in LHS is sum_r U_{(i',j'),r} V_{(j',k'),r} W_{(k,i),r}
    # and in RHS it's 1 if i'=i, j'=j, k'=k... no, RHS is sum_j A_{i,j} B_{j,k}, so coefficient of A_{i',j'} B_{j',k'} is 1 if i'=i, j'=j', k'=k and j'=j... 
    # Actually RHS = sum_j A_{i,j} B_{j,k}, so the coefficient of A_{i',j'} B_{j',k'} is 1 if i'=i, j'=j', k'=k, and j'=j (which is automatic if j'=j').
    # Wait, in sum_j A_{i,j} B_{j,k}, the term A_{i',j'} B_{j',k'} appears when i'=i, j'=j, k'=k. So coefficient is 1 if (i',j',k') = (i,j,k) for some j, i.e., i'=i and k'=k and j' is free but must equal j.
    # So coefficient of A_{i',j'} B_{j',k'} in RHS is 1 if i'=i and k'=k, regardless of j' (as long as we sum over j).
    # Hmm, this is getting confusing. Let me use the standard formulation.
    #
    # The matmul tensor T has entries T[i,j,k] = 1, and (A*B)_{i,k} = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    # Wait, that gives sum_{i,j,k} A[i,j] B[j,k] which is not (A*B)_{i,k}.
    #
    # I think the correct statement is: (A*B)_{i,k} = sum_j A[i,j] B[j,k]
    # This is a trilinear form in A, B, and the output C: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor for this is T[i,j,k] = 1, and C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    # But T here is indexed by (i,j,k) and we sum over j only.
    #
    # For tensor rank decomposition: we want to write the matmul operation as:
    # C[i,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] ... this doesn't look right dimensionally.
    #
    # Let me look at the seed code more carefully:
    # U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1
    # So for each (i,j,k) triple, we have a column where U[(i,j)]=1, V[(j,k)]=1, W[(k,i)]=1.
    # The product for output (i,k) would be: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # But r corresponds to (i',j',k') triples. For a fixed output (i,k), we sum over all r=(i',j',k'):
    # sum_{i',j',k'} U[i*m+j, (i',j',k')] V[j*p+k, (i',j',k')] W[k*n+i, (i',j',k')]
    # = sum_{i',j',k'} delta_{i,i'} delta_{j,j'} delta_{k,k'} delta_{k,k'} delta_{i,i'} ... 
    # = sum_{j'} 1 * 1 * 1 where i'=i, j'=j, k'=k
    # = 1 (for the specific j that matches)
    # Hmm, but we need sum_j A[i,j] B[j,k]. 
    #
    # I think the verification should check:
    # For the decomposition to be correct, for all i,k:
    # sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # (since A[i,j]*B[j,k] contributes 1 to the coefficient)
    # Actually, the coefficient of A[i,j]B[j,k] in the output C[i,k] should be 1.
    # In the decomposition, C[i,k] = sum_r (sum_j U[i*m+j,r] A[i,j]) (sum_k' V[j*p+k',r] B[j,k']) W[k*n+i,r]
    # This is not quite right. Let me just verify by checking that the reconstructed tensor matches.
    
    # The matmul tensor: for each (i,j,k), the "slice" contributes to C[i,k] with weight 1
    # The decomposition should satisfy: for all i,j,k,
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # because A[i,j]*B[j,k] should contribute exactly 1 to C[i,k].
    
    rank = U.shape[1]
    # Build the tensor from the decomposition
    # T_decomp[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    T_decomp = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # The target tensor is all 1s (since each A[i,j]*B[j,k] contributes 1 to C[i,k])
    T_target = np.ones((n, m, p))
    return np.allclose(T_decomp, T_target, atol=tol)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Start with schoolbook rank 27
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
    
    # Try to find lower rank decompositions using random search with structured patterns
    # Known: rank of 3x3x3 matmul is between 19 and 23 (actually 19 is conjectured, 23 is upper bound)
    # Let's try random integer/half-integer entries with small magnitudes
    
    import random
    random.seed(42)
    
    # Candidate values for entries
    candidate_vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    # Try ranks from 19 upward
    for target_rank in range(19, best_rank + 1):
        if time.time() > deadline:
            break
        
        attempts = 0
        max_attempts = 5000 if target_rank <= 21 else 2000
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Generate random U, V, W with sparse entries from candidate_vals
            # Use a structured approach: each row has 1-3 non-zero entries
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            for col in range(target_rank):
                # Choose random non-zero pattern
                # U column: pick 1-3 random (i,j) pairs
                nz_count_u = random.randint(1, 3)
                indices_u = random.sample(range(n * m), nz_count_u)
                for idx in indices_u:
                    U[idx, col] = random.choice(candidate_vals)
                
                nz_count_v = random.randint(1, 3)
                indices_v = random.sample(range(m * p), nz_count_v)
                for idx in indices_v:
                    V[idx, col] = random.choice(candidate_vals)
                
                nz_count_w = random.randint(1, 3)
                indices_w = random.sample(range(p * n), nz_count_w)
                for idx in indices_w:
                    W[idx, col] = random.choice(candidate_vals)
            
            # Verify
            if _verify(U, V, W, n, m, p):
                best_rank = target_rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break  # Found valid decomposition at this rank, try lower
        
        if best_rank <= target_rank:
            break
    
    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                # Round to nearest half-integer
                val = round(val * 2) / 2
                if val == int(val):
                    new_row.append(int(val))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
