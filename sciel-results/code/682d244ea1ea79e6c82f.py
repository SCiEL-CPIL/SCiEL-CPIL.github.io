"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]
_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' (list-of-lists) and rank = number of columns.
    """
    n, m, p = DIMENSIONS
    # Target tensor: T[i,j,k] = 1 if output (i,k) receives A[i,j]*B[j,k]
    # Convention: W[k*n+i, r], U[i*m+j, r], V[j*p+k, r]
    
    # --- Strategy: Build a schoolbook decomposition first, then try to reduce rank
    #     by exploiting structure. For 2x5x6, known bounds suggest rank <= 60 (schoolbook).
    #     We'll attempt a structured decomposition using block decomposition.
    
    # Schoolbook gives rank 60. Let's try to do better.
    # For small cases, we can use known optimal or near-optimal decompositions.
    
    # Let's build the full tensor and try to find a low-rank decomposition.
    # T is of shape (n*p, m) in the sense that T[i,k,j] = A[i,j]*B[j,k] -> C[i,k]
    # Actually, the matrix multiplication tensor T has indices (i, j, k) with:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # As a 3-tensor: T[i,j,k] = 1 (it's the structure tensor)
    
    # The rank of this tensor is what we want to minimize.
    # For 2x5x6, let's try to use the fact that we can decompose 2x5x6 as a combination.
    
    # Approach: Use a greedy randomized approach to find a low-rank decomposition.
    # Start with a set of rank-1 tensors and try to cover the multiplication tensor.
    
    # The multiplication tensor for (n,m,p) has n*m*p non-zero entries (all 1s in the 
    # appropriate positions when viewed as a 3-way array with specific indexing).
    
    # Actually, let's think differently. The tensor T for matrix multiplication (n x m) * (m x p) -> (n x p)
    # has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, that's not right either. Let me reconsider.
    
    # The matrix multiplication map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # As a trilinear map, this corresponds to the tensor:
    # T[i, j, k] such that the coefficient of A[i,j] * B[j,k] in C[i,k] is 1.
    # So if we view the tensor as having indices (output_index, input_A_index, input_B_index),
    # then T[(i,k), (i,j), (j,k)] = 1.
    
    # In the convention used:
    # U has shape (n*m, rank): U[i*m + j, r]
    # V has shape (m*p, rank): V[j*p + k, r]  
    # W has shape (p*n, rank): W[k*n + i, r]
    # The decomposition satisfies: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other combinations, the sum = 0.
    
    # This is the standard matrix multiplication tensor.
    
    # For n=2, m=5, p=6, the schoolbook rank is 60.
    # Let's try to find a lower rank decomposition.
    
    # Known: For 2x2x2, rank is 7 (Strassen).
    # For larger matrices, we can use block decompositions.
    
    # Let's try a block approach: decompose 2x5x6 using smaller blocks.
    # 5 = 2+3, 6 = 2+4 or 3+3, etc.
    
    # Actually, let's just do a direct randomized search for a low-rank decomposition.
    
    # Build the full multiplication tensor as a vector and try to express it as a sum of rank-1 tensors.
    
    def _build_multiplication_tensor(n, m, p):
        """Build the multiplication tensor as a dictionary of non-zero entries."""
        # T[(i,j,k)] = 1 for all valid i,j,k
        # But in the decomposition convention, we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
        # = 0 for all other combinations
        
        # The "target" is that for each (i,j,k) in the valid range, the product should be 1.
        # For invalid combinations (where the indices don't form a valid multiplication step), it should be 0.
        
        # Actually, in matrix multiplication, ALL combinations (i,j,k) with i<n, j<m, k<p are valid.
        # The constraint is that U, V, W must be such that the rank-1 decomposition reproduces
        # the correct bilinear form.
        
        # The condition is:
        # For all i in [0,n), j in [0,m), k in [0,p):
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # And implicitly, the decomposition must not create spurious terms.
        
        # Wait, that's not quite right. The full condition for exact decomposition is:
        # The tensor T with T[i, j, k] = 1 (for all i,j,k) must be expressible as
        # sum_r u_r ⊗ v_r ⊗ w_r where u_r[i*m+j] = U[i*m+j, r], etc.
        
        # But this would mean the tensor is all-ones, which is rank 1!
        # That can't be right for matrix multiplication.
        
        # Let me reconsider the indexing. The key insight is that the multiplication tensor
        # is NOT all-ones. It's a specific structure.
        
        # Actually, I think I'm confusing myself. Let me look at this more carefully.
        
        # The matrix multiplication tensor for (n,m,p) is defined such that:
        # C = A * B where A is n×m, B is m×p, C is n×p
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # As a trilinear form: <C, A, B> = sum_{i,j,k} C[i,k] * A[i,j] * B[j,k]
        # = sum_{i,j,k} (sum_j' A[i,j'] * B[j',k]) * A[i,j] * B[j,k]
        # This gets complicated. Let me just use the standard approach.
        
        # The standard way: the multiplication tensor T has:
        # T[a, b, c] = 1 if a = (i,j), b = (j,k), c = (i,k) for some valid i,j,k
        # i.e., the middle index of a and b must match, and c is determined by the first of a and second of b.
        
        # In the flattened indexing:
        # a = i*m + j (for i in [0,n), j in [0,m))
        # b = j*p + k (for j in [0,m), k in [0,p))
        # c = k*n + i (for k in [0,p), i in [0,n))
        
        # The condition for a valid triple is: the j in a must equal the j in b.
        # So T is zero except for triples where the j-index matches.
        
        pass
    
    # Let's just do a randomized greedy approach.
    
    # Build the target: for each (i,j,k), we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # And for triples where the j-indices don't match appropriately, the sum should be 0.
    
    # Actually, the full condition is more subtle. Let me just verify by checking all possible
    # matrix multiplications.
    
    def _verify_decomposition(U, V, W, n, m, p, max_trials=50):
        """Verify that the decomposition correctly computes matrix multiplication."""
        rng = np.random.RandomState(42)
        for _ in range(max_trials):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_expected = A @ B
            
            # Compute using the decomposition
            U_arr = np.array(U)
            V_arr = np.array(V)
            W_arr = np.array(W)
            
            # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * V[j*p+k, r] * W[k*n+i, r]
            # Wait, that's not the right formula. Let me think again.
            
            # The decomposition says: the bilinear map (A, B) -> C is represented by
            # C = sum_r (U^T A_vec) * (V^T B_vec) * W_vec
            # where A_vec is A flattened, etc.
            
            # More precisely:
            # Let U have columns u_r (vectors of length n*m)
            # Let V have columns v_r (vectors of length m*p)
            # Let W have columns w_r (vectors of length p*n)
            
            # Then C[i,k] = sum_r (u_r^T A_flat) * (v_r^T B_flat) * w_r[k*n+i]
            
            # Where A_flat[i*m+j] = A[i,j], B_flat[j*p+k] = B[j,k], C_flat[k*n+i] = C[i,k]
            
            C_computed = np.zeros((p, n))  # in the k*n+i convention
            for r in range(len(U[0])):
                u_r = np.array(U)[:, r]
                v_r = np.array(V)[:, r]
                w_r = np.array(W)[:, r]
                
                A_flat = A.flatten()  # [i*m + j]
                B_flat = B.flatten()  # [j*p + k]
                
                scalar_u = np.dot(u_r, A_flat)
                scalar_v = np.dot(v_r, B_flat)
                
                # Add to C
                c_vec = scalar_u * scalar_v * w_r
                C_computed_flat = C_computed.flatten()
                # Actually, let's just accumulate
                C_computed_flat += c_vec
            C_computed = C_computed_flat.reshape((p, n))
            C_computed = C_computed.T  # convert from [k*n+i] to [i,k]
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True
    
    # Let's try to build a good decomposition.
    # Start with schoolbook (rank 60) and try to reduce.
    
    # Schoolbook:
    def _schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if _verify_decomposition(U, V, W, n, m, p, max_trials=10):
        pass  # Schoolbook is valid
    
    # Try to reduce rank by checking if some columns are redundant
    # For now, return schoolbook as a safe baseline.
    # A more sophisticated approach would use randomized SVD or tensor decomposition methods.
    
    # Given the time constraints and the difficulty of finding a provably lower rank,
    # let's return the schoolbook decomposition.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
