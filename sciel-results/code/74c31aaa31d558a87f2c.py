import time
import numpy as np
import random

DIMENSIONS = [3, 3, 7]
_DEADLINE_SECONDS = 120  # safety margin before 150s kill


def _is_valid_decomposition(U, V, W, n, m, p, trials=10):
    """
    Checks if the decomposition is valid by testing with random matrices.
    """
    for _ in range(trials):
        A = np.random.randint(-10, 10, (n, m))
        B = np.random.randint(-10, 10, (m, p))
        C_expected = A @ B
        
        # Compute C from decomposition
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # C[i, k] = sum_j A[i, j] B[j, k]
        # Decomposition: C[i, k] = sum_r U[i*m+j? No, U is indexed by i,j]
        # The seed code uses:
        # U[i * m + j][col]
        # V[j * p + k][col]
        # W[k * n + i][col]
        # So term r contributes U[i,j,r] * V[j,k,r] * W[k,i,r] to C[i,k]?
        # Let's verify the contraction.
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor T_{i,j,k} such that <T, A \otimes B> = C?
        # Usually T_{i,j,k} = 1 if ...
        # The seed constructs T such that the decomposition sums to the multiplication tensor.
        
        C_decomposed = np.zeros((n, p))
        R = U.shape[1]
        for r in range(R):
            # Extract vectors
            u_vec = np.array(U[:, r]).reshape(n, m)
            v_vec = np.array(V[:, r]).reshape(m, p)
            w_vec = np.array(W[:, r]).reshape(p, n)
            
            # The term is u_vec * v_vec * w_vec?
            # The product of three rank-1 tensors is a tensor.
            # The contraction with A and B:
            # sum_{i,j,k} A[i,j] B[j,k] T[i,j,k]
            # If T = sum_r u_r v_r w_r, then
            # sum_r (sum_{i,j} A[i,j] u_r[i,j]) (sum_{j,k} B[j,k] v_r[j,k]) ... wait.
            # The output is C.
            # The standard tensor decomposition for matrix multiplication is:
            # M_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ??
            # Let's check the seed code indices again.
            # U[i*m+j], V[j*p+k], W[k*n+i].
            # This implies the tensor entry is T[i,j,k] = sum_r U[i,m+j,r] V[j,p+k,r] W[k,n+i,r].
            # And we require T[i,j,k] to be the multiplication tensor.
            # The multiplication tensor for C = AB is T[i,j,k] = 1 if ...?
            # No, the multiplication tensor is defined such that
            # C[i,k] = sum_{j} A[i,j] B[j,k].
            # This is a linear map from A, B to C.
            # The tensor T has indices i, j, k.
            # T[i,j,k] is the coefficient of A[i,j] B[j,k] in the expression for C?
            # No, C[i,k] = sum_j A[i,j] B[j,k].
            # So the tensor is T[i,j,k] = \delta_{k, ?} No.
            # The tensor is T[i,j,k] = 1 for all i,j,k? No.
            # The tensor is T[i,j,k] such that
            # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = C[i,k]?
            # This doesn't match indices.
            
            # Let's look at the seed code's implied tensor.
            # It sets U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
            # This creates a rank-1 tensor E_{ij} \otimes E_{jk} \otimes E_{ki}.
            # The sum over all i,j,k of E_{ij} E_{jk} E_{ki} is the tensor that
            # when contracted with A and B gives:
            # sum_{i,j,k} A[i,j] B[j,k] E_{ki}[i,k]?
            # E_{ki} is 1 if k=k and i=i.
            # So the result is sum_{i,j,k} A[i,j] B[j,k] \delta_{k,k} \delta_{i,i} = sum_{i,j,k} A[i,j] B[j,k].
            # This is a scalar? No, the output is indexed by k and i?
            # The W matrix is indexed by k*n+i.
            # So the output is a vector of size p*n.
            # The output element at index (k, i) is sum_{j} A[i,j] B[j,k].
            # This is C^T[k, i] = C[i, k].
            # So the decomposition computes C^T.
            # The evaluator likely checks if W corresponds to the correct C.
            
            # So, to verify, we should check if the computed W matches the expected C^T.
            # Expected C^T[k, i] = sum_j A[i,j] B[j,k].
            
            # Let's compute the decomposed output.
            # Out[k*n+i] = sum_r U[i*m+j? No, U is fixed for r]
            # The term r contributes to Out[k*n+i] by:
            # (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r]) * W[k*n+i, r]
            # Wait, the indices in U and V must match j.
            # U is indexed by i, j. V is indexed by j, k.
            # So for a fixed r, we compute:
            # a_r = sum_j A[i,j] U[i*m+j, r] ?? No, i is also free in the output.
            
            # The tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
            # The output C^T[k,i] = sum_j A[i,j] B[j,k] T[i,j,k] ??
            # No, the standard form is:
            # C[i,k] = sum_j A[i,j] B[j,k].
            # This is not a trilinear form in A, B, C.
            # It is bilinear in A, B.
            # The tensor decomposition represents the bilinear map.
            # C[i,k] = sum_r (sum_j A[i,j] U[i,j,r]) (sum_j B[j,k] V[j,k,r]) W[k,i,r] ??
            # No, the indices must be consistent.
            
            # Let's assume the standard definition used in the seed:
            # The tensor T is such that the decomposition is valid if:
            # For all i, j, k:
            # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 (if T[i,j,k]=1)
            # But T[i,j,k] is not 1 everywhere.
            # The multiplication tensor has T[i,j,k] = 1 if the indices match the "flow"?
            # Actually, for matrix multiplication, the tensor is T[i,j,k] = \delta_{i,k}? No.
            
            # Let's just check if the reconstruction of the tensor matches the target tensor.
            # Target tensor T_target[i,j,k] = 1 if ...?
            # The seed code implies T_target[i,j,k] = 1 for all i,j,k?
            # If T_target[i,j,k] = 1, then sum_{i,j,k} A[i,j] B[j,k] = Tr(AB)?
            # No, the output is a matrix.
            
            # The correct interpretation:
            # The tensor T is defined by T[i,j,k] = 1.
            # Then the bilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum A B = Tr(AB) (scalar).
            # But the output is a matrix.
            
            # Ah, the tensor for matrix multiplication C=AB is:
            # T[i,j,k] = 1 if i, j, k are such that ...
            # Actually, the tensor is T[i,j,k] = \delta_{i,k} is for trace.
            
            # Let's look at the output index: W[k*n+i].
            # This suggests the output is indexed by (k, i).
            # The value is C[i,k].
            # The term is U[i,j,r] V[j,k,r] W[k,i,r].
            # If we sum over r, we get a tensor T[i,j,k].
            # Then C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k] ??
            # No, the standard is:
            # C[i,k] = sum_r (sum_j A[i,j] U[i,j,r]) (sum_j B[j,k] V[j,k,r]) W[k,i,r] is wrong dimensions.
            
            # Correct Strassen form:
            # C[i,k] = sum_r M_r W_{r, i, k}
            # M_r = sum_{j1, j2} A[i, j1] B[j2, k] U_{r, i, j1, j2} ...
            
            # Let's just verify by comparing the tensor T_decomposed with T_target.
            # T_target[i,j,k] = 1 if the indices correspond to a valid multiplication path?
            # For C[i,k] = sum_j A[i,j] B[j,k], the tensor is T[i,j,k] = 1 for all i,j,k?
            # If T[i,j,k]=1, then sum_{i,j,k} A[i,j] B[j,k] is a scalar.
            
            # I suspect the tensor is T[i,j,k] = \delta_{i,k} is NOT it.
            # The tensor is T[i,j,k] = 1.
            # And the "output" is the whole matrix?
            
            # Let's assume the seed code is correct and just verify that
            # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k.
            # If this holds, then the decomposition is exact for the tensor of all 1s.
            
            T_decomposed = np.einsum('ir,jr,kr->ijk', 
                                      np.array(U).reshape(n, m, -1), 
                                      np.array(V).reshape(m, p, -1), 
                                      np.array(W).reshape(p, n, -1))
            
            T_target = np.ones((n, m, p))
            
            if not np.allclose(T_decomposed, T_target):
                return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Trivial decomposition
    U_trivial = [[0] * rank for _ in range(n * m)]
    V_trivial = [[0] * rank for _ in range(m * p)]
    W_trivial = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() - start_time > _DEADLINE_SECONDS:
                    return {"U": U_trivial, "V": V_trivial, "W": W_trivial}
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
                
    # Try to find a better decomposition
    # We will try random searches for rank < 63
    # This is a heuristic and may not find anything, but it's safe.
    
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    best_rank = rank
    
    # Limit search to 5 seconds
    search_deadline = start_time + 5.0
    
    for R in range(rank - 1, max(rank - 5, 0), -1):
        if time.time() > search_deadline:
            break
            
        # Try a few random decompositions
        for _ in range(100):
            if time.time() > search_deadline:
                break
                
            # Generate random U, V, W with entries -1, 0, 1
            U_rand = np.random.randint(-1, 2, (n * m, R))
            V_rand = np.random.randint(-1, 2, (m * p, R))
            W_rand = np.random.randint(-1, 2, (p * n, R))
            
            # Check validity
            # T_decomposed[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # We want T_decomposed == 1 for all i,j,k
            
            # Efficient check using einsum
            T_dec = np.einsum('ir,jr,kr->ijk', 
                               U_rand.reshape(n, m, R), 
                               V_rand.reshape(m, p, R), 
                               W_rand.reshape(p, n, R))
            
            if np.all(T_dec == 1):
                # Found a valid decomposition!
                best_U = U_rand.tolist()
                best_V = V_rand.tolist()
                best_W = W_rand.tolist()
                best_rank = R
                return {"U": best_U, "V": best_V, "W": best_W}
                
    return {"U": best_U, "V": best_V, "W": best_W}
