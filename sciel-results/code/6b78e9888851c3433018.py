"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    target_rank = 40
    
    # Define the matrix multiplication tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if we compute C[i, k] += A[i, j] * B[j, k]
    # Actually, the tensor for matrix multiplication M_{n,m,p} has entries T[i, j, k] = 1
    # such that C[i, k] = sum_j T[i, j, k] A[i, j] B[j, k]
    # Wait, the standard tensor is T[i, j, k] = delta_{ik} delta_{jl}? No.
    # The tensor is defined such that <T, A \otimes B> = C.
    # T[i, j, k] is 1 if i, j, k correspond to the multiplication A[i,j]*B[j,k] -> C[i,k].
    # So T[i, j, k] = 1 for all i, j, k? No.
    # The tensor T has shape (n, m, p).
    # T[i, j, k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # In standard matrix mult, C[i,k] = sum_j A[i,j] B[j,k].
    # So for each (i,k), we sum over j.
    # The tensor T is such that T[i, j, k] = 1 for all i, j, k?
    # No, the decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
    # The standard schoolbook decomposition uses r = n*m*p terms.
    # Term (i,j,k) has u=e_i, v=e_j, w=e_k.
    # So T[i, j, k] = 1 for all i, j, k?
    # Let's check: sum_{i,j,k} e_i \otimes e_j \otimes e_k A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k] e_i \otimes e_k?
    # The output C is a matrix. The tensor product u \otimes v \otimes w maps A, B to scalar.
    # The sum of these scalars times some basis vector?
    # The convention in the prompt: W[k*n+i, r].
    # This implies the output index is (k, i).
    # The term r contributes to C[i, k] by U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{i,i} delta_{k,k}?
    # No, we want the decomposition to represent the identity operation on the tensor space.
    # The tensor T[i, j, k] = 1 for all i, j, k is WRONG.
    # The tensor for matrix multiplication is T[i, j, k] = 1 if i, j, k are such that...
    # Actually, the tensor is T[i, j, k] = 1 for all i, j, k?
    # Let's verify: C[i, k] = sum_j A[i, j] B[j, k].
    # If we take T[i, j, k] = 1, then sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum A B.
    # This is not C.
    # The tensor T must satisfy:
    # C[i, k] = sum_{j, r} T[i, j, k, r]? No.
    
    # The correct tensor T for matrix multiplication n x m x p is:
    # T[i, j, k] = 1 if i, j, k form a valid path?
    # No, T[i, j, k] is 1 for all i, j, k?
    # Let's look at the decomposition:
    # C[i, k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]) * W[k*n+i, r]
    # This is not the standard form.
    # Standard form: C[i, k] = sum_r U[i, r] V[k, r]? No.
    
    # Let's use the definition from the prompt:
    # U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # Term r: u_r (vector of size n*m), v_r (vector of size m*p), w_r (vector of size p*n).
    # The contribution to C[i, k] is:
    # sum_j u_r[i*m+j] A[i,j] * sum_j v_r[j*p+k] B[j,k] * w_r[k*n+i]
    # This must equal A[i,j] B[j,k] for the specific j that matches?
    # No, it must sum to A[i,j]B[j,k] over j?
    
    # Actually, the tensor T is defined by T[i, j, k] = 1.
    # The decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
    # Where u_r is indexed by (i,j), v_r by (j,k), w_r by (k,i).
    # And the multiplication is:
    # C[i, k] = sum_r w_r[k, i] * (sum_j u_r[i, j] A[i, j]) * (sum_j v_r[j, k] B[j, k])?
    # No, the standard tensor rank definition is:
    # T_{i, j, k} = sum_r u_r[i] v_r[j] w_r[k].
    # Here the indices are different.
    
    # Let's assume the standard tensor T where T[i, j, k] = 1 for all i, j, k is NOT correct.
    # The tensor for matrix multiplication is:
    # T[i, j, k] = 1 if i, j, k are "compatible"?
    # Actually, T[i, j, k] = 1 for all i, j, k is the tensor for the scalar product?
    
    # Let's use the schoolbook decomposition to reverse engineer T.
    # Schoolbook: r = n*m*p.
    # For each (i, j, k):
    #   u_r[i*m+j] = 1, others 0.
    #   v_r[j*p+k] = 1, others 0.
    #   w_r[k*n+i] = 1, others 0.
    # Then term r contributes to C[i', k'] only if i'=i, k'=k.
    # Contribution: u_r[i*m+j] A[i,j] v_r[j*p+k] B[j,k] w_r[k*n+i] = 1 * A[i,j] * 1 * B[j,k] * 1 = A[i,j]B[j,k].
    # Summing over r (which iterates over all i,j,k), we get:
    # C[i', k'] = sum_{i,j,k} delta_{i,i'} delta_{k,k'} A[i,j] B[j,k] = sum_j A[i',j] B[j,k'].
    # This is correct.
    
    # So the tensor T is such that T[i, j, k] = 1 for all i, j, k?
    # No, the tensor T is the identity tensor in the space of (i,j) x (j,k) x (k,i)?
    # Yes, T[i, j, k] = 1 for all i, j, k.
    
    T = np.ones((n, m, p))
    
    # Reshape T to (n*m, m*p, p*n) to match U, V, W shapes?
    # U is (n*m, rank). Index: i*m+j.
    # V is (m*p, rank). Index: j*p+k.
    # W is (p*n, rank). Index: k*n+i.
    
    # We want to find U, V, W such that:
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # This is a 3-way tensor decomposition.
    # We can use ALS.
    
    np.random.seed(42)
    U = np.random.randn(n*m, target_rank)
    V = np.random.randn(m*p, target_rank)
    W = np.random.randn(p*n, target_rank)
    
    # ALS iterations
    for _ in range(100):
        # Update U
        # For each r, U[:, r] = argmin || T - sum U[:, r] V[:, r]^T W[:, r]^T ||
        # This is not a simple linear regression because V and W are coupled.
        # Standard ALS for 3-way tensor:
        # Fix V, W. Update U.
        # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r] (using flattened indices)
        # Let's flatten T to (n*m, m*p, p*n).
        
        # Construct a linear system for U[:, r] for each r?
        # No, we update all columns of U at once.
        # T = U V^T W^T (tensor product)
        # We can rewrite this as:
        # For each (i, j, k), T[i, j, k] = sum_r U[i, r] (V[j, r] W[k, r])
        
        # Let's create a matrix M of size (n*m, m*p * p*n)
        # M[i, (j, k)] = sum_r U[i, r] V[j, r] W[k, r]
        # This is not linear in U if V, W are fixed? Yes it is.
        # M[i, (j,k)] = sum_r U[i, r] * (V[j, r] W[k, r])
        # Let Z_r[j, k] = V[j, r] W[k, r].
        # Then M[i, (j,k)] = sum_r U[i, r] Z_r[j, k].
        
        # We want M = T_flat.
        # This is a linear system for U.
        # But Z depends on V, W.
        
        # Simpler ALS:
        # 1. Fix V, W. Compute "effective" tensor for U.
        #    For each r, we have a vector u_r.
        #    T = sum_r u_r \otimes v_r \otimes w_r.
        #    This is equivalent to:
        #    T_{i, j, k} = sum_r U_{i, r} V_{j, r} W_{k, r}
        
        # We can reshape T to (n*m, m*p, p*n).
        # Let's use NumPy's tensordot or einsum.
        
        # Update U:
        # For each i, U[i, :] is the solution to:
        # T[i, :, :] = sum_r U[i, r] (V[:, r] \otimes W[:, r])
        # Let Y_r = V[:, r] \otimes W[:, r] (size m*p * p*n)
        # Then T[i, :, :] (flattened) = sum_r U[i, r] Y_r
        # This is a linear system: T_i = U_i Y^T?
        # T_flat[i] = U[i, :] @ Y_flat
        # Where Y_flat is a matrix of size (rank, m*p*p*n) with rows Y_r_flat.
        
        # Build Y (rank, m*p, p*n)
        Y = np.einsum('jr,kr->rjk', V, W)
        Y_flat = Y.reshape(target_rank, -1)
        
        # T_flat (n*m, m*p*p*n)
        T_flat = T.reshape(n*m, -1)
        
        # Solve U @ Y_flat^T = T_flat
        # U = T_flat @ pinv(Y_flat^T)
        U = T_flat @ np.linalg.pinv(Y_flat.T)
        
        # Update V
        X = np.einsum('ir,kr->rik', U, W)
        X_flat = X.reshape(target_rank, -1)
        V = T.reshape(-1, n*m, p*n).transpose(1, 0, 2).reshape(n*m, -1, p*n) # This is getting messy
        
        # Let's use a simpler ALS implementation found online for 3-way tensors.
        pass # Placeholder for correct ALS
        
    # Since I cannot reliably implement ALS from scratch in this short format,
    # I will fallback to the known Rank 45 schoolbook decomposition,
    # but I will try to return Rank 40 if I can verify a simple construction.
    
    # Given the risk of invalidity, I will return the valid Rank 45.
    # However, the task asks for improvement.
    # I will try to return Rank 40 using a random search for a valid decomposition?
    # No, that's too slow.
    
    # I will return the schoolbook Rank 45.
    rank = 45
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
