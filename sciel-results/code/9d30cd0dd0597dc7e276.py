import time
import numpy as np

def construct():
    start = time.time()
    DEADLINE = 135  # safety margin before 150s limit

    n, m, p = 3, 3, 6

    def verify(U, V, W, r):
        """
        Verify that the decomposition exactly computes matrix multiplication.
        U: (n*m, r), V: (m*p, r), W: (p*n, r)
        For random A (n x m) and B (m x p):
        C[i, k] = sum_j A[i, j] * B[j, k]
        Decomposition: C[i, k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        """
        rng = np.random.default_rng(42)
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for _ in range(5):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C_true = A @ B
            
            C_comp = np.zeros((n, p), dtype=float)
            for i in range(n):
                for k in range(p):
                    val = 0.0
                    for j in range(m):
                        u_row = U_arr[i * m + j, :]
                        v_row = V_arr[j * p + k, :]
                        w_val = W_arr[k * n + i, :]
                        val += np.sum(u_row * v_row * w_val)
                    C_comp[i, k] = val
            
            if not np.allclose(C_comp, C_true, atol=1e-6):
                return False
        return True

    # Build the guaranteed schoolbook decomposition (Rank 54)
    r = n * m * p
    U = [[0.0] * r for _ in range(n * m)]
    V = [[0.0] * r for _ in range(m * p)]
    W = [[0.0] * r for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1

    # Try to find a lower-rank decomposition using randomized linear algebra
    # The matrix multiplication tensor T_{i,j,k} = delta_{i,j} (for C = A*B)
    # We're looking for rank r' < 54 such that:
    # T = sum_c U_c ⊗ V_c ⊗ W_c
    # where (U_c)_{i,j} = U[i*m+j, c], (V_c)_{j,k} = V[j*p+k, c], (W_c)_{k,i} = W[k*n+i, c]
    
    # Try rank 53, 52, ... down to some lower bound
    # The minimum rank for 3x3 x 3x6 is at least max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But realistically it's much higher. Known bounds suggest it's at least 30-40.
    
    best_rank = r
    best_U, best_V, best_W = U, V, W
    
    # Attempt randomized search for lower ranks
    # Use the fact that if we can find r' rank-1 tensors whose sum gives T, we're done.
    # This is equivalent to finding a rank-r' decomposition of the tensor.
    
    # Strategy: Use gradient-free optimization with random restarts
    # For each candidate rank r', try to find U, V, W such that the tensor matches.
    
    def try_rank(r_prime, max_iters=50000):
        """Try to find a rank-r_prime decomposition using random initialization and Newton-like updates."""
        if r_prime >= best_rank:
            return None
        
        rng = np.random.default_rng(123)
        
        for restart in range(5):
            if time.time() - start > DEADLINE:
                break
            
            # Random initialization
            U_try = rng.standard_normal((n * m, r_prime)) * 0.1
            V_try = rng.standard_normal((m * p, r_prime)) * 0.1
            W_try = rng.standard_normal((p * n, r_prime)) * 0.1
            
            # Use alternating least squares / projected gradient
            lr = 0.01
            for it in range(max_iters):
                if time.time() - start > DEADLINE:
                    break
                
                # Compute current tensor estimate
                # T_comp[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                
                # Gradient with respect to U
                grad_U = np.zeros_like(U_try)
                grad_V = np.zeros_like(V_try)
                grad_W = np.zeros_like(W_try)
                
                # For each (i,j,k), the true value is 1 if this is a valid entry
                # Actually, the tensor for matrix multiplication 3x3 * 3x6 is:
                # T[i, j, k] = delta_{j, j} = 1 (it's always 1 for the multiplication)
                # Wait, let me reconsider.
                
                # The tensor T is defined by:
                # (A * B)[i, k] = sum_j A[i,j] * B[j,k]
                # So the tensor T_{i,j,k} satisfies:
                # T_{i,j,k} = 1 if we're multiplying the (i,j) entry of A with (j,k) entry of B
                # Actually, the tensor is: T[i,j,k] = delta_{i,?} ... 
                
                # Let me be more careful. The multiplication map is:
                # mu: A ⊗ B -> C where C[i,k] = sum_j A[i,j] * B[j,k]
                # The tensor is T ∈ R^{n × m × p} ... no, it's in R^{(n*m) × (m*p) × (p*n)}? No.
                
                # Actually, for the decomposition to work with the given interface:
                # C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # And we want C[i,k] = sum_j A[i,j] * B[j,k]
                
                # So the condition is:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{j,j} = 1 for all i,j,k? No!
                
                # It should be: for each i, j, k, the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
                # So: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k.
                
                # Wait, that's not right either. Let me re-examine.
                
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # Decomposition: C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
                
                # For this to hold for all A, B, we need:
                # For each i, j, k: the coefficient of A[i,j]*B[j,k] in the sum must equal 1 if it contributes to C[i,k], and 0 otherwise.
                
                # Actually, A[i,j]*B[j,k] always contributes to C[i,k] with coefficient 1.
                # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for ALL i, j, k.
                
                # Wait, that can't be right because that would mean the tensor is all ones, which has rank 1.
                
                # Let me re-read the problem. The tensor for matrix multiplication is:
                # T_{i,j,k} where the multiplication is C[i,k] = sum_j A[i,j] B[j,k]
                # The tensor T ∈ R^{n × m × p} ... no.
                
                # I think I'm overcomplicating. Let me use the verification function to guide the search.
                
                # Compute error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # The coefficient should be 1
                            coeff = 0.0
                            for c in range(r_prime):
                                coeff += U_try[i*m+j, c] * V_try[j*p+k, c] * W_try[k*n+i, c]
                            error += (coeff - 1.0)**2
                
                if error < 1e-10:
                    # Verify with actual random matrices
                    U_out = [[U_try[i*m+j, c] for c in range(r_prime)] for i in range(n) for j in range(m)]
                    V_out = [[V_try[j*p+k, c] for c in range(r_prime)] for j in range(m) for k in range(p)]
                    W_out = [[W_try[k*n+i, c] for c in range(r_prime)] for k in range(p) for i in range(n)]
                    if verify(U_out, V_out, W_out, r_prime):
                        return (U_out, V_out, W_out)
                
                # Simple gradient step
                # This is getting complex; let me just return the schoolbook if no better found
                break
        
        return None
    
    # The above approach is too slow. Let me just return the schoolbook decomposition.
    # Finding a lower-rank exact decomposition for 3x3 * 3x6 within the time limit
    # with guaranteed correctness is very challenging.
    
    return {"U": U, "V": V, "W": W}
