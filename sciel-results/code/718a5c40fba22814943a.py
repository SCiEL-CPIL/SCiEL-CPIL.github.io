import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Tensor T of shape (p, n, m) such that T[w, i, j] = 1 if w=k, i=i, j=j corresponds to mult
    # Actually, standard form: C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T has indices i, j, k. T[i, j, k] = 1 if it contributes to C[i, k].
    # In the decomposition: C[i, k] = sum_r U[i, r] V[k, r] W[r] ? No.
    # Standard decomposition: T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r] is for 3-way.
    # For matrix mult, we want: sum_j A[i, j] B[j, k] = C[i, k].
    # This is a tensor of order 3 with indices i, j, k.
    # The decomposition is T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r] ?
    # Wait, the interface specifies:
    # U: (n*m, rank) -> Index i*m + j
    # V: (m*p, rank) -> Index j*p + k
    # W: (p*n, rank) -> Index k*n + i
    # And the trace convention W[k*n+i, r].
    # The reconstruction is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T is defined such that the multiplication is represented.
    # Usually, the matrix multiplication tensor T is defined by T[i, j, k] = 1 if the term A[i, j]B[j, k] appears in C[i, k].
    # There is exactly one such term for each i, j, k. So T[i, j, k] = 1 for all i, j, k?
    # No, that would mean C[i, k] = sum_{j} A[i, j] B[j, k].
    # The decomposition is:
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r] ?
    # If we flatten U to size n*m, V to m*p, W to p*n...
    # Let's check the interface shapes again.
    # U: (n*m, rank). Index u = i*m + j.
    # V: (m*p, rank). Index v = j*p + k.
    # W: (p*n, rank). Index w = k*n + i.
    # The product sum is sum_r U[u, r] V[v, r] W[w, r]?
    # If we sum this over r, we get a scalar for each (u, v, w).
    # But we want to compute C[i, k] from A and B.
    # The standard "tensor rank" of matrix multiplication is the rank of the tensor T where T[i, j, k] = 1 if i, j, k form a valid multiplication term.
    # Wait, for matrix multiplication, the tensor is sparse.
    # T[i, j, k] = 1 if the term A[i, j] B[j, k] contributes to C[i, k].
    # Actually, for every i, j, k, the term A[i, j] B[j, k] contributes to C[i, k].
    # So T[i, j, k] = 1 for ALL i, j, k?
    # If T[i, j, k] = 1 for all i, j, k, then the tensor is all-ones.
    # Rank of all-ones tensor of size n x m x p is 1?
    # No, that's not right.
    # The matrix multiplication tensor is defined such that the bilinear map (A, B) -> C is represented.
    # The tensor T is in V^* \otimes (W^* \otimes X).
    # Let's stick to the interface.
    # The evaluator checks if the decomposition "exactly decomposes the matrix multiplication tensor".
    # This implies there is a specific target tensor T.
    # Based on the seed code:
    # T_slices[w, u, v] = 1 if u=i*m+j, v=j*p+k, w=k*n+i.
    # This means for a fixed w (which encodes k and i), the slice T[w] is a matrix of size (n*m) x (m*p).
    # T[w, u, v] = 1 if u and v are compatible with w.
    # u = i*m + j, v = j*p + k, w = k*n + i.
    # So for a given w = k*n + i, the only non-zero entry in T[w] is at (i*m+j, j*p+k) for j=0..m-1?
    # No, j is free.
    # For fixed i, k, w is fixed.
    # The entries are at u = i*m + j and v = j*p + k.
    # So T[w, i*m+j, j*p+k] = 1 for all j.
    # All other entries are 0.
    
    # So the target is:
    # For each w in 0..p*n-1:
    #   i = w % n
    #   k = w // n
    #   T_slice[w] is a matrix where T_slice[w, i*m+j, j*p+k] = 1 for j in 0..m-1.
    #   Else 0.

    # We want to find U (n*m, R), V (m*p, R), W (p*n, R) such that:
    # T_slice[w] = sum_r W[w, r] * (U[:, r] V[:, r]^T)
    
    # This is a tensor decomposition problem.
    
    # Strategy:
    # 1. Build T_slices.
    # 2. Try ranks R from a high estimate down to a low estimate.
    # 3. For each R, use randomized ALS to find U, V, W.
    # 4. Round to half-integers and verify.

    T_slices = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for k in range(p):
            w = k * n + i
            for j in range(m):
                u = i * m + j
                v = j * p + k
                T_slices[w, u, v] = 1.0

    # Precompute flattened T_slices for faster LSQ
    T_flat = T_slices.reshape(p * n, -1)
    
    start_time = time.time()
    time_limit = 115
    
    # Heuristic range for R
    # Schoolbook is n*m*p = 90.
    # Lower bound is at least n*m + m*p + p*n - 2 ? No, that's for something else.
    # Let's try from 80 down to 50.
    
    best_rank = n * m * p
    best_U = np.zeros((n * m, best_rank))
    best_V = np.zeros((m * p, best_rank))
    best_W = np.zeros((p * n, best_rank))
    
    # Initialize best with schoolbook just in case
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1

    found_solution = False

    for R in range(80, 40, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple random restarts for this rank
        found = False
        for attempt in range(20): # 20 restarts per rank
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V, W
            # Random initialization
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # ALS Optimization
            # We want to minimize || T_flat - W * (U \otimes V) ||_F
            # Where (U \otimes V) is a matrix of size (n*m*m*p, R) with columns vec(U_r V_r^T)
            
            # Precompute basis for current U, V?
            # In each step, we update one factor.
            
            # Step 1: Update W
            # T_flat[w] = sum_r W[w, r] B_r
            # B_r = vec(U_r V_r^T)
            # B is (n*m*m*p, R)
            B = np.zeros((n * m * m * p, R))
            for r in range(R):
                B[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            
            # Solve W @ B.T = T_flat.T ?
            # T_flat[w] = W[w] @ B.T
            # So T_flat.T = B @ W.T
            # W.T = lstsq(B, T_flat.T)
            W_new, residuals, rank_B, s = np.linalg.lstsq(B, T_flat.T, rcond=None)
            W = W_new.T
            
            # Step 2: Update U
            # Fix V, W.
            # Objective: sum_w || T_w - sum_r W[w, r] U_r V_r^T ||_F^2
            # Let's derive the normal equations for U.
            # This is equivalent to flattening the tensor differently.
            # Let's reshape T into a matrix T_U of size (n*m, R * m * p) ?
            # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            # This is not quite right because V and W are coupled in r.
            # Actually, for fixed V, W, the term is sum_r (W[k, r] V[j, r]) U[i, r].
            # Let M[i, j, k] = T[i, j, k].
            # M = sum_r U_r \otimes (V_r \odot W_r) ? No.
            # M[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
            # This is a standard 3-way tensor decomposition.
            # We can solve for U by unfolding the tensor along mode 1.
            # T_(1) is a matrix of size (n, m*p*R) ? No.
            # Let's use the property:
            # T_(1) = U * (W \odot V).T ?
            # Where \odot is Khatri-Rao product (column-wise Kronecker).
            # T_(1)[i, (j, k, r)] = T[i, j, k] if we unroll (j, k, r)?
            # No, the decomposition is sum_r U[i, r] V[j, r] W[k, r].
            # The matricization T_(1) has columns indexed by (j, k).
            # T_(1)[i, j, k] = sum_r U[i, r] (V[j, r] W[k, r]).
            # So T_(1) = U * (V \odot W).T
            # Where (V \odot W) is a matrix of size (m*p, R) with columns V_r \otimes W_r?
            # No, V is (m*p, R) and W is (p*n, R).
            # The indices j and k are coupled in the target tensor structure?
            # In our specific problem, T[w, u, v] is non-zero only if u, v, w are compatible.
            # This structure is crucial.
            # The generic ALS for 3-way tensors might not exploit this sparsity/structure efficiently or correctly if we just treat it as a dense tensor.
            # However, the "Tensor" here is defined by the slices T[w].
            # T[w] is a matrix.
            # T[w] = sum_r W[w, r] U_r V_r^T.
            # This is a linear matrix decomposition for each w, with shared U, V.
            
            # Let's stick to the slice-based ALS.
            # Update U:
            # Minimize sum_w || T_w - sum_r W[w, r] U_r V_r^T ||_F^2
            # = sum_w sum_{u, v} (T_w[u, v] - sum_r W[w, r] U[u, r] V[v, r])^2
            # Derivative w.r.t U[u, r]:
            # -2 sum_w sum_v V[v, r] (T_w[u, v] - sum_{r'} W[w, r'] U[u, r'] V[v, r'])
            # = 0
            # sum_w sum_v V[v, r] T_w[u, v] = sum_w sum_v V[v, r] W[w, r] U[u, r] V[v, r] + sum_{r' != r} ...
            # This is a system of linear equations for U.
            # Let's construct the normal equations matrix A_U and vector b_U.
            # A_U is (n*m*R, n*m*R). This is 90*R x 90*R. If R=50, 4500x4500. Solvable.
            # But we can do better.
            # Note that sum_w sum_v V[v, r] T_w[u, v] can be precomputed if V is fixed?
            # Let C_U[u, r] = sum_w sum_v V[v, r] T_w[u, v].
            # C_U is (n*m, R).
            # The RHS term: sum_w W[w, r] (sum_v V[v, r] V[v, r'] U[u, r']) ?
            # Let G[r, r'] = sum_w W[w, r] W[w, r'] (V_r \cdot V_r') ?
            # No, sum_v V[v, r] V[v, r'] is the dot product of columns r, r' of V.
            # Let H = V.T @ V. (R x R)
            # Let K = W.T @ W. (R x R)
            # Then the term is sum_{r'} U[u, r'] (sum_w W[w, r] W[w, r'] H[r, r'] ) ?
            # No, the sum over w is separate.
            # Term = sum_w W[w, r] W[w, r'] (V_r \cdot V_r') U[u, r']
            # = sum_{r'} U[u, r'] ( (W.T @ W)[r, r'] * (V.T @ V)[r, r'] ) ?
            # No, it's element-wise product of the matrices?
            # Let's check indices.
            # sum_w W[w, r] W[w, r'] (V_r \cdot V_r')
            # This is (W.T @ W)[r, r'] * (V.T @ V)[r, r'] ? No.
            # sum_w W[w, r] W[w, r'] is (W.T @ W)[r, r'].
            # But (V_r \cdot V_r') is inside the sum over v, but independent of w.
            # So yes, it is (W.T @ W)[r, r'] * (V.T @ V)[r, r']?
            # Wait, V_r \cdot V_r' is a scalar.
            # So the coefficient for U[u, r'] is sum_w W[w, r] W[w, r'] * (V_r \cdot V_r')
            # = (V_r \cdot V_r') * (W.T @ W)[r, r']
            
            # So A_U[r, r'] = (V.T @ V)[r, r'] * (W.T @ W)[r, r'] ?
            # This would mean A_U is the Hadamard product of H and K.
            # And b_U[u, r] = sum_w sum_v V[v, r] T_w[u, v].
            
            # Let's verify this derivation.
            # Objective J = sum_w || T_w - sum_r W[w, r] U_r V_r^T ||_F^2
            # J = sum_w tr( (T_w - M_w)^T (T_w - M_w) ) where M_w = sum_r W[w, r] U_r V_r^T
            # dJ/dU[u, r] = -2 sum_w sum_v V[v, r] (T_w[u, v] - sum_{r'} W[w, r'] U[u, r'] V[v, r'])
            # = -2 ( sum_w sum_v V[v, r] T_w[u, v] - sum_{r'} U[u, r'] sum_w W[w, r'] V[v, r] V[v, r'] )
            # The inner sum over v: sum_v V[v, r] V[v, r'] = (V.T @ V)[r, r'] = H[r, r'].
            # So term is sum_{r'} U[u, r'] H[r, r'] sum_w W[w, r'] W[w, r] ?
            # No, W[w, r] is outside the r' sum?
            # sum_w W[w, r] W[w, r'] H[r, r']
            # = H[r, r'] sum_w W[w, r] W[w, r']
            # = H[r, r'] (W.T @ W)[r, r'] = H[r, r'] K[r, r'].
            
            # So the equation is:
            # b_U[u, r] = sum_{r'} A_U[r, r'] U[u, r']
            # where A_U = H \odot K (Hadamard product)
            # and b_U[u, r] = sum_w sum_v V[v, r] T_w[u, v].
            
            # We can solve this system for each u independently?
            # No, A_U is R x R. U is (n*m, R).
            # For each u, we solve A_U @ U[u, :] = b_U[u, :].
            # This is efficient!
            
            # Compute H, K
            H = V.T @ V
            K = W.T @ W
            A_U = H * K
            
            # Compute b_U
            # b_U[u, r] = sum_w sum_v V[v, r] T_w[u, v]
            # We can compute this as:
            # For each r, b_U[:, r] = sum_w (T_w @ V[:, r])
            # T_w is (n*m, m*p). V[:, r] is (m*p, 1).
            # T_w @ V[:, r] is (n*m, 1).
            # Sum over w.
            # b_U = sum_w T_w @ V[:, r] for each r.
            # b_U = (sum_w T_w) @ V ? No, sum_w T_w is not defined as matrix mult.
            # b_U[:, r] = sum_w (T_slices[w] @ V[:, r])
            
            b_U = np.zeros((n * m, R))
            for r in range(R):
                for w in range(p * n):
                    b_U[:, r] += T_slices[w] @ V[:, r]
            
            # Solve A_U X = b_U.T for each u?
            # A_U is R x R. b_U is (n*m, R).
            # We want U_new such that A_U @ U_new.T = b_U.T ?
            # Equation: sum_{r'} A_U[r, r'] U[u, r'] = b_U[u, r]
            # So A_U @ U[u, :] = b_U[u, :]
            # So U_new.T = A_U^{-1} b_U.T
            # Or U_new = b_U @ A_U^{-1}.T
            
            try:
                U_new = np.linalg.solve(A_U, b_U.T).T
            except np.linalg.LinAlgError:
                U_new = np.linalg.lstsq(A_U, b_U.T, rcond=None)[0].T
                
            U = U_new
            
            # Step 3: Update V
            # Symmetric to U.
            # A_V = H \odot K
            # b_V[v, r] = sum_w sum_u U[u, r] T_w[u, v]
            # b_V[:, r] = sum_w (T_w.T @ U[:, r])
            
            b_V = np.zeros((m * p, R))
            for r in range(R):
                for w in range(p * n):
                    b_V[:, r] += T_slices[w].T @ U[:, r]
                    
            try:
                V_new = np.linalg.solve(A_V, b_V.T).T
            except np.linalg.LinAlgError:
                V_new = np.linalg.lstsq(A_V, b_V.T, rcond=None)[0].T
                
            V = V_new
            
            # Check convergence / time
            if time.time() - start_time > time_limit:
                break
            
            # After some iterations, check if we can round
            if attempt > 5: # Do a few iterations of ALS
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                # Reconstruct T_slices
                max_diff = 0
                for w in range(p * n):
                    T_recon = np.zeros((n * m, m * p))
                    for r in range(R):
                        if W_rounded[w, r] != 0:
                            T_recon += W_rounded[w, r] * np.outer(U_rounded[:, r], V_rounded[:, r])
                    diff = np.max(np.abs(T_recon - T_slices[w]))
                    if diff > max_diff:
                        max_diff = diff
                    if max_diff > 1e-9:
                        break
                
                if max_diff < 1e-9:
                    found = True
                    best_rank = R
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    break

        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
