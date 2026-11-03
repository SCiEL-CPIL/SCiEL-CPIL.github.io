"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Target output shape: p*n = 28
    # U shape: n*m = 16
    # V shape: m*p = 28
    
    # The multiplication tensor T has indices (i, j, k) for i in 0..n-1, j in 0..m-1, k in 0..p-1
    # We want to decompose T = sum_{r} u_r \otimes v_r \otimes w_r
    # U[i*m+j, r] = u_r[i,j]
    # V[j*p+k, r] = v_r[j,k]
    # W[k*n+i, r] = w_r[k,i]
    
    # Strategy:
    # 1. Try to find a rank r < 112.
    # 2. We fix U to have a specific structure to reduce search space or make linear system solvable.
    #    However, a direct linear solve for V and W given U is not straightforward because 
    #    the decomposition is a sum of outer products.
    #    Actually, the condition is:
    #    T[i,j,k] = sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is a multilinear equation.
    # If we fix U and W, V is not linear.
    # If we fix U, the equation is:
    #    T[i,j,k] = sum_{r} U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r])
    # Let Z[j,k,i] = sum_{r} V[j*p+k, r] * W[k*n+i, r]. This is the product of V and W matrices (transposed appropriately).
    # Specifically, if we view V as a matrix of size (m*p) x R and W as (p*n) x R.
    # This doesn't form a simple matrix product in standard indices unless we reshape.
    
    # Alternative Strategy: Randomized Linear Algebra / Rank Estimation
    # We can try to find a decomposition of rank R by constructing U, V, W such that they satisfy the tensor equation.
    # A common heuristic for finding low-rank decompositions of matrix multiplication tensors is to use 
    # "structured" bases. For 4x4x4, Strassen's algorithm gives rank 7.
    # For 4x4x7, we can potentially use a block approach.
    
    # Let's try a specific structured ansatz for U, V, W.
    # Since n=4, m=4, p=7.
    # We can view the 4x4 matrix A and 4x7 matrix B.
    # C = A B.
    
    # Let's attempt a search over ranks R from 100 down to 80 (conservative) or lower.
    # We will use a randomized approach to initialize U, then solve for V and W?
    # No, solving for V and W given U is still non-linear (bilinear in V and W).
    
    # Better approach: 
    # The tensor T can be seen as a matrix of size (n*m) * (m*p) for each k? No.
    # T is a 3-way tensor.
    
    # Let's use the fact that if we fix the "U" part to be a random matrix of rank R, 
    # we can try to find V and W such that the reconstruction is exact.
    # Actually, a simpler structured approach:
    # Use the Kronecker product structure.
    # A (4x4) can be viewed as a 2x2 block matrix of 2x2 blocks.
    # B (4x7) can be viewed as a 2x2 block matrix of 2x(7/2)? No, 7 is odd.
    
    # Let's try a direct optimization/search for a small number of ranks.
    # We will use a "fix U, solve for V, W" strategy where we alternate?
    # Or better: Use the known low-rank decompositions for smaller cases and combine.
    
    # Known:
    # 2x2x2: Rank 7 (Strassen)
    # 4x4x4: Rank 49 (using Strassen on 2x2 blocks? 7*7=49) or lower (Laderman 1979 for 3x3x3 is 23, 4x4x4 is ~28-30?)
    # Actually, 4x4x4 rank is at most 49 (Strassen) but can be lower.
    # For 4x4x7, we can split B into B1 (4x4) and B2 (4x3).
    # C1 = A B1 (4x4), C2 = A B2 (4x3).
    # Rank(C1) + Rank(C2) is an upper bound.
    # If we use rank 49 for C1 and rank 4*3*4=48 (schoolbook) for C2, total 97.
    # If we use Strassen for C1 (rank 7 per 2x2 block? No, 4x4x4 Strassen is 49).
    # Wait, Strassen for 4x4x4:
    # Split into 2x2 blocks. 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7 (Strassen).
    # So 4x4x4 rank <= 7 * 7 = 49.
    
    # For 4x4x3:
    # Split A into 2x2 blocks.
    # B2 is 4x3.
    # This is harder.
    
    # Let's try a generic randomized search for a low rank decomposition.
    # We will fix R and try to find U, V, W.
    # We can use a gradient-free approach or linear programming if we fix two matrices.
    
    # If we fix U and W, the equation for V is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is linear in V if U and W are fixed!
    # Let's verify:
    # For each (i,j,k), we have a linear equation in the variables V[j*p+k, r].
    # However, V[j*p+k, r] is multiplied by U[i*m+j, r] * W[k*n+i, r].
    # So for a fixed (j,k), the variable V[j*p+k, r] appears in equations for all i.
    # Let's group by (j,k).
    # For fixed j,k, we have equations for all i:
    # T[i,j,k] = sum_r (U[i*m+j, r] * W[k*n+i, r]) * V[j*p+k, r]
    # Let A_{i,r} = U[i*m+j, r] * W[k*n+i, r].
    # Then T[i,j,k] = sum_r A_{i,r} * V[j*p+k, r].
    # This is a linear system A * V_jk = T_jk, where V_jk is a vector of length R.
    # If we can find U and W such that for all (j,k), the linear system is consistent and has a solution,
    # then we have a valid decomposition.
    
    # So the strategy is:
    # 1. Choose R.
    # 2. Randomly initialize U and W.
    # 3. For each (j,k), check if the linear system A_{j,k} * V_{j,k} = T_{j,k} has a solution.
    #    (i.e., T_{j,k} is in the column space of A_{j,k}).
    # 4. If yes for all (j,k), we found a decomposition.
    
    # This is a feasibility problem. We can iterate by updating U or W to make the systems more consistent.
    # Or simply try random U and W and see if we get lucky? Unlikely for low R.
    
    # We can use a local search:
    # Start with schoolbook (R=112).
    # Try to reduce R by 1, 2, etc.
    # For a given R, initialize U, V, W with random values (or structured).
    # Then alternate:
    #   - Fix U, W. Solve for V (least squares). Check residual.
    #   - Fix V, W. Solve for U (least squares). Check residual.
    #   - Fix U, V. Solve for W (least squares). Check residual.
    # This is ALS (Alternating Least Squares) for tensor decomposition.
    # If the residual reaches 0, we have an exact decomposition.
    
    # Let's implement ALS.
    
    def random_matrix(rows, cols, seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        return rng.randint(-1, 2, size=(rows, cols))  # Integer entries
    
    # We want integer/half-integer entries. 
    # The problem states "integer/half-integer entries".
    # ALS might produce floats. We can round them and check if the rounding preserves the decomposition.
    # Or we can constrain the search to integers.
    
    # Let's try to find a decomposition with integer entries.
    # We will use a randomized search with ALS refinement.
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Precompute T as a tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1  # Wait, the multiplication tensor is not all 1s.
                # The tensor T_{ijk} is 1 if the indices match the multiplication pattern?
                # No, the tensor T represents the bilinear map.
                # T is the tensor such that C = A * B is given by contracting A and B with T.
                # Specifically, C_{ik} = sum_j A_{ij} B_{jk}.
                # This can be written as C_{ik} = sum_{i,j,k} T_{ijk} A_{ij} B_{jk} ? No.
                # The standard definition:
                # T is a tensor of size (n*m) x (m*p) x (p*n) ? No.
                # The decomposition is U \otimes V \otimes W.
                # U is (n*m) x R, V is (m*p) x R, W is (p*n) x R.
                # The product is sum_r U[:,r] V[:,r] W[:,r].
                # This produces a tensor of size (n*m) x (m*p) x (p*n).
                # Let's call this tensor X.
                # X[a, b, c] = sum_r U[a,r] V[b,r] W[c,r].
                # We require X to match the "multiplication tensor" M.
                # What is M?
                # M[a, b, c] is 1 if a=i*m+j, b=j*p+k, c=k*n+i, and 0 otherwise.
                # Let's verify:
                # If we contract M with A (reshaped to n*m) and B (reshaped to m*p), we get C (reshaped to p*n).
                # C[c] = sum_{a,b} M[a,b,c] A[a] B[b].
                # c = k*n+i.
                # M[a,b,c] is non-zero only if a=i*m+j, b=j*p+k.
                # So sum_{a,b} M[a,b,c] A[a] B[b] = sum_{j} A[i*m+j] B[j*p+k].
                # This is exactly C_{ik} = sum_j A_{ij} B_{jk}.
                # So M is the tensor with 1s at (i*m+j, j*p+k, k*n+i) for all i,j,k.
    
    M = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a, b, c] = 1.0
    
    # Now we want to approximate M with sum_{r=0}^{R-1} U[:,r] \otimes V[:,r] \otimes W[:,r].
    # This is a rank-R decomposition of the 3-way tensor M.
    
    # ALS Strategy:
    # 1. Initialize U, V, W randomly.
    # 2. Iterate:
    #    - Fix V, W. Solve for U.
    #      M[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    #      For each a, we have a linear system in U[a,:].
    #      Let Y_a[b,c] = M[a,b,c].
    #      Let Z_a[b,c] = sum_r V[b,r] W[c,r].
    #      Then Y_a = U[a,:] \cdot Z_a ? No.
    #      Y_a[b,c] = sum_r U[a,r] (V[b,r] W[c,r]).
    #      This is a linear system: U[a,:] * (V \otimes W)_{:,:,a} = Y_a.
    #      Specifically, let S_r[b,c] = V[b,r] W[c,r].
    #      Then Y_a[b,c] = sum_r U[a,r] S_r[b,c].
    #      This is a linear least squares problem for U[a,:] given Y_a and S.
    #    - Similarly for V and W.
    
    # We will try ranks from 100 down to 80.
    # If we find a solution with residual < 1e-9, we check if we can round to integers/half-integers.
    
    for R in range(100, 79, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline:
                break
            
            # Initialize with integers
            U = random_matrix(n*m, R, seed=trial*100+R)
            V = random_matrix(m*p, R, seed=trial*100+R+1)
            W = random_matrix(p*n, R, seed=trial*100+R+2)
            
            # ALS iterations
            for it in range(50):
                if time.time() > deadline:
                    break
                
                # Update U
                # For each a, solve U[a,:] * S = Y_a
                # S is R x (m*p) x (p*n). We can flatten S to R x (m*p*p*n).
                # Y_a is (m*p) x (p*n). Flatten to (m*p*p*n).
                # U[a,:] is (R,).
                # System: S_flat.T * U[a,:] = Y_a_flat
                # S_flat[r, idx] = V[b,r] W[c,r] where idx = b*(p*n)+c.
                
                # Precompute S for current V, W
                # S_r = V[:,r] \otimes W[:,r]
                # Shape: (R, m*p, p*n)
                S = np.einsum('br,cr->bcr', V, W)  # (m*p, p*n, R) -> transpose to (R, m*p, p*n)
                S = S.transpose(2, 0, 1) # (R, m*p, p*n)
                S_flat = S.reshape(R, -1) # (R, m*p*p*n)
                
                for a in range(n*m):
                    Y_a = M[a].reshape(-1)
                    # Solve S_flat.T * U[a,:] = Y_a
                    # Use least squares
                    try:
                        U[a,:], residuals, rank, sv = np.linalg.lstsq(S_flat.T, Y_a, rcond=None)
                    except:
                        pass
                
                # Update V
                # For each b, solve V[b,:] * T_b = Z_b
                # T_r[a,c] = U[a,r] W[c,r]
                T_ = np.einsum('ar,cr->acr', U, W) # (a, c, r) -> (r, a, c)
                T_ = T_.transpose(2, 0, 1) # (r, a, c)
                T_flat = T_.reshape(R, -1) # (R, n*m*p*n)
                
                for b in range(m*p):
                    Z_b = M[:, b, :].reshape(-1) # (n*m, p*n) -> flatten
                    try:
                        V[b,:], residuals, rank, sv = np.linalg.lstsq(T_flat.T, Z_b, rcond=None)
                    except:
                        pass
                
                # Update W
                # For each c, solve W[c,:] * P_c = Q_c
                # P_r[a,b] = U[a,r] V[b,r]
                P_ = np.einsum('ar,br->abr', U, V) # (a, b, r) -> (r, a, b)
                P_ = P_.transpose(2, 0, 1) # (r, a, b)
                P_flat = P_.reshape(R, -1) # (R, n*m*m*p)
                
                for c in range(p*n):
                    Q_c = M[:, :, c].reshape(-1) # (n*m, m*p) -> flatten
                    try:
                        W[c,:], residuals, rank, sv = np.linalg.lstsq(P_flat.T, Q_c, rcond=None)
                    except:
                        pass
                
                # Check residual
                # Reconstruct M
                M_recon = np.einsum('ar,br,cr->abc', U, V, W)
                residual = np.linalg.norm(M - M_recon)
                
                if residual < 1e-9:
                    # Found a solution!
                    # Try to round to integers or half-integers
                    # Check if rounding preserves the decomposition
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    M_recon_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                    residual_int = np.linalg.norm(M - M_recon_int)
                    
                    if residual_int < 1e-9:
                        # Success
                        best_rank = R
                        best_U = U_int
                        best_V = V_int
                        best_W = W_int
                        break
                    else:
                        # Try rounding to integers only
                        U_int2 = np.round(U)
                        V_int2 = np.round(V)
                        W_int2 = np.round(W)
                        M_recon_int2 = np.einsum('ar,br,cr->abc', U_int2, V_int2, W_int2)
                        residual_int2 = np.linalg.norm(M - M_recon_int2)
                        if residual_int2 < 1e-9:
                            best_rank = R
                            best_U = U_int2
                            best_V = V_int2
                            best_W = W_int2
                            break
                
                if best_rank == R:
                    break
            
            if best_rank == R:
                break
        
        if best_rank == R:
            break
    
    # If no better rank found, return schoolbook
    if best_U is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    else:
        # Convert numpy arrays to lists
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
