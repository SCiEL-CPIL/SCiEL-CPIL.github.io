"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Build the multiplication tensor
    # T[i, j, k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) of C
    # U index: i*m + j (0..7)
    # V index: j*p + k (0..19)  
    # W index: k*n + i (0..9)
    
    num_u = n * m  # 8
    num_v = m * p  # 20
    num_w = p * n  # 10
    
    # Build tensor T of shape (num_u, num_v, num_w)
    T = np.zeros((num_u, num_v, num_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Try to find low-rank decomposition via iterative refinement
    # Start with the known rank-36 solution and try to compress
    
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, 20s margin
    
    # First, build the rank-36 solution as baseline
    def build_rank36():
        rank = 36
        U = np.zeros((num_u, rank))
        V = np.zeros((num_v, rank))
        W = np.zeros((num_w, rank))
        
        col = 0
        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            u00 = a_r * m + a_c
            u01 = a_r * m + a_c + 1
            u10 = (a_r + 1) * m + a_c
            u11 = (a_r + 1) * m + a_c + 1
            
            v00 = b_r * p + b_c
            v01 = b_r * p + b_c + 1
            v10 = (b_r + 1) * p + b_c
            v11 = (b_r + 1) * p + b_c + 1
            
            w00 = c_c * n + c_r
            w01 = (c_c + 1) * n + c_r
            w10 = c_c * n + c_r + 1
            w11 = (c_c + 1) * n + c_r + 1
            
            # M1 = (a00+a11)(b00+b11)
            U[u00, col] += 1; U[u11, col] += 1
            V[v00, col] += 1; V[v11, col] += 1
            W[w00, col] += 1; W[w11, col] += 1
            col += 1
            
            # M2 = (a10+a11)b00
            U[u10, col] += 1; U[u11, col] += 1
            V[v00, col] += 1
            W[w10, col] += 1; W[w11, col] -= 1
            col += 1
            
            # M3 = a00(b01-b11)
            U[u00, col] += 1
            V[v01, col] += 1; V[v11, col] -= 1
            W[w01, col] += 1; W[w11, col] += 1
            col += 1
            
            # M4 = a11(b10-b00)
            U[u11, col] += 1
            V[v10, col] += 1; V[v00, col] -= 1
            W[w00, col] += 1; W[w10, col] += 1
            col += 1
            
            # M5 = (a00+a01)b11
            U[u00, col] += 1; U[u01, col] += 1
            V[v11, col] += 1
            W[w00, col] -= 1; W[w01, col] += 1
            col += 1
            
            # M6 = (a10-a00)(b00+b01)
            U[u10, col] += 1; U[u00, col] -= 1
            V[v00, col] += 1; V[v01, col] += 1
            W[w11, col] += 1
            col += 1
            
            # M7 = (a01-a11)(b10+b11)
            U[u01, col] += 1; U[u11, col] -= 1
            V[v10, col] += 1; V[v11, col] += 1
            W[w00, col] += 1
            col += 1

        def add_schoolbook_2x1(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            u00 = a_r * m + a_c
            u01 = a_r * m + a_c + 1
            u10 = (a_r + 1) * m + a_c
            u11 = (a_r + 1) * m + a_c + 1
            
            v0 = b_r * p + b_c
            v1 = (b_r + 1) * p + b_c
            
            w0 = c_c * n + c_r
            w1 = c_c * n + c_r + 1
            
            U[u00, col] = 1; V[v0, col] = 1; W[w0, col] = 1
            col += 1
            U[u01, col] = 1; V[v1, col] = 1; W[w0, col] = 1
            col += 1
            U[u10, col] = 1; V[v0, col] = 1; W[w1, col] = 1
            col += 1
            U[u11, col] = 1; V[v1, col] = 1; W[w1, col] = 1
            col += 1

        add_strassen(0, 0, 0, 0, 0, 0)
        add_strassen(0, 2, 2, 0, 0, 0)
        add_schoolbook_2x1(0, 0, 0, 2, 0, 2)
        add_schoolbook_2x1(0, 2, 2, 2, 0, 2)
        add_strassen(0, 0, 0, 3, 0, 3)
        add_strassen(0, 2, 2, 3, 0, 3)
        
        return U, V, W
    
    U_base, V_base, W_base = build_rank36()
    
    # Verify the base solution
    def verify(U, V, W, tol=1e-10):
        # Reconstruct tensor: T_reconstructed[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # Check against T
        for i in range(num_u):
            for j in range(num_v):
                for k in range(num_w):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i, r] * V[j, r] * W[k, r]
                    if abs(val - T[i, j, k]) > tol:
                        return False
        return True
    
    # Try to reduce rank by finding linear dependencies
    # The columns of [U; V; W] (stacked) span a space; if there's a dependency, we can reduce rank
    
    # Stack into a single matrix: each rank-1 term contributes to all three matrices
    # If we have rank r, we have r "gates". We can try to find if any gate is redundant.
    
    best_U, best_V, best_W = U_base.copy(), V_base.copy(), W_base.copy()
    best_rank = 36
    
    # Try removing each column and checking if the tensor is still satisfied
    # This is expensive but let's try a few
    
    for attempt in range(min(35, best_rank)):
        if time.time() > deadline:
            break
        
        # Try to find a column that can be eliminated
        # For each column r, check if the tensor contribution can be represented by others
        # This is complex; instead, try random perturbations and rank minimization
        
        # Alternative: use the fact that we can add zero columns
        # Try to find a linear combination of columns that gives zero tensor contribution
        
        # Simpler approach: try all pairs of columns to see if one is in the span of others
        # in terms of tensor contribution
        
        # Actually, let's try a different decomposition strategy
        # What if we use a 3×3 Strassen-like for some blocks?
        
        # Let me try: split B into 4×2 + 4×2 + 4×1
        # For the 4×1 part, use a different approach
        # C[:,4] = sum_{j=0}^{3} A[:,j] * B[j,4]
        # This is 8 scalar multiplications, rank 8
        
        # For C[:,0:2] and C[:,2:4], use Strassen as before: 4*7 = 28
        # Total: 28 + 8 = 36. Same.
        
        # What about: split into 4×1 + 4×1 + 4×1 + 4×1 + 4×1
        # Each 4×1 column: rank 8. Total: 5*8 = 40. Worse.
        
        # Split into 4×4 + 4×1:
        # C[:,0:4]: A(2×4) * B(4×4). Split A into A1(2×2), A2(2×2).
        #   C1 = A1*B11(2×2) + A2*B21(2×2) where B11 is B[0:2,0:4], B21 is B[2:4,0:4]
        #   But B11 is 2×4, not 2×2. Need to split further.
        #   C1 = A1*B11a(2×2) + A1*B11b(2×2) + A2*B21a(2×2) + A2*B21b(2×2)
        #   = 4 * Strassen(2×2×2) = 4*7 = 28
        # C[:,4]: rank 8
        # Total: 28 + 8 = 36. Same again.
        
        # It seems 36 is the natural bound for this block decomposition.
        # Let's try to verify if a lower rank is possible using numerical methods.
        
        # Use iterative rank minimization: start with rank 35, try to fit
        # This is NP-hard in general, but for small tensors we can try
        
        target_rank = 35
        if time.time() > deadline - 20:
            break
        
        # Random initialization for rank target_rank
        np.random.seed(42 + attempt)
        U_try = np.random.randn(num_u, target_rank) * 0.1
        V_try = np.random.randn(num_v, target_rank) * 0.1
        W_try = np.random.randn(num_w, target_rank) * 0.1
        
        # Gradient descent on the loss: ||T - sum_r U[:,r] V[:,r] W[:,r]||^2
        # This is a 3-way tensor decomposition, non-convex
        
        # Use alternating least squares (ALS)
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each i, U[i,:] = argmin ||T[i,:,:] - U[i,:] * (V*W)||_F^2
            # T[i,:,:] is a 20x10 matrix
            # U[i,:] is 1xtarget_rank, V*W is target_rank x (20*10)
            # This is a linear system
            
            for i in range(num_u):
                # Residual: T[i,:,:] - sum_r U[i,r] * outer(V[:,r], W[:,r])
                # Let M[i,:,:] = T[i,:,:]
                # We want: min_U ||M - U * (V ⊙ W)||^2 where ⊙ is Khatri-Rao
                # V_kr = [V[:,0]*W[:,0], V[:,1]*W[:,1], ...] shape (200, target_rank)
                # Reshape M to (200, 1)
                M_flat = T[i, :, :].flatten()
                # V_kr[:, r] = V[:,r] * W[:,r] element-wise, then flatten
                V_kr = np.zeros((num_v * num_w, target_rank))
                for r in range(target_rank):
                    V_kr[:, r] = (V_try[:, r] * W_try[:, r]).flatten()
                # Solve: min ||M_flat - V_kr @ U[i,:]||^2
                # U[i,:] = (V_kr^T V_kr)^{-1} V_kr^T M_flat
                try:
                    U_try[i, :] = np.linalg.lstsq(V_kr, M_flat, rcond=None)[0]
                except:
                    pass
            
            # Update V: for each j, V[j,:] = argmin ||T[:,j,:] - (U*W) @ V[j,:]||^2
            for j in range(num_v):
                M_flat = T[:, j, :].flatten()
                U_kr = np.zeros((num_u * num_w, target_rank))
                for r in range(target_rank):
                    U_kr[:, r] = (U_try[:, r] * W_try[:, r]).flatten()
                try:
                    V_try[j, :] = np.linalg.lstsq(U_kr, M_flat, rcond=None)[0]
                except:
                    pass
            
            # Update W: for each k, W[k,:] = argmin ||T[:,:,k] - (U*V) @ W[k,:]||^2
            for k in range(num_w):
                M_flat = T[:, :, k].flatten()
                UV_kr = np.zeros((num_u * num_v, target_rank))
                for r in range(target_rank):
                    UV_kr[:, r] = (U_try[:, r] * V_try[:, r]).flatten()
                try:
                    W_try[k, :] = np.linalg.lstsq(UV_kr, M_flat, rcond=None)[0]
                except:
                    pass
        
        # Check if the solution is good
        # Compute the error
        error = 0.0
        for i in range(num_u):
            for j in range(num_v):
                for k in range(num_w):
                    val = 0.0
                    for r in range(target_rank):
                        val += U_try[i, r] * V_try[j, r] * W_try[k, r]
                    error += (val - T[i, j, k]) ** 2
        error = np.sqrt(error)
        
        if error < 1e-8:
            # Found a valid rank-35 decomposition!
            # Try to round to rational numbers
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = 35
            # Try to go lower
            target_rank = 34
            if time.time() > deadline - 15:
                break
    
    # Convert to list format with integer/half-integer entries if possible
    # For the base solution, entries are already integers
    # For the ALS solution, try to round
    
    def to_list(mat):
        # Try to round to nearest 0.5
        rounded = np.round(mat * 2) / 2
        # Check if rounded is close to original
        if np.max(np.abs(rounded - mat)) < 1e-6:
            return [[float(x) for x in row] for row in rounded]
        else:
            return [[float(x) for x in row] for row in mat]
    
    if best_rank == 35:
        U_out = to_list(best_U)
        V_out = to_list(best_V)
        W_out = to_list(best_W)
    else:
        U_out = [[float(x) for x in row] for row in U_base]
        V_out = [[float(x) for x in row] for row in V_base]
        W_out = [[float(x) for x in row] for row in W_base]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
