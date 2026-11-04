"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0  # safety margin before 150s limit
    
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    schoolbook_rank = n * m * p  # 120
    
    # Build target entries: (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i
    target_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries.add((a, b, c))
    
    # Schoolbook decomposition as fallback
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def validate_decomp(U, V, W):
        """Check if decomposition exactly reproduces the tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Check all target entries
        for (a, b, c) in target_entries:
            val = np.dot(U[a, :] * V[b, :], W[c, :])
            if abs(val - 1.0) > 1e-9:
                return False
        
        # Check off-diagonal entries
        # For each (i, j, jp, k) with j != jp, the value must be 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    if j == jp:
                        continue
                    for k in range(p):
                        a = i * m + j
                        b = jp * p + k
                        c = k * n + i
                        val = np.dot(U[a, :] * V[b, :], W[c, :])
                        if abs(val) > 1e-9:
                            return False
        return True
    
    def try_solve_for_W(U, V, R):
        """Given U (nm x R) and V (mp x R), solve for W (pn x R).
        
        The constraint is: for all i, j, jp, k:
            sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r] = delta(j, jp)
        
        This is linear in W. We can set up a linear system.
        """
        # Number of equations: n * m * m * p
        num_eqs = n * m * m * p
        # Number of variables: p * n * R
        num_vars = p * n * R
        
        # Build A matrix and b vector
        # For efficiency, we build A as a sparse-like structure
        # A[eq_idx, var_idx] where:
        #   eq_idx = ((i*m + j)*m + jp)*p + k
        #   var_idx = (k*n + i)*R + r
        #   A[eq_idx, var_idx] = U[i*m+j, r] * V[jp*p+k, r]
        
        # b[eq_idx] = 1 if j == jp else 0
        
        # Instead of building the full dense matrix (which is 1200 x R*24 for R~100),
        # let's use a smarter approach.
        
        # For each (i, k), the variable W[k*n+i, :] appears in equations
        # indexed by (i, j, jp, k) with coefficient U[i*m+j, r] * V[jp*p+k, r]
        
        # Let's group by (i, k) blocks
        # For fixed i, k:
        #   Variables: W[k*n+i, r] for r=0..R-1
        #   Equations: (i, j, jp, k) for j=0..m-1, jp=0..m-1
        #   Coefficient for var r in eq (j, jp): U[i*m+j, r] * V[jp*p+k, r]
        
        # So for each (i, k), we have a m*m x R sub-system:
        #   sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r] = delta(j,jp) for each (j,jp)
        
        # This means: (U_block_i * diag(W_k_i)) * V_block_k^T = delta matrix
        # where U_block_i is m x R, V_block_k is m x R (rows indexed by jp, cols by r)
        
        # Actually, let me think differently.
        # For fixed i, k, define:
        #   M[j, jp, r] = U[i*m+j, r] * V[jp*p+k, r]
        # Then: sum_r M[j, jp, r] * W[k*n+i, r] = delta(j, jp)
        
        # This is a linear system in W[k*n+i, :]:
        #   A_ik * W_ik = b_ik
        # where A_ik is (m*m) x R, with A_ik[j*m+jp, r] = U[i*m+j, r] * V[jp*p+k, r]
        # and b_ik[j*m+jp] = 1 if j==jp else 0
        
        # We can solve this with least squares or by finding a particular solution.
        
        W = np.zeros((pn, R))
        
        for i in range(n):
            for k in range(p):
                # Build A_ik: (m*m) x R
                A_ik = np.zeros((m * m, R))
                b_ik = np.zeros(m * m)
                
                for j in range(m):
                    for jp in range(m):
                        eq_idx = j * m + jp
                        for r in range(R):
                            A_ik[eq_idx, r] = U[i * m + j, r] * V[jp * p + k, r]
                        if j == jp:
                            b_ik[eq_idx] = 1.0
                
                # Solve A_ik * W_ik = b_ik
                # Use least squares
                if R >= m * m:
                    # Overdetermined or square - use lstsq
                    W_ik, residuals, rank_ik, sv = np.linalg.lstsq(A_ik, b_ik, rcond=None)
                else:
                    # Underdetermined - find minimum norm solution
                    # Use SVD
                    try:
                        U_svd, s, Vt = np.linalg.svd(A_ik, full_matrices=False)
                        # Minimum norm solution: V^T * (U * b / s)
                        U_b = U_svd.T @ b_ik
                        # Handle small singular values
                        mask = s > 1e-10
                        W_ik = np.zeros(R)
                        for idx in np.where(mask)[0]:
                            W_ik[idx] = (U_b[idx] / s[idx]) * Vt[idx, :]
                        # Actually this isn't right. Let me redo.
                        # Min norm solution to A x = b: x = V^T * (U^T b / s)
                        # where A = U S V^T
                        W_ik = Vt.T @ (U_b / np.where(s > 1e-10, s, 1.0))
                    except:
                        continue
                
                W[k * n + i, :] = W_ik
        
        return W
    
    def check_integer_half_integer(W, tol=1e-6):
        """Check if W entries are integer or half-integer."""
        for row in W:
            for val in row:
                # Check if close to integer
                if abs(val - round(val)) > tol:
                    # Check if close to half-integer
                    if abs(val - (round(val * 2) / 2.0)) > tol:
                        return False
        return True
    
    def snap_to_integer_half_integer(W):
        """Snap values to nearest integer or half-integer."""
        W_snapped = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                val = W[i, j]
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) <= abs(val - nearest_half):
                    W_snapped[i, j] = nearest_int
                else:
                    W_snapped[i, j] = nearest_half
        return W_snapped
    
    # Try ranks from schoolbook down
    for R in range(schoolbook_rank - 1, 50, -1):
        if time.time() > deadline - 15:
            break
        
        # Try several random initializations of U and V
        for attempt in range(20):
            if time.time() > deadline - 15:
                break
            
            random.seed(R * 100 + attempt)
            np.random.seed(R * 100 + attempt)
            
            # Initialize U and V with sparse random entries
            # Use small integers: -1, 0, 1
            density = max(0.05, 10.0 / R)  # sparser for larger R
            
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            
            # Random sparse initialization
            for r in range(R):
                # Pick a few nonzero entries for U
                num_nonzero_u = max(1, int(nm * density))
                u_indices = random.sample(range(nm), min(num_nonzero_u, nm))
                for idx in u_indices:
                    U[idx, r] = random.choice([-1, 1])
                
                num_nonzero_v = max(1, int(mp * density))
                v_indices = random.sample(range(mp), min(num_nonzero_v, mp))
                for idx in v_indices:
                    V[idx, r] = random.choice([-1, 1])
            
            # Solve for W
            W = try_solve_for_W(U, V, R)
            
            # Check if the solution works
            if validate_decomp(U, V, W):
                # Check if entries are close to integer/half-integer
                if check_integer_half_integer(W):
                    # Snap and re-verify
                    W_snapped = snap_to_integer_half_integer(W)
                    if validate_decomp(U, V, W_snapped):
                        if R < best_rank:
                            best_rank = R
                            best_U = U.copy()
                            best_V = V.copy()
                            best_W = W_snapped.copy()
                            print(f"Found rank {R} at attempt {attempt}")
                            break
                else:
                    # Try snapping anyway and verify
                    W_snapped = snap_to_integer_half_integer(W)
                    if validate_decomp(U, V, W_snapped):
                        if R < best_rank:
                            best_rank = R
                            best_U = U.copy()
                            best_V = V.copy()
                            best_W = W_snapped.copy()
                            print(f"Found rank {R} (snapped) at attempt {attempt}")
                            break
            
            # Also try with structured initialization
            if attempt % 5 == 0:
                # Use a more structured approach
                # Initialize U based on the "i" component, V based on "k" component
                U_struct = np.zeros((nm, R))
                V_struct = np.zeros((mp, R))
                
                for r in range(R):
                    # Assign each column to a random (i, k) pair
                    i_r = r % n
                    k_r = (r // n) % p
                    
                    # U: nonzero at positions where first index = i_r
                    for j in range(m):
                        if random.random() < 0.8:
                            U_struct[i_r * m + j, r] = random.choice([-1, 1])
                    
                    # V: nonzero at positions where second index = k_r  
                    for jp in range(m):
                        if random.random() < 0.8:
                            V_struct[jp * p + k_r, r] = random.choice([-1, 1])
                
                W_struct = try_solve_for_W(U_struct, V_struct, R)
                if validate_decomp(U_struct, V_struct, W_struct):
                    if check_integer_half_integer(W_struct):
                        W_snapped = snap_to_integer_half_integer(W_struct)
                        if validate_decomp(U_struct, V_struct, W_snapped):
                            if R < best_rank:
                                best_rank = R
                                best_U = U_struct.copy()
                                best_V = V_struct.copy()
                                best_W = W_snapped.copy()
                                print(f"Found rank {R} (structured) at attempt {attempt}")
                                break
                    else:
                        W_snapped = snap_to_integer_half_integer(W_struct)
                        if validate_decomp(U_struct, V_struct, W_snapped):
                            if R < best_rank:
                                best_rank = R
                                best_U = U_struct.copy()
                                best_V = V_struct.copy()
                                best_W = W_snapped.copy()
                                print(f"Found rank {R} (structured snapped) at attempt {attempt}")
                                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
