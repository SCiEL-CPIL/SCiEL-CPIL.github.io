"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product
from fractions import Fraction
from collections import defaultdict

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    tensor = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute lower bounds
    T0 = tensor.reshape(n*m, -1)
    rank0 = np.linalg.matrix_rank(T0)
    T1 = tensor.transpose(1,0,2).reshape(m*p, -1)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = tensor.transpose(2,0,1).reshape(p*n, -1)
    rank2 = np.linalg.matrix_rank(T2)
    lower_bound = max(rank0, rank1, rank2)
    
    # Trivial solution with rank = n*m*p
    rank_trivial = n * m * p
    U_trivial = np.zeros((n*m, rank_trivial))
    V_trivial = np.zeros((m*p, rank_trivial))
    W_trivial = np.zeros((p*n, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j, col] = 1
                V_trivial[j*p+k, col] = 1
                W_trivial[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
    best_rank = rank_trivial
    
    def check_time():
        return time.time() < deadline - 5
    
    def verify_exact(U, V, W, rank):
        """Verify exact decomposition using Fraction arithmetic."""
        # Convert to fractions for exact verification
        U_frac = [[Fraction(U[a, r]).limit_denominator(1000000) for r in range(rank)] for a in range(n*m)]
        V_frac = [[Fraction(V[b, r]).limit_denominator(1000000) for r in range(rank)] for b in range(m*p)]
        W_frac = [[Fraction(W[c, r]).limit_denominator(1000000) for r in range(rank)] for c in range(p*n)]
        
        # Check all tensor entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_frac[a][r] * V_frac[b][r] * W_frac[c][r]
                    if s != 1:
                        return False
                    # Check off-diagonal entries are zero
                    # For entries where the indices don't match the pattern
        # Also verify that non-pattern entries are zero
        # Pattern entry: T[i*m+j, j*p+k, k*n+i] = 1
        # All other entries should be 0
        # But we only need to check the pattern entries equal 1 and verify the reconstruction
        # Let's do a full check via einsum with fractions for a subset
        return True
    
    def verify_exact_fast(U, V, W, rank):
        """Faster verification: check pattern entries = 1 and use numpy for zero check."""
        # Check pattern entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check a sample of off-pattern entries
        np.random.seed(42)
        num_checks = min(1000, n*m*m*p*p*n - n*m*p)
        for _ in range(num_checks):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            # Check if this is a pattern entry
            is_pattern = False
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if a == i*m+j and b == j*p+k and c == k*n+i:
                            is_pattern = True
                            break
                    if is_pattern:
                        break
                if is_pattern:
                    break
            if not is_pattern:
                s = 0.0
                for r in range(rank):
                    s += U[a, r] * V[b, r] * W[c, r]
                if abs(s) > 1e-9:
                    return False
        return True
    
    # Strategy: Use structured decomposition exploiting n=2
    # Key idea: For each k, the contribution involves both i=0 and i=1
    # Try to find minimal rank by solving a system of equations
    
    def try_structured_rank(trial_rank, max_attempts=50):
        """Try to find a decomposition of given rank using structured patterns."""
        if not check_time():
            return None
        
        # Approach: Parameterize U with structure
        # U[i*m+j, r] for i in {0,1}, j in {0,1,2}
        # Try patterns where U entries are in {-1, 0, 1}
        
        # For each rank, try different U structures
        # Structure 1: Each row of U has exactly 1 non-zero (sparse)
        # Structure 2: Each row of U has 2 non-zeros
        # Structure 3: Random sparse patterns
        
        for attempt in range(max_attempts):
            if not check_time():
                break
            
            np.random.seed(10000 + trial_rank * 100 + attempt)
            
            # Generate structured U
            U = np.zeros((n*m, trial_rank))
            
            # Each row (i,j) gets 1-3 non-zero entries
            for i in range(n):
                for j in range(m):
                    a = i*m + j
                    num_nz = np.random.randint(1, 4)  # 1 to 3 non-zeros
                    if num_nz > trial_rank:
                        num_nz = trial_rank
                    cols = np.random.choice(trial_rank, size=num_nz, replace=False)
                    for c in cols:
                        U[a, c] = np.random.choice([-1, 1])
            
            # Check U has no zero rows
            if np.any(np.all(U == 0, axis=1)):
                continue
            
            # For given U, solve for V and W
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For each c: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # This is: (U * V^T)_{a,b} * W[c,r] ... not quite linear in W alone
            
            # Better: For fixed U and V, W is determined by:
            # For each c: sum_r W[c,r] * (sum_a U[a,r] * V[b,r] * delta_{T[a,b,c],1}) ... 
            # Actually, let's use the fact that for each c, we have:
            # sum_r W[c,r] * A_r[c] = T[:, :, c] (as a vector)
            # where A_r[c] involves U and V
            
            # Alternative: Use the matrix form
            # Let M_r = U[:, r] * V[:, r]^T (outer product)
            # Then T[:, :, c] = sum_r W[c, r] * M_r[:, :, ...]
            # Wait, T is 3D. Let me think differently.
            
            # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed c: T[:, :, c] = sum_r W[c,r] (U[:,r] V[:,r]^T)
            # So T[:, :, c] is in the span of {U[:,r] V[:,r]^T}
            
            # Given U and V, for each c we need:
            # T[:, :, c] = sum_r W[c,r] * (U[:,r] ⊗ V[:,r])
            # This is a linear system in W[:, c]
            
            # Build the system:
            # For each (a,b): T[a,b,c] = sum_r W[c,r] * U[a,r] * V[b,r]
            # So: T[:, :, c] = (U * V^T)_{a,b} weighted by W[c,r]
            # Actually: T[a,b,c] = sum_r (U[a,r] * V[b,r]) * W[c,r]
            # Let B[a,b,r] = U[a,r] * V[b,r]
            # Then T[a,b,c] = sum_r B[a,b,r] * W[c,r]
            # Reshape: T.reshape(n*m*m*p, p*n) = B.reshape(n*m*m*p, rank) * W.T
            # Wait: T is (n*m, m*p, p*n), so T.reshape(n*m*m*p, p*n) has rows (a,b) and cols c
            # B is (n*m, m*p, rank), so B.reshape(n*m*m*p, rank) has rows (a,b) and cols r
            # We need: T_flat = B_flat @ W.T
            # So W.T = B_flat^+ @ T_flat (pseudoinverse)
            
            # But we don't know V yet. Let's try ALS.
            
            # ALS approach:
            # Initialize V randomly
            V = np.random.randn(m*p, trial_rank)
            W = np.random.randn(p*n, trial_rank)
            
            UtU = U.T @ U
            
            for iteration in range(100):
                if not check_time():
                    return None
                
                # Update W: For each c, W[c,:] solves:
                # T[:, :, c] = sum_r W[c,r] * (U[:,r] * V[:,r])
                # In matrix form: T[:,:,c].reshape(-1) = (U*V).reshape(-1, rank) @ W[c,:]
                # where (U*V)[a,b,r] = U[a,r]*V[b,r]
                
                UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (n*m, m*p, rank)
                UV_flat = UV.reshape(n*m*m*p, trial_rank)
                
                T_flat = tensor.reshape(n*m*m*p, p*n)
                
                # T_flat = UV_flat @ W.T
                # So W.T = UV_flat^+ @ T_flat
                try:
                    W_new = np.linalg.lstsq(UV_flat, T_flat, rcond=None)[0].T
                except:
                    W_new = W
                
                # Update V: Similarly
                UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]  # (n*m, p*n, rank)
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed b: T[:,b,:].reshape(-1) = (U*W).reshape(-1, rank) @ V[b,:]
                # where (U*W)[a,c,r] = U[a,r]*W[c,r]
                
                UW_flat = UW.reshape(n*m*p*n, trial_rank)
                T_flat2 = tensor.transpose(0, 2, 1).reshape(n*m*p*n, m*p)  # (a,c) -> b
                
                try:
                    V_new = np.linalg.lstsq(UW_flat, T_flat2, rcond=None)[0].T
                except:
                    V_new = V
                
                # Check convergence
                recon = np.einsum('ar,br,cr->abc', U, V_new, W_new)
                error = np.linalg.norm(recon - tensor)
                
                if error < 1e-8:
                    # Verify exact
                    if verify_exact_fast(U, V_new, W_new, trial_rank):
                        return U.copy(), V_new.copy(), W_new.copy()
                    # Try rounding
                    for factor in [1, 2, 3, 4, 6, 8, 12, 16, 24]:
                        V_r = np.round(V_new * factor) / factor
                        W_r = np.round(W_new * factor) / factor
                        recon_r = np.einsum('ar,br,cr->abc', U, V_r, W_r)
                        if np.linalg.norm(recon_r - tensor) < 1e-8:
                            if verify_exact_fast(U, V_r, W_r, trial_rank):
                                return U.copy(), V_r.copy(), W_r.copy()
                    break
                
                V = V_new
                W = W_new
            
            # After ALS, try rounding
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            if np.linalg.norm(recon - tensor) < 1e-4:
                for factor in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]:
                    V_r = np.round(V * factor) / factor
                    W_r = np.round(W * factor) / factor
                    recon_r = np.einsum('ar,br,cr->abc', U, V_r, W_r)
                    if np.linalg.norm(recon_r - tensor) < 1e-8:
                        if verify_exact_fast(U, V_r, W_r, trial_rank):
                            return U.copy(), V_r.copy(), W_r.copy()
        
        return None
    
    # Try ranks from lower_bound upward
    for trial_rank in range(lower_bound, min(rank_trivial, lower_bound + 30)):
        if not check_time():
            break
        
        result = try_structured_rank(trial_rank, max_attempts=30)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = trial_rank
            break
    
    # If no better found, also try a more systematic approach for low ranks
    if best_rank == rank_trivial and check_time():
        # Try rank = lower_bound with more careful initialization
        # Use the fact that the tensor has a specific structure
        # For n=2, we can try to separate the i=0 and i=1 contributions
        
        for trial_rank in range(lower_bound, lower_bound + 5):
            if not check_time():
                break
            
            # More structured U: try all combinations of small patterns
            # U has 6 rows, trial_rank columns
            # Try U where each row has exactly 2 non-zeros from a small set
            
            num_attempts = min(100, int((deadline - time.time()) / 2))
            for attempt in range(num_attempts):
                if not check_time():
                    break
                
                np.random.seed(50000 + trial_rank * 1000 + attempt)
                
                U = np.zeros((n*m, trial_rank))
                for a in range(n*m):
                    num_nz = np.random.randint(2, 4)
                    if num_nz > trial_rank:
                        num_nz = trial_rank
                    cols = np.random.choice(trial_rank, size=num_nz, replace=False)
                    for c in cols:
                        U[a, c] = np.random.choice([-1, 1])
                
                if np.any(np.all(U == 0, axis=1)):
                    continue
                
                # ALS
                V = np.random.randn(m*p, trial_rank)
                W = np.random.randn(p*n, trial_rank)
                
                for iteration in range(50):
                    if not check_time():
                        break
                    
                    UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
                    UV_flat = UV.reshape(n*m*m*p, trial_rank)
                    T_flat = tensor.reshape(n*m*m*p, p*n)
                    
                    try:
                        W = np.linalg.lstsq(UV_flat, T_flat, rcond=None)[0].T
                    except:
                        pass
                    
                    UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]
                    UW_flat = UW.reshape(n*m*p*n, trial_rank)
                    T_flat2 = tensor.transpose(0, 2, 1).reshape(n*m*p*n, m*p)
                    
                    try:
                        V = np.linalg.lstsq(UW_flat, T_flat2, rcond=None)[0].T
                    except:
                        pass
                    
                    recon = np.einsum('ar,br,cr->abc', U, V, W)
                    error = np.linalg.norm(recon - tensor)
                    
                    if error < 1e-8:
                        if verify_exact_fast(U, V, W, trial_rank):
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            best_rank = trial_rank
                            break
                        for factor in [1, 2, 3, 4, 6, 8, 12, 16, 24]:
                            V_r = np.round(V * factor) / factor
                            W_r = np.round(W * factor) / factor
                            recon_r = np.einsum('ar,br,cr->abc', U, V_r, W_r)
                            if np.linalg.norm(recon_r - tensor) < 1e-8:
                                if verify_exact_fast(U, V_r, W_r, trial_rank):
                                    best_U, best_V, best_W = U.copy(), V_r.copy(), W_r.copy()
                                    best_rank = trial_rank
                                    break
                        break
                
                if best_rank == trial_rank:
                    break
            
            if best_rank == trial_rank:
                break
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
