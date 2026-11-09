import time
import numpy as np
from fractions import Fraction
import itertools

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for safety
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    def verify(U, V, W, tol=1e-8):
        """Verify decomposition matches T everywhere."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        r = U_arr.shape[1]
        # Check all entries
        for c in range(p*n):
            M = U_arr * W_arr[c, :]
            T_recon_slice = M @ V_arr.T
            if not np.allclose(T_recon_slice, T[:, :, c], atol=tol):
                return False
        return True
    
    def is_half_int_array(arr, tol=1e-6):
        """Check if all entries are close to half-integers."""
        return np.all(np.abs(arr * 2 - np.round(arr * 2)) < tol)
    
    def to_half_int_list(arr):
        """Convert to list of half-integers."""
        rounded = np.round(arr * 2) / 2
        return rounded.tolist()
    
    def try_structured_rank(r, seed=0):
        """Try to find rank-r decomposition with structured approach."""
        if r >= 60:
            return None
        
        np.random.seed(seed)
        
        # Strategy: Use the fact that for 2×m×p, we can try to find
        # decompositions where U has a specific block structure.
        # 
        # For n=2, the tensor has a special structure. Let's try:
        # U[i*m+j, c] = alpha[i,c] * beta[j,c] for some structured alpha, beta
        # This is a Kronecker product structure.
        
        # Alternative: Try random but with sparsity patterns
        # Generate sparse random matrices
        
        for attempt in range(5):
            # Different sparsity levels
            sparsity = 0.3 + 0.1 * attempt
            
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.zeros((p * n, r))
            
            # Zero out random entries
            mask_U = np.random.rand(n * m, r) < sparsity
            mask_V = np.random.rand(m * p, r) < sparsity
            U = U * mask_U
            V = V * mask_V
            
            # Solve for W
            success = True
            for k in range(p):
                for i in range(n):
                    c_idx = k * n + i
                    Y = np.zeros((m, r))
                    for j in range(m):
                        a_idx = i * m + j
                        b_idx = j * p + k
                        Y[j, :] = U[a_idx, :] * V[b_idx, :]
                    
                    rhs = np.ones(m)
                    
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(Y, rhs, rcond=None)
                    except:
                        success = False
                        break
                    
                    if not np.allclose(Y @ x, rhs, atol=1e-8):
                        success = False
                        break
                    
                    W[c_idx, :] = x
            
            if not success:
                continue
            
            # Verify full decomposition
            if verify(U, V, W):
                # Check half-integer constraint
                if is_half_int_array(U) and is_half_int_array(V) and is_half_int_array(W):
                    return {
                        "U": to_half_int_list(U),
                        "V": to_half_int_list(V),
                        "W": to_half_int_list(W)
                    }
        
        return None
    
    def try_kronecker_rank(r, seed=0):
        """Try Kronecker product structure for n=2."""
        if r >= 60:
            return None
        
        np.random.seed(seed)
        
        # For n=2, try U[i*m+j, c] = A[i, c1] * B[j, c2] where c = c1*m + c2
        # This gives r = 2 * m * something... let's think differently.
        
        # Actually, for the 2×m×p case, there's a known construction.
        # The tensor can be written using the identity:
        # (A*B)_{ik} = sum_j A_{ij} B_{jk}
        # 
        # One approach: use the "sum of products" structure.
        # For each (j,k), we have A_{ij} B_{jk} for i=0,1
        # So we need 2*m*p terms in schoolbook.
        
        # To reduce rank, we need to combine terms.
        # Let's try: for each k, combine the j terms using a small matrix.
        
        # Specifically, for fixed k, we need:
        # sum_j A_{0,j} B_{j,k} = C_{0,k}
        # sum_j A_{1,j} B_{j,k} = C_{1,k}
        # 
        # This is 2 equations per k, with m unknowns each.
        # If we use r_k terms for each k, total rank is p * r_k.
        # We need r_k such that we can represent the bilinear forms.
        
        # For m=3, p=10: schoolbook uses r_k = 3 for each k, total 30.
        # But we have n=2, so actually 2*3*10 = 60.
        
        # To reduce, we need to share terms across different k's.
        
        # Let's try a different approach: use random search with more iterations
        # and better initialization.
        
        for attempt in range(10):
            # Initialize with random Gaussian
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.zeros((p * n, r))
            
            # Solve for W
            success = True
            for k in range(p):
                for i in range(n):
                    c_idx = k * n + i
                    Y = np.zeros((m, r))
                    for j in range(m):
                        a_idx = i * m + j
                        b_idx = j * p + k
                        Y[j, :] = U[a_idx, :] * V[b_idx, :]
                    
                    rhs = np.ones(m)
                    
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(Y, rhs, rcond=None)
                    except:
                        success = False
                        break
                    
                    if not np.allclose(Y @ x, rhs, atol=1e-8):
                        success = False
                        break
                    
                    W[c_idx, :] = x
            
            if not success:
                continue
            
            # Verify
            if verify(U, V, W):
                # Round to half-integers and re-verify
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if verify(U_r, V_r, W_r, tol=1e-6):
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        
        return None
    
    def try_exhaustive_small_rank(r):
        """For very small r, try more systematic search."""
        if r > 20:
            return None
        
        # This is a simplified exhaustive search for small r
        # We'll use random restarts with more attempts
        
        for seed in range(50):
            np.random.seed(seed)
            
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.zeros((p * n, r))
            
            success = True
            for k in range(p):
                for i in range(n):
                    c_idx = k * n + i
                    Y = np.zeros((m, r))
                    for j in range(m):
                        a_idx = i * m + j
                        b_idx = j * p + k
                        Y[j, :] = U[a_idx, :] * V[b_idx, :]
                    
                    rhs = np.ones(m)
                    
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(Y, rhs, rcond=None)
                    except:
                        success = False
                        break
                    
                    if not np.allclose(Y @ x, rhs, atol=1e-8):
                        success = False
                        break
                    
                    W[c_idx, :] = x
            
            if not success:
                continue
            
            if verify(U, V, W):
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if verify(U_r, V_r, W_r, tol=1e-6):
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        
        return None
    
    # Try various ranks from high to low
    # First try structured approaches for ranks 59 down to 40
    for r in range(59, 39, -1):
        if time.time() > deadline:
            break
        
        # Try structured random search
        for seed in range(20):
            if time.time() > deadline:
                break
            result = try_structured_rank(r, seed)
            if result is not None:
                return result
        
        # Try Kronecker-style approach
        for seed in range(10):
            if time.time() > deadline:
                break
            result = try_kronecker_rank(r, seed)
            if result is not None:
                return result
    
    # For smaller ranks, try more exhaustive search
    for r in range(39, 29, -1):
        if time.time() > deadline:
            break
        
        result = try_exhaustive_small_rank(r)
        if result is not None:
            return result
    
    # Fallback: Schoolbook
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
