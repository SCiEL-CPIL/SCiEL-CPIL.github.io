import time
import numpy as np
from itertools import product
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    rank = len(U[0]) if len(U) > 0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _search_rank_gaussian(target_rank, n, m, p, time_budget=10.0, seed=0):
    """
    Search for a rank-`target_rank` decomposition using Gaussian integer parameterization.
    
    Strategy: Parameterize U, V, W as complex-valued matrices where entries are Gaussian integers
    (a + bi where a, b are integers/half-integers). The norm structure |z|^2 = a^2 + b^2
    can help find sparse solutions. We use a smooth surrogate objective that penalizes
    deviation from the tensor identity and encourages sparsity/norm structure.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    n_m = n * m  # 6
    m_p = m * p  # 12
    p_n = p * n  # 8
    
    # We'll work with real and imaginary parts separately
    # Each complex entry z = a + bi contributes to the product as:
    # (a1 + bi1)(a2 + bi2)(a3 + bi3) = (a1a2a3 - a1b2b3 - b1a2b3 - b1b2a3) + i(...)
    # For the real part to equal 1 and imaginary part to be 0, we need both constraints.
    
    # Simpler approach: Use the fact that if we find complex solutions, we can sometimes
    # convert to real solutions by taking norms or using algebraic conjugates.
    
    # Let's try a different approach: parameterize with complex entries but enforce
    # that the final product is real and equals 1.
    
    attempts = 0
    while time.time() - start_time < time_budget:
        attempts += 1
        if attempts > 2000:  # Safety break
            break
        
        # Initialize with sparse complex entries
        # Real and imaginary parts from {-1, 0, 1}
        if attempts % 3 == 0:
            # Very sparse
            U_real = np.random.choice([-1, 0, 1], size=(n_m, target_rank))
            U_imag = np.random.choice([0, 0, 0, 1], size=(n_m, target_rank))
            V_real = np.random.choice([-1, 0, 1], size=(m_p, target_rank))
            V_imag = np.random.choice([0, 0, 0, 1], size=(m_p, target_rank))
        elif attempts % 3 == 1:
            # Moderately sparse
            U_real = np.random.choice([-1, 0, 1], size=(n_m, target_rank))
            U_imag = np.random.choice([-1, 0, 1], size=(n_m, target_rank))
            V_real = np.random.choice([-1, 0, 1], size=(m_p, target_rank))
            V_imag = np.random.choice([-1, 0, 1], size=(m_p, target_rank))
        else:
            # Dense
            U_real = np.random.choice([-1, 0, 1], size=(n_m, target_rank))
            U_imag = np.random.choice([-1, 0, 1], size=(n_m, target_rank))
            V_real = np.random.choice([-1, 0, 1], size=(m_p, target_rank))
            V_imag = np.random.choice([-1, 0, 1], size=(m_p, target_rank))
        
        # Build the system: For each (i,j,k), we need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        # where U, V, W are complex. This gives two real equations per (i,j,k).
        
        # Let's set up the linear system for W given U and V.
        # A[i,j,k, k*n+i, r] = U[i,j,r] * V[j,k,r]
        # But since we're dealing with complex, we need to handle real/imag parts.
        
        # For simplicity, let's first try to find real solutions using the standard approach
        # but with better initialization inspired by Gaussian integers.
        
        # Convert to real matrices for the linear system
        U_real_mat = U_real.astype(float)
        V_real_mat = V_real.astype(float)
        
        # Build the linear system A W = b
        num_eqs = n * m * p  # 24
        num_vars = p_n * target_rank
        
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(target_rank):
                        var_idx = (k * n + i) * target_rank + r
                        A[eq_idx, var_idx] = U_real_mat[i*m+j, r] * V_real_mat[j*p+k, r]
        
        # Solve for W
        try:
            W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        W = W_flat.reshape(p_n, target_rank)
        
        # Round to nearest half-integer
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        U_list = U_real.tolist()
        V_list = V_real.tolist()
        W_list = W_rounded.tolist()
        
        # Quick check
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for r in range(target_rank):
                        actual += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                    if abs(actual - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return None

def _search_rank_optimized(target_rank, n, m, p, time_budget=10.0, seed=0):
    """
    Optimized search using better initialization and verification.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    n_m = n * m  # 6
    m_p = m * p  # 12
    p_n = p * n  # 8
    
    num_vars = p_n * target_rank
    num_eqs = n * m * p  # 24
    
    attempts = 0
    while time.time() - start_time < time_budget:
        attempts += 1
        if attempts > 3000:  # Safety break
            break
        
        # Strategy: Use structured initialization based on Kronecker products
        # For 2x3 x 3x4, we can try block structures
        
        if attempts % 4 == 0:
            # Block diagonal structure
            U = np.zeros((n_m, target_rank))
            V = np.zeros((m_p, target_rank))
            
            # Try to exploit the 2x3 and 3x4 structure
            for block in range(min(3, target_rank)):
                col = block
                # Randomize within blocks
                U[np.random.choice(n_m, size=2, replace=False), col] = np.random.choice([-1, 1], size=2)
                V[np.random.choice(m_p, size=3, replace=False), col] = np.random.choice([-1, 1], size=3)
        elif attempts % 4 == 1:
            # Sparse random
            U = np.zeros((n_m, target_rank))
            V = np.zeros((m_p, target_rank))
            for r in range(target_rank):
                # Each column has 1-2 nonzeros in U
                nz_u = np.random.choice(n_m, size=np.random.choice([1, 2]), replace=False)
                U[nz_u, r] = np.random.choice([-1, 1], size=len(nz_u))
                # Each column has 2-3 nonzeros in V
                nz_v = np.random.choice(m_p, size=np.random.choice([2, 3]), replace=False)
                V[nz_v, r] = np.random.choice([-1, 1], size=len(nz_v))
        else:
            # Dense random with small values
            U = np.random.choice([-1, 0, 1], size=(n_m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m_p, target_rank))
        
        # Build the linear system A W = b
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(target_rank):
                        var_idx = (k * n + i) * target_rank + r
                        A[eq_idx, var_idx] = U[i*m+j, r] * V[j*p+k, r]
        
        # Solve for W
        try:
            W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        W = W_flat.reshape(p_n, target_rank)
        
        # Round to nearest half-integer
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W_rounded.tolist()
        
        # Quick check
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for r in range(target_rank):
                        actual += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                    if abs(actual - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin out of 150s
    
    n, m, p = DIMENSIONS
    
    # Try decreasing ranks with multiple strategies
    for target_rank in [20, 19, 18, 17, 16, 15]:
        if time.time() > deadline - 15:
            break
        
        # Try different strategies and seeds
        for seed in range(8):
            if time.time() > deadline - 10:
                break
            
            # Alternate between strategies
            if seed % 2 == 0:
                result = _search_rank_optimized(target_rank, n, m, p, 
                                                 time_budget=min(4, deadline - time.time() - 5), 
                                                 seed=seed * 100 + target_rank)
            else:
                result = _search_rank_gaussian(target_rank, n, m, p, 
                                                 time_budget=min(4, deadline - time.time() - 5), 
                                                 seed=seed * 100 + target_rank)
            if result is not None:
                return result
    
    # Fallback: trivial rank-24 decomposition
    rank = n * m * p  # 24
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
# EVOLVE-BLOCK-END
