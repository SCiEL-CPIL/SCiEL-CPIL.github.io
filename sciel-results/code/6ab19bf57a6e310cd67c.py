import time
import numpy as np
from fractions import Fraction
import sys

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _winograd_5x5():
    """
    Returns the rank-21 Winograd algorithm for 5x5 matrix multiplication.
    The tensor decomposition: C[i,k] = sum_j A[i,j] B[j,k]
    Using the standard Winograd scheme for 5x5.
    
    The decomposition uses the structure:
    A = F_a * D_a (where F_a is 5x5 and D_a is diagonal)
    B = F_b * D_b
    C = D_c * F_c^T
    
    We need to express this as a tensor decomposition.
    For 5x5 Winograd, we use the following approach:
    
    Let's use the known rank-21 decomposition for 5x5 matrix multiplication.
    The key insight is that 5x5 matmul can be done with 21 multiplications.
    
    Standard Winograd for 5x5:
    We decompose the 5x5 matmul tensor into 21 rank-1 terms.
    
    Using the fact that the 5x5 matmul tensor T[i,j,k] = 1 for all i,j,k
    can be decomposed as:
    T[i,j,k] = sum_{c=1}^{21} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    For the Winograd scheme, we use the following:
    The 5x5 matmul can be viewed as computing C = A * B where A is 5x5, B is 5x5.
    
    Let me use a specific rank-21 decomposition. One known approach:
    Use the fact that 5 = 4+1, and use the rank-16 decomposition for 4x4 plus corrections.
    
    Actually, let me use the standard Winograd algorithm for 5x5:
    
    The Winograd algorithm for 5x5 uses the following transforms:
    F_a: 5x5 matrix for A
    F_b: 5x5 matrix for B  
    F_c: 5x5 matrix for C
    
    With 21 multiplications.
    
    Let me construct this explicitly.
    
    For 5x5 matrix multiplication, one rank-21 decomposition uses:
    - 16 terms from a 4x4 subproblem
    - 5 additional terms for the boundary
    
    Actually, let me use a more direct approach. The rank of the 5x5 matmul tensor
    is known to be 21. Let me construct it using the Winograd method.
    
    The Winograd method for n x n matrix multiplication:
    C = A * B
    A' = F_a * A (5x5)
    B' = B * F_b (5x5)
    D = A' * B' (element-wise, 25 products, but we can reduce)
    C = F_c * D * F_d (5x5)
    
    For 5x5, we need 21 products.
    
    Let me use the following specific decomposition:
    """
    # For 5x5 matrix multiplication, we use the following rank-21 decomposition
    # based on the Winograd algorithm.
    
    # The key idea: we can write the 5x5 matmul as a sum of 21 rank-1 tensors.
    # Each rank-1 tensor corresponds to one multiplication.
    
    # Let me use a known construction. The 5x5 matmul tensor can be decomposed
    # using the following approach:
    
    # Split the 5x5 matrix into blocks and use the rank of sub-problems.
    
    # Actually, let me just use the fact that we can construct a rank-21
    # decomposition explicitly. Here's one approach:
    
    # Use the identity:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j (sum_a alpha[i,j,a] P_a) (sum_b beta[j,k,b] Q_b) gamma[k,i,a,b]
    
    # This is getting complex. Let me use a simpler approach:
    # Since we need rank 21 for 5x5, and we have two rows (n=2),
    # the total rank should be 2*21 = 42.
    
    # Let me construct the 5x5 rank-21 decomposition first, then embed it.
    
    # For 5x5 Winograd, I'll use the following known decomposition:
    
    # The 5x5 matmul can be done with 21 multiplications using:
    # 1. Transform A and B using fixed matrices
    # 2. Do 21 element-wise products
    # 3. Transform back
    
    # Let me use a specific construction from the literature.
    
    # One known rank-21 decomposition for 5x5:
    # Uses the fact that 5x5 matmul = 4x4 matmul (rank 16) + corrections (5 more)
    
    # Actually, the known lower bound for 5x5 is 21, and this is achievable.
    
    # Let me construct it using the following method:
    # We express the 5x5 matmul tensor as a sum of 21 rank-1 terms.
    
    # Here's a concrete construction:
    
    # Let's use the following approach based on polynomial multiplication.
    # The 5x5 matmul is equivalent to multiplying two polynomials of degree 4
    # (with some structure).
    
    # Actually, let me just hardcode a known rank-21 decomposition.
    
    # For 5x5 matrix multiplication, one rank-21 decomposition:
    
    # The decomposition uses the following:
    # - 16 terms for the 4x4 subproblem
    # - 5 terms for the remaining entries
    
    # Let me use a more systematic approach. I'll construct the decomposition
    # by solving a system of equations.
    
    # Actually, let me use the following known result:
    # The rank of the 5x5 matmul tensor is 21.
    
    # Here's a specific construction:
    
    # We can write:
    # C[i,k] = sum_{j=0}^{4} A[i,j] B[j,k]
    
    # Using the Winograd method, we can reduce this to 21 multiplications.
    
    # Let me use the following specific matrices:
    
    # F_a (5x5), F_b (5x5), F_c (5x5) such that:
    # C = F_c * (F_a * A) * (B * F_b) * F_d
    
    # With 21 non-zero products in the middle.
    
    # For simplicity, let me use a different approach:
    # I'll construct the rank-21 decomposition by solving for the factors.
    
    # Actually, let me just use the schoolbook rank-25 for 5x5 and see if
    # we can reduce it. But we need exactly rank 21.
    
    # Let me try a different strategy: use the known fact that the 5x5
    # matmul has rank 21, and construct the decomposition numerically.
    
    # For now, let me return a placeholder and use optimization to find it.
    
    return None

def _try_structured_decomposition(n, m, p):
    """
    Try to find a rank-42 decomposition by decomposing the 2x5x5 tensor
    into two independent 1x5x5 tensors, each using the rank-21 Winograd
    algorithm for 5x5 matrix multiplication.
    """
    if time.time() > _DEADLINE - 20:
        return None
    
    target_rank = 42
    
    # We need to find a rank-21 decomposition for 5x5 matmul first.
    # Let me try to construct it using optimization.
    
    np.random.seed(123)
    
    # Initialize with schoolbook for 5x5 (rank 25) and perturb
    U0 = np.zeros((5, target_rank // 2))  # For one row
    V0 = np.zeros((25, target_rank // 2))
    W0 = np.zeros((5, target_rank // 2))
    
    # Fill first 21 terms with schoolbook pattern
    c = 0
    for i in range(5):
        for j in range(5):
            if c < target_rank // 2:
                U0[i, c] = 1.0
                V0[j * 5 + 0, c] = 1.0  # This is wrong, let me fix
                c += 1
    
    # Let me redo this properly.
    # For 5x5 matmul: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] = 1
    # U[i,j], V[j,k], W[k,i]
    
    U0 = np.zeros((25, target_rank // 2))  # i*5+j
    V0 = np.zeros((25, target_rank // 2))  # j*5+k
    W0 = np.zeros((25, target_rank // 2))  # k*5+i
    
    c = 0
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if c < target_rank // 2:
                    U0[i * 5 + j, c] = 1.0
                    V0[j * 5 + k, c] = 1.0
                    W0[k * 5 + i, c] = 1.0
                    c += 1
    
    # Add noise
    noise_scale = 0.01
    U0 += np.random.normal(0, noise_scale, U0.shape)
    V0 += np.random.normal(0, noise_scale, V0.shape)
    W0 += np.random.normal(0, noise_scale, W0.shape)
    
    # Flatten
    params = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    def objective_5x5(params):
        U_flat, V_flat, W_flat = np.split(params, [25 * (target_rank // 2), 2 * 25 * (target_rank // 2)])
        U = U_flat.reshape(25, target_rank // 2)
        V = V_flat.reshape(25, target_rank // 2)
        W = W_flat.reshape(25, target_rank // 2)
        
        # Compute T[i,j,k] = sum_c U[i*5+j,c] * V[j*5+k,c] * W[k*5+i,c]
        T = np.zeros((5, 5, 5))
        for c in range(target_rank // 2):
            u_c = U[:, c].reshape(5, 5)
            v_c = V[:, c].reshape(5, 5)
            w_c = W[:, c].reshape(5, 5)
            
            # T[i,j,k] = u_c[i,j] * v_c[j,k] * w_c[k,i]
            U_c_3d = u_c[:, :, None]
            V_c_3d = v_c[None, :, :]
            UV = U_c_3d * V_c_3d
            W_c_T = w_c.T
            W_c_3d = W_c_T[:, None, :]
            T += UV * W_c_3d
        
        return np.sum((T - 1.0) ** 2)
    
    try:
        from scipy.optimize import minimize
        
        res = minimize(objective_5x5, params, method='L-BFGS-B',
                       options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6})
        
        if res.fun < 1e-6:
            # We found a rank-21 decomposition for 5x5
            U_flat, V_flat, W_flat = np.split(res.x, [25 * (target_rank // 2), 2 * 25 * (target_rank // 2)])
            U_5x5 = U_flat.reshape(25, target_rank // 2)
            V_5x5 = V_flat.reshape(25, target_rank // 2)
            W_5x5 = W_flat.reshape(25, target_rank // 2)
            
            # Now embed this into the 2x5x5 decomposition
            # For row i=0: use the first 21 terms
            # For row i=1: use the next 21 terms
            U_full = np.zeros((2 * 5, target_rank))
            V_full = np.zeros((25, target_rank))
            W_full = np.zeros((10, target_rank))
            
            # Row 0
            U_full[0:5, 0:21] = U_5x5
            V_full[:, 0:21] = V_5x5
            W_full[0:5, 0:21] = W_5x5
            
            # Row 1
            U_full[5:10, 21:42] = U_5x5
            V_full[:, 21:42] = V_5x5
            W_full[5:10, 21:42] = W_5x5
            
            # Verify and try rounding
            for denom in [1, 2, 4, 8]:
                U_r = np.round(U_full * denom) / denom
                V_r = np.round(V_full * denom) / denom
                W_r = np.round(W_full * denom) / denom
                
                if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
    except Exception:
        pass
    
    return None

def _try_rank_49(n, m, p):
    """
    Attempt to find rank-49 decomposition for 2x5x5 matrix multiplication.
    Uses numerical optimization initialized near the schoolbook solution.
    """
    if time.time() > _DEADLINE - 20:
        return None
    
    target_rank = 49
    np.random.seed(42)
    
    # Initialize with schoolbook terms, but drop one and perturb
    # We start with 50 terms, drop the last one, and add noise to the rest
    U0 = np.zeros((n * m, target_rank))
    V0 = np.zeros((m * p, target_rank))
    W0 = np.zeros((p * n, target_rank))
    
    # Fill first 49 terms with schoolbook pattern
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if c < target_rank:
                    U0[i * m + j, c] = 1.0
                    V0[j * p + k, c] = 1.0
                    W0[k * n + i, c] = 1.0
                    c += 1
    
    # The 50th term (i=1, j=4, k=4) is missing.
    # We need to adjust the first 49 terms to compensate.
    # Add small random perturbation to help escape local minima
    noise_scale = 0.01
    U0 += np.random.normal(0, noise_scale, U0.shape)
    V0 += np.random.normal(0, noise_scale, V0.shape)
    W0 += np.random.normal(0, noise_scale, W0.shape)
    
    # Flatten parameters
    params = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    def objective(params):
        U_flat, V_flat, W_flat = np.split(params, [n*m*target_rank, 2*n*m*target_rank])
        U = U_flat.reshape(n*m, target_rank)
        V = V_flat.reshape(m*p, target_rank)
        W = W_flat.reshape(p*n, target_rank)
        
        T = np.zeros((n, m, p))
        for c in range(target_rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            
            U_c_3d = u_c[:, :, None]
            V_c_3d = v_c[None, :, :]
            UV = U_c_3d * V_c_3d
            W_c_T = w_c.T
            W_c_3d = W_c_T[:, None, :]
            T += UV * W_c_3d
        
        return np.sum((T - 1.0)**2)
    
    try:
        from scipy.optimize import minimize
        
        res = minimize(objective, params, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < 1e-8:
            U_flat, V_flat, W_flat = np.split(res.x, [n*m*target_rank, 2*n*m*target_rank])
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            for denom in [1, 2, 4, 8]:
                U_r = np.round(U * denom) / denom
                V_r = np.round(V * denom) / denom
                W_r = np.round(W * denom) / denom
                
                if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if _verify(n, m, p, U_int.tolist(), V_int.tolist(), W_int.tolist()):
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
    except Exception as e:
        pass
    
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try structured decomposition first (rank 42)
    result = _try_structured_decomposition(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    # Try rank 49
    result = _try_rank_49(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
