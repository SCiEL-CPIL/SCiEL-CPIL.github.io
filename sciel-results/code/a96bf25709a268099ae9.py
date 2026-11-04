import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Check shapes
    if U.shape != (n*m, U.shape[1]):
        return False
    if V.shape != (m*p, V.shape[1]):
        return False
    if W.shape != (p*n, W.shape[1]):
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    # For each output element C[i,k], check that:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This must hold for all i, j, k
    
    for i in range(n):
        for k in range(p):
            # Build the left-hand side: sum_j A[i,j]*B[j,k]
            lhs = np.zeros((n*m, m*p))
            for j in range(m):
                lhs[i*m+j, j*p+k] = 1.0
            
            # Build the right-hand side: sum_r U[:,r] * V[:,r] * W[k*n+i, r]
            rhs = np.zeros((n*m, m*p))
            for r in range(rank):
                rhs += np.outer(U[:, r], V[:, r]) * W[k*n+i, r]
            
            # Check if lhs == rhs
            if not np.allclose(lhs, rhs, atol=1e-10):
                return False
    
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            if val != 0:
                U[i * m + j][r] = val
        for (j, k), val in v_dict.items():
            if val != 0:
                V[j * p + k][r] = val
        for (k, i), val in w_dict.items():
            if val != 0:
                W[k * n + i][r] = val
    
    return {"U": U, "V": V, "W": W}

def _try_find_decomposition(n, m, p, target_rank, deadline, seed=42):
    """
    Try to find a rank-target_rank decomposition using structured search.
    
    Strategy: Use the fact that the multiplication tensor has structure.
    For each (i,j), the contribution to output C[i,k] for all k must satisfy:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
    
    We'll try to find solutions by fixing some variables and solving.
    """
    rng = np.random.RandomState(seed)
    
    # The key insight: for a given (i,j), we need the vector 
    # u_ij = U[i*m+j, :] to satisfy:
    # u_ij . (V[j*p+k, :] * W[k*n+i, :]) = 1 for all k
    
    # Let's try a structured approach where we assume certain patterns.
    
    # Approach: Try to find a decomposition where U, V, W have entries from 
    # a small set like {-1, 0, 1} or {-1, 0, 0.5, 1}.
    
    # First, let's try to see if we can do better than schoolbook by
    # exploiting the linearity in each mode.
    
    # For fixed i, define M_i as the matrix with entries M_i[j, k] = 1
    # (this is what we're computing for row i of C).
    # We need: M_i = sum_r W[k*n+i, r] * outer(U[i*m+0:m, r], V[0:m*p, r])[j, k]
    # Actually more precisely:
    # For each i: the matrix C_i (which is row i of C, shape 1 x p) is:
    # C_i[k] = sum_j A[i,j] * B[j,k]
    # In tensor form: for fixed i, we have a bilinear form in A[i,:] and B[:,k]
    
    # Let me try a direct optimization approach using alternating least squares
    # or a similar method.
    
    # Initialize with schoolbook-like structure
    U = np.zeros((n*m, target_rank))
    V = np.zeros((m*p, target_rank))
    W = np.zeros((p*n, target_rank))
    
    # Try random initialization
    for attempt in range(50):
        if time.time() > deadline:
            return None
        
        # Random initialization with some structure
        U_init = rng.randn(n*m, target_rank) * 0.5
        V_init = rng.randn(m*p, target_rank) * 0.5
        W_init = rng.randn(p*n, target_rank) * 0.5
        
        # Normalize to avoid overflow
        for r in range(target_rank):
            norm_u = np.linalg.norm(U_init[:, r])
            norm_v = np.linalg.norm(V_init[:, r])
            norm_w = np.linalg.norm(W_init[:, r])
            if norm_u > 0 and norm_v > 0 and norm_w > 0:
                scale = 1.0 / (norm_u * norm_v * norm_w) ** (1/3)
                U_init[:, r] *= scale
                V_init[:, r] *= scale
                W_init[:, r] *= scale
        
        U_curr = U_init.copy()
        V_curr = V_init.copy()
        W_curr = W_init.copy()
        
        # Alternating minimization: fix two, optimize the third
        # The objective is: minimize sum_{i,j,k} (T[i*m+j, j*p+k, k*n+i] - 
        #   sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
        # Plus penalty for non-zero entries where T=0
        
        # Actually, let's just check if we can find an exact solution by
        # trying to solve the system of equations.
        
        # For each (i,j,k), we have:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
        
        # This is a system of n*m*p equations with n*m*r + m*p*r + p*n*r unknowns.
        # For r < n*m*p, this is underdetermined, so solutions may exist.
        
        # Let's try a different approach: use the fact that we can absorb
        # scaling between U and V.
        
        # Try structured initialization based on Kronecker products
        # For the 3x3x5 case, let's try to find patterns.
        
        # Actually, let me try a simpler approach: just return None and 
        # let the caller try other strategies.
        pass
    
    return None

def _try_algebraic_search(n, m, p, target_rank, deadline):
    """
    Try to find a decomposition by solving the algebraic system.
    
    We have variables U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for r=0..target_rank-1.
    Constraints: for all (i,j,k):
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is valid
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 0 otherwise
    
    This is a system of polynomial equations. For small ranks, we can try
    to solve it using numerical methods.
    """
    # Total number of constraints: n*m*p (for the "1" constraints)
    # Plus many more for the "0" constraints, but those are automatically
    # satisfied if we only care about the diagonal entries.
    
    # Actually, the tensor T has:
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some valid (i,j,k)
    # T[a, b, c] = 0 otherwise
    
    # So we need: for all a in [0, n*m), b in [0, m*p), c in [0, p*n):
    #   sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # Number of nonzero T entries: n*m*p = 45
    # Total entries in T: n*m * m*p * p*n = 9*15*15 = 2025
    
    # So we have 2025 constraints, but only 45 are non-trivial (the rest are = 0).
    
    # Let's set up the system. For the 45 non-trivial constraints:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # For the zero constraints, we need:
    # For any (a,b,c) not of the form (i*m+j, j*p+k, k*n+i):
    #   sum_r U[a,r] * V[b,r] * W[c,r] = 0
    
    # This is a complex system. Let's try to use the structure.
    
    # Key observation: if we fix U and V, then W is determined linearly!
    # For each c = k*n+i:
    #   sum_r W[c,r] * (sum over a,b: T[a,b,c] * U[a,r] * V[b,r]) ... 
    # No, that's not right. Let me think again.
    
    # For fixed c = k*n+i, the constraints involving W[c, :] are:
    # For all (a, b) such that T[a,b,c] != 0:
    #   sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # But T[a,b,c] is only nonzero when a = i*m+j, b = j*p+k for the specific
    # i,k that define c. So for c = k*n+i, T[a,b,c] = 1 only when 
    # a = i*m+j and b = j*p+k for some j.
    
    # So for fixed (i,k), we have m constraints:
    # For each j: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1
    
    # This is m equations in the r unknowns of W[k*n+i, :] (with U and V fixed).
    # But it's not linear in W because U and V also appear.
    
    # If we fix U and V, then for each (i,k), we need:
    # sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1 for all j
    
    # This IS linear in W[k*n+i, :]! The coefficients are U[i*m+j, r] * V[j*p+k, r].
    
    # So the strategy is:
    # 1. Fix U and V
    # 2. For each (i,k), solve the linear system for W[k*n+i, :]
    # 3. Check if the zero constraints are also satisfied
    
    # Let's implement this!
    
    rng = np.random.RandomState(42)
    
    for attempt in range(100):
        if time.time() > deadline:
            return None
        
        # Random U and V
        U = rng.randn(n*m, target_rank)
        V = rng.randn(m*p, target_rank)
        
        # For each (i,k), solve for W[k*n+i, :]
        W = np.zeros((p*n, target_rank))
        feasible = True
        
        for i in range(n):
            for k in range(p):
                # Build the m x target_rank matrix: A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                A = np.zeros((m, target_rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                
                # Solve A @ w = ones(m) for w = W[k*n+i, :]
                try:
                    w, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    # Check if the solution is good
                    if rank_a < m or np.linalg.norm(A @ w - np.ones(m)) > 1e-8:
                        feasible = False
                        break
                    W[k*n+i, :] = w
                except:
                    feasible = False
                    break
        
        if not feasible:
            continue
        
        # Now check all constraints
        if _verify_decomposition(U, V, W, n, m, p):
            # Found a solution!
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_structured_merge(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition.
    """
    # Try ranks from lower bound to schoolbook
    lower_bound = max(n*p, n*m, m*p)  # 15
    schoolbook_rank = n * m * p  # 45
    
    # Try a few ranks
    for target_rank in range(lower_bound, min(schoolbook_rank, 30) + 1):
        if time.time() > deadline:
            break
        
        result = _try_algebraic_search(n, m, p, target_rank, deadline)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix 
    by a 3x5 matrix.
    """
    n, m, p = DIMENSIONS  # 3, 3, 5
    deadline = time.time() + 140.0  # Safety margin
    
    # Try to find a better decomposition
    result = _try_structured_merge(n, m, p, deadline)
    
    # Verify the result
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback to schoolbook
        result = _schoolbook(n, m, p)
    
    return result
# EVOLVE-BLOCK-END
