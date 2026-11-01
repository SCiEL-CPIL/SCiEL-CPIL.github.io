"""Matrix multiplication 2x5x6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return U, V, W as lists for the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _verify(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check that all entries are integer or half-integer
    for mat, name in [(U, 'U'), (V, 'V'), (W, 'W')]:
        doubled = mat * 2
        if not np.allclose(doubled, np.round(doubled), atol=1e-8):
            return False
    
    # Check the tensor decomposition
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # (since the matrix multiplication tensor has all 1s for valid indices)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-8:
                    return False
    return True


def _try_partition_decomposition(n, m, p, deadline):
    """
    Try to find a low-rank decomposition using partition strategy.
    
    Partition m=5 into [2,3] and p=6 into [2,4].
    This gives sub-problems:
    - [2,2,2]: rank 7 (Strassen)
    - [2,2,4]: rank 14 (Strassen x 2)
    - [2,3,2]: rank 7 (Strassen-like)
    - [2,3,4]: rank 14 (Strassen-like x 2)
    
    Total: 7 + 14 + 7 + 14 = 42
    """
    # Let's try a direct approach: use Strassen for the 2x2x2 block
    # and schoolbook for the rest, then see if we can optimize.
    
    # Actually, let's try a more systematic approach.
    # The lower bound is max(n*m, m*p, p*n) = max(10, 30, 12) = 30.
    
    # Let's try to find a decomposition of rank 30, 31, ..., 40
    # by solving the system of equations.
    
    # For rank r, we have:
    # Unknowns: r*(n*m + m*p + p*n) = r*(10 + 30 + 12) = 52r
    # Equations: n*m*p = 60
    
    # For r >= 2, we have more unknowns than equations.
    # But the system is nonlinear (trilinear).
    
    # Let's use a different approach: fix W and solve for U, V.
    # For fixed W, the equations become bilinear in U and V.
    
    # Actually, let's try a randomized approach:
    # 1. Initialize W randomly
    # 2. For each (i,k), solve for the r x (m) system
    # 3. Iterate
    
    # Let's try rank 30 first (the theoretical lower bound)
    for target_rank in [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]:
        if time.time() > deadline - 5:
            break
        result = _try_fixed_rank(n, m, p, target_rank, deadline)
        if result is not None:
            return result
    
    return None


def _try_fixed_rank(n, m, p, rank, deadline):
    """
    Try to find a decomposition of exactly the given rank.
    Uses iterative optimization with random restarts.
    """
    best_result = None
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 3:
            break
        
        # Initialize W randomly with values from {-1, -0.5, 0, 0.5, 1}
        W = np.zeros((p * n, rank))
        for idx in range(p * n):
            for r in range(rank):
                W[idx, r] = np.random.choice([-1, -0.5, 0, 0.5, 1])
        
        # Try to solve for U and V given W
        U, V = _solve_uv_given_w(n, m, p, rank, W, deadline)
        
        if U is not None:
            # Verify
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _verify(n, m, p, U_list, V_list, W_list):
                best_result = (U_list, V_list, W_list)
                break
            else:
                # Try to polish
                U, V, W = _polish(n, m, p, rank, U, V, W, deadline)
                if U is not None:
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if _verify(n, m, p, U_list, V_list, W_list):
                        best_result = (U_list, V_list, W_list)
                        break
    
    return best_result


def _solve_uv_given_w(n, m, p, rank, W, deadline):
    """
    Given W, solve for U and V.
    
    The constraint is: for all i,j,k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    For fixed i,k, this becomes:
    sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Let's define for each i,k:
    Let w_ik = W[k*n+i, :] (vector of length rank)
    Then: sum_j (U[i*m+j, :] * w_ik) dot V[j*p+k, :] = 1
    
    This is still bilinear. Let's fix U and solve for V, or vice versa.
    
    Actually, let's use a different parameterization.
    For each i, define A_i = U[i*m:(i+1)*m, :] (m x rank matrix)
    For each k, define B_k = V[:, k] (m x 1 column, but actually V is (m*p) x rank)
    
    Let me re-index: V[j*p+k, r] for fixed k is a vector of length m (over j) and rank.
    
    Let's define:
    For each i: U_i = U[i*m:(i+1)*m, :]  # shape (m, rank)
    For each k: V_k = V[::p, k] reshaped... no, V is indexed by (j*p+k)
    
    V[j*p+k, r] for fixed k: as j varies, we get rows j*p+k of V.
    Let V_k be the matrix with entries V_k[j, r] = V[j*p+k, r], shape (m, rank)
    
    Then the constraint for (i,k) is:
    sum_j sum_r U_i[j, r] * V_k[j, r] * W[k*n+i, r] = 1
    
    Let w_ik = W[k*n+i, :]  # shape (rank,)
    Then: sum_r w_ik[r] * (U_i[:, r] dot V_k[:, r]) = 1
    
    This is a linear equation in the products U_i[:, r] * V_k[:, r].
    
    This is getting complex. Let's try a simpler approach:
    Use alternating least squares.
    """
    
    # Initialize U and V randomly
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    for idx in range(n * m):
        for r in range(rank):
            U[idx, r] = np.random.choice([-1, -0.5, 0, 0.5, 1])
    for idx in range(m * p):
        for r in range(rank):
            V[idx, r] = np.random.choice([-1, -0.5, 0, 0.5, 1])
    
    # Alternating minimization
    for iteration in range(100):
        if time.time() > deadline - 2:
            return None, None
        
        # Fix V and W, solve for U
        # For each (i,j): U[i*m+j, :] should satisfy
        # sum_k sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
        
        # This is: for each i,j, and for each k:
        # U[i*m+j, :] dot (V[j*p+k, :] * W[k*n+i, :]) = 1
        
        # So for fixed i,j, we have p equations in rank unknowns.
        # Let M_ik = V[j*p+k, :] * W[k*n+i, :]  (element-wise product)
        # Then: U[i*m+j, :] dot M_ik = 1 for all k
        
        # This is a linear system: M @ U_col = ones
        # where M is p x rank
        
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build the p x rank matrix M
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                
                # Solve M @ x = ones(p)
                # Least squares
                try:
                    x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U_new[i*m+j, :] = x
                except:
                    U_new[i*m+j, :] = U[i*m+j, :]
        
        # Fix U and W, solve for V
        # For each (j,k): V[j*p+k, :] should satisfy
        # sum_i sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
        
        # For fixed j,k, and for each i:
        # V[j*p+k, :] dot (U[i*m+j, :] * W[k*n+i, :]) = 1
        
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # Build the n x rank matrix M
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                
                # Solve M @ x = ones(n)
                try:
                    x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V_new[j*p+k, :] = x
                except:
                    V_new[j*p+k, :] = V[j*p+k, :]
        
        U, V = U_new, V_new
    
    # Round to nearest half-integer
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    
    return U, V


def _polish(n, m, p, rank, U, V, W, deadline):
    """
    Try to polish the solution to make it exact.
    """
    # Try local search: perturb entries slightly and see if we can improve
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for _ in range(100):
        if time.time() > deadline - 1:
            break
        
        # Randomly perturb one entry
        mat = np.random.choice(['U', 'V', 'W'])
        if mat == 'U':
            idx = np.random.randint(n * m)
            r = np.random.randint(rank)
            old = best_U[idx, r]
            best_U[idx, r] = np.random.choice([-1, -0.5, 0, 0.5, 1])
            if _verify(n, m, p, best_U.tolist(), best_V.tolist(), best_W.tolist()):
                return best_U, best_V, best_W
            best_U[idx, r] = old
        elif mat == 'V':
            idx = np.random.randint(m * p)
            r = np.random.randint(rank)
            old = best_V[idx, r]
            best_V[idx, r] = np.random.choice([-1, -0.5, 0, 0.5, 1])
            if _verify(n, m, p, best_U.tolist(), best_V.tolist(), best_W.tolist()):
                return best_U, best_V, best_W
            best_V[idx, r] = old
        else:
            idx = np.random.randint(p * n)
            r = np.random.randint(rank)
            old = best_W[idx, r]
            best_W[idx, r] = np.random.choice([-1, -0.5, 0, 0.5, 1])
            if _verify(n, m, p, best_U.tolist(), best_V.tolist(), best_W.tolist()):
                return best_U, best_V, best_W
            best_W[idx, r] = old
    
    return None, None, None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions
    result = _try_partition_decomposition(n, m, p, deadline)
    if result is not None:
        U, V, W = result
        best_rank = len(U[0])
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
