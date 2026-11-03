"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy import linalg
from itertools import product

DIMENSIONS = [3, 4, 4]
_START_TIME = time.time()
_TIMEOUT = 115  # safety margin before 120s limit

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    rank = U.shape[1]
    # Reshape to 3D tensors
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # Contract over rank dimension
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    # Check all entries are 1
    if not np.allclose(T, 1.0, atol=1e-10):
        return False
    return True

def _winograd_2x2_block_decomposition():
    """
    Try to build a decomposition using Winograd-style 2x2 block structure.
    For 3x4 * 4x4, we can partition into overlapping 2x2 blocks.
    
    Winograd's F(2,2,2) algorithm computes 2x2 * 2x2 with 7 multiplications.
    We'll try to tile this across the matrices.
    
    For 3x4 * 4x4:
    - A is 3x4, B is 4x4
    - We can split A into rows [0:2, 1:3] and B into cols [0:2, 2:4] etc.
    
    Actually, let's think about this differently. The standard Winograd algorithm
    for 2x2 matrix multiplication uses 7 scalar multiplications instead of 8.
    
    For the full 3x4 * 4x4 problem, we can try a structured approach:
    - Partition the computation into overlapping 2x2 sub-problems
    - Use the 7-term Winograd formula for each 2x2 block
    - Combine with appropriate offsets for the overlapping regions
    
    This is complex. Let's instead try a direct rank reduction search
    with the alternating least squares approach, but with better initialization.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try ranks from 47 down to 40
    for target_rank in range(schoolbook_rank - 1, 39, -1):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        success = _try_rank_reduction_als(target_rank, n, m, p)
        if success:
            return True
    
    return False

def _try_rank_reduction_als(target_rank, n, m, p, max_iter=50, n_restarts=5):
    """
    Alternating least squares to find rank-target_rank decomposition.
    """
    for restart in range(n_restarts):
        if time.time() - _START_TIME > _TIMEOUT:
            return False
        
        np.random.seed(42 + restart * 1000)
        
        # Initialize with random values
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Scale
        U /= np.sqrt(target_rank)
        V /= np.sqrt(target_rank)
        W /= np.sqrt(target_rank)
        
        for it in range(max_iter):
            if time.time() - _START_TIME > _TIMEOUT:
                return False
            
            # Solve for U given V and W
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((p, target_rank))
                    for k in range(p):
                        A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b_vec = np.ones(p)
                    try:
                        sol, _, _, _ = linalg.lstsq(A_mat, b_vec)
                        U[i * m + j, :] = sol
                    except:
                        pass
            
            # Solve for V given U and W
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, target_rank))
                    for i in range(n):
                        A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b_vec = np.ones(n)
                    try:
                        sol, _, _, _ = linalg.lstsq(A_mat, b_vec)
                        V[j * p + k, :] = sol
                    except:
                        pass
            
            # Solve for W given U and V
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, target_rank))
                    for j in range(m):
                        A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b_vec = np.ones(m)
                    try:
                        sol, _, _, _ = linalg.lstsq(A_mat, b_vec)
                        W[k * n + i, :] = sol
                    except:
                        pass
            
            # Check error
            U3 = U.reshape(n, m, target_rank)
            V3 = V.reshape(m, p, target_rank)
            W3 = W.reshape(p, n, target_rank)
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            err = np.max(np.abs(T - 1.0))
            
            if err < 1e-10:
                # Found a valid decomposition! Store it globally
                _global_result = (U.copy(), V.copy(), W.copy())
                return True
    
    return False

_global_result = None

def _find_best_rank():
    """Search for minimum rank decomposition."""
    global _global_result
    n, m, p = DIMENSIONS
    max_rank = n * m * p
    
    # Start from schoolbook
    U, V, W = _schoolbook()
    
    # Try to reduce rank using ALS
    if _try_rank_reduction_als(max_rank - 1, n, m, p):
        U, V, W = _global_result
        if _verify_fast(U, V, W, n, m, p):
            return _to_lists(U, V, W)
    
    # Try more ranks if time permits
    for target_rank in range(max_rank - 2, 39, -1):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        if _try_rank_reduction_als(target_rank, n, m, p):
            U, V, W = _global_result
            if _verify_fast(U, V, W, n, m, p):
                return _to_lists(U, V, W)
    
    # Return schoolbook if no reduction found
    return _to_lists(U, V, W)

def _to_lists(U, V, W):
    """Convert numpy arrays to lists of lists with integer/half-integer entries."""
    def convert(mat):
        rows = []
        for row in mat:
            r = []
            for val in row:
                val_f = float(val)
                nearest_int = round(val_f)
                nearest_half = round(val_f * 2) / 2.0
                if abs(val_f - nearest_int) < 1e-10:
                    r.append(int(nearest_int))
                elif abs(val_f - nearest_half) < 1e-10:
                    r.append(nearest_half)
                else:
                    r.append(val_f)
            rows.append(r)
        return rows
    return {"U": convert(U), "V": convert(V), "W": convert(W)}

def construct():
    """Find rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication."""
    result = _find_best_rank()
    return result
# EVOLVE-BLOCK-END
