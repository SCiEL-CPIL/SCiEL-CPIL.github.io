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
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the multiplication tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k]
    # The decomposition requires: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    # Actually, the tensor for matrix mult is: T[i,j,k] where output[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, let me re-read: W[k*n+i,r]
    # So W is indexed by (k, i) where k is from B's columns and i is from A's rows
    # The tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This should equal 1 for all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    rank = U.shape[1]
    # Reshape to 3D tensors
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # Contract over rank dimension
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    # Check all entries are 1
    if not np.allclose(T, 1.0, atol=1e-10):
        return False
    return True

def _try_rank_reduction(U, V, W, target_rank, n, m, p, max_iter=100):
    """Try to reduce rank to target_rank by solving linear systems."""
    current_rank = U.shape[1]
    if target_rank >= current_rank:
        return U, V, W, False
    
    # We want to find new U', V', W' of rank target_rank
    # such that sum_r' U'[i*m+j,r'] * V'[j*p+k,r'] * W'[k*n+i,r'] = 1
    # 
    # One approach: fix W, solve for U and V, or use alternating least squares
    # with exact constraints
    #
    # Better approach: use the fact that the tensor has structure
    # For matrix multiplication, we can try structured decompositions
    #
    # Let's try a different approach: randomly initialize and use exact linear solve
    # The constraint is bilinear in (U,V) for fixed W, etc.
    
    # Try alternating approach:
    # 1. Fix V and W, solve for U
    # 2. Fix U and W, solve for V  
    # 3. Fix U and V, solve for W
    # Repeat until convergence
    
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    # Truncate to target_rank by keeping first target_rank columns
    # This won't work directly, need to find new factors
    
    # Alternative: use random initialization and exact solve
    np.random.seed(42)
    
    # Initialize with random values
    U_rand = np.random.randn(n * m, target_rank)
    V_rand = np.random.randn(m * p, target_rank)
    W_rand = np.random.randn(p * n, target_rank)
    
    # Scale to have reasonable magnitude
    U_rand /= np.sqrt(target_rank)
    V_rand /= np.sqrt(target_rank)
    W_rand /= np.sqrt(target_rank)
    
    best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
    best_err = float('inf')
    
    for it in range(max_iter):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        # Solve for U given V and W
        # For each (i,j): U[i*m+j,:] = ?
        # Constraint: for all k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # This is: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
        # Let M[i,j] be a matrix where M[i,j,k,r] = V[j*p+k,r] * W[k*n+i,r]
        # Then we need: sum_r U[i*m+j,r] * M[i,j,k,r] = 1 for all k
        # This is a linear system: A_{ij} * U[i*m+j,:] = b_{ij}
        # where A_{ij} is p x target_rank with A_{ij}[k,r] = V[j*p+k,r] * W[k*n+i,r]
        # and b_{ij} = ones(p)
        
        U_new = np.zeros((n * m, target_rank))
        for i in range(n):
            for j in range(m):
                # Build A matrix: p x target_rank
                A_mat = np.zeros((p, target_rank))
                for k in range(p):
                    for r in range(target_rank):
                        A_mat[k, r] = V_curr[j * p + k, r] * W_curr[k * n + i, r]
                b_vec = np.ones(p)
                # Solve A_mat @ U[i*m+j,:] = b_vec
                try:
                    sol, residuals, rank_, singular_values = linalg.lstsq(A_mat, b_vec)
                    U_new[i * m + j, :] = sol
                except:
                    U_new[i * m + j, :] = U_curr[i * m + j, :target_rank]
        
        # Solve for V given U and W
        V_new = np.zeros((m * p, target_rank))
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, target_rank))
                for i in range(n):
                    for r in range(target_rank):
                        A_mat[i, r] = U_curr[i * m + j, r] * W_curr[k * n + i, r]
                b_vec = np.ones(n)
                try:
                    sol, residuals, rank_, singular_values = linalg.lstsq(A_mat, b_vec)
                    V_new[j * p + k, :] = sol
                except:
                    V_new[j * p + k, :] = V_curr[j * p + k, :target_rank]
        
        # Solve for W given U and V
        W_new = np.zeros((p * n, target_rank))
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, target_rank))
                for j in range(m):
                    for r in range(target_rank):
                        A_mat[j, r] = U_curr[i * m + j, r] * V_curr[j * p + k, r]
                b_vec = np.ones(m)
                try:
                    sol, residuals, rank_, singular_values = linalg.lstsq(A_mat, b_vec)
                    W_new[k * n + i, :] = sol
                except:
                    W_new[k * n + i, :] = W_curr[k * n + i, :target_rank]
        
        # Check error
        U3 = U_new.reshape(n, m, target_rank)
        V3 = V_new.reshape(m, p, target_rank)
        W3 = W_new.reshape(p, n, target_rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = np.max(np.abs(T - 1.0))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
        
        if err < 1e-10:
            return U_new, V_new, W_new, True
        
        # Update
        U_curr, V_curr, W_curr = U_new, V_new, W_new
    
    return best_U, best_V, best_W, best_err < 1e-10

def _find_best_rank():
    """Search for minimum rank decomposition."""
    n, m, p = DIMENSIONS
    max_rank = n * m * p
    
    # Start from schoolbook
    U, V, W = _schoolbook()
    
    # Try to reduce rank
    for target_rank in range(max_rank - 1, 0, -1):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        U_new, V_new, W_new, success = _try_rank_reduction(U, V, W, target_rank, n, m, p)
        if success:
            # Verify
            if _verify_fast(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                # Convert to lists with integer/half-integer entries if possible
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
                # Try to convert to exact rational
                val_f = float(val)
                # Check if it's close to an integer or half-integer
                nearest_int = round(val_f)
                nearest_half = round(val_f * 2) / 2.0
                if abs(val_f - nearest_int) < 1e-10:
                    r.append(int(nearest_int))
                elif abs(val_f - nearest_half) < 1e-10:
                    r.append(nearest_half)
                else:
                    # Keep as float
                    r.append(val_f)
            rows.append(r)
        return rows
    return {"U": convert(U), "V": convert(V), "W": convert(W)}

def construct():
    """Find rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication."""
    result = _find_best_rank()
    return result
# EVOLVE-BLOCK-END
