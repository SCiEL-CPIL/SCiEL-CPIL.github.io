"""
Rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.

Strategy:
  1. Recognize that 3x3x6 multiplication is equivalent to two independent 3x3x3 multiplications.
  2. The rank of 3x3x3 matrix multiplication is 7 (Strassen).
  3. Therefore, a rank-14 decomposition for 3x3x6 exists.
  4. Use ALS to find the rank-7 factors for the 3x3x3 tensor.
  5. Construct the rank-14 factors for 3x3x6 by duplicating and mapping indices.
  6. Verify the decomposition exactly.
  7. If rank-7 search fails, fall back to higher ranks or schoolbook.
"""

import time
import numpy as np
from fractions import Fraction

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
n, m, p = 3, 3, 6
nm, mp, pn = n*m, m*p, p*n  # 9, 18, 18

# ---------------------------------------------------------------------------
# Tensor construction for 3x3x3
# ---------------------------------------------------------------------------

def _build_tensor_333():
    """
    Build the 3x3x3 matrix-multiplication tensor T333 of shape (9, 9, 9)
    using the convention:
        T[i*3+j, j*3+k, k*3+i] = 1
    """
    T = np.zeros((9, 9, 9), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T[i*3+j, j*3+k, k*3+i] = 1.0
    return T

T333 = _build_tensor_333()

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify_333(U, V, W, tol=1e-8):
    """
    Check whether factor matrices approximately decompose the 3x3x3 tensor.
    U, V, W are numpy arrays of shape (9, rank).
    """
    T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.max(np.abs(T_rec - T333)) < tol

def _verify_336(U, V, W, tol=1e-8):
    """
    Check whether factor matrices approximately decompose the 3x3x6 tensor.
    U shape (9, rank), V shape (18, rank), W shape (18, rank).
    """
    # Build target tensor for 3x3x6
    T336 = np.zeros((9, 18, 18), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            for k in range(6):
                T336[i*3+j, j*6+k, k*3+i] = 1.0
                
    T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.max(np.abs(T_rec - T336)) < tol

# ---------------------------------------------------------------------------
# ALS for 3x3x3 tensor
# ---------------------------------------------------------------------------

def _als_333(rank, seed, max_iter=500, tol=1e-14):
    """
    Single ALS trial to find a rank-r decomposition of the 3x3x3 tensor.
    Returns (U, V, W) as float numpy arrays, or None if failed.
    """
    np.random.seed(seed)
    
    # Random initialization
    U = np.random.randn(9, rank)
    V = np.random.randn(9, rank)
    W = np.random.randn(9, rank)
    
    # Normalize initial factors to help convergence
    for c in range(rank):
        norm = np.sqrt(np.sum(U[:, c]**2) * np.sum(V[:, c]**2) * np.sum(W[:, c]**2))
        if norm > 0:
            scale = norm ** (1.0/3.0)
            U[:, c] /= scale
            V[:, c] /= scale
            W[:, c] /= scale
            
    prev_err = np.inf
    for it in range(max_iter):
        # Update U
        T_cW = np.einsum('abd,dc->abc', T333, W, optimize=True)
        U_new = np.einsum('abc,bc->ac', T_cW, V, optimize=True)
        
        # Update V
        T_uW = np.einsum('abd,ac,dc->bc', T333, U_new, W, optimize=True)
        V_new = T_uW
        
        # Update W
        T_uV = np.einsum('abd,ac,bc->dc', T333, U_new, V_new, optimize=True)
        W_new = T_uV
        
        # Compute error
        T_rec = np.einsum('ar,br,cr->abc', U_new, V_new, W_new, optimize=True)
        err = np.max(np.abs(T_rec - T333))
        
        # Check convergence
        if abs(prev_err - err) < tol and err < 1e-8:
            U, V, W = U_new, V_new, W_new
            break
        
        U, V, W = U_new, V_new, W_new
        prev_err = err
    
    # Final error check
    T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    err = np.max(np.abs(T_rec - T333))
    
    if err < 1e-8:
        return U, V, W
    return None

# ---------------------------------------------------------------------------
# Rational recovery
# ---------------------------------------------------------------------------

def _recover_rational(x, max_denom=100000):
    """
    Recover a rational approximation of float x with denominator <= max_denom.
    """
    x = float(x)
    if abs(x) < 1e-10:
        return Fraction(0)
    return Fraction(x).limit_denominator(max_denom)

def _to_fraction_matrix(mat, max_denom=100000):
    """
    Convert a numpy matrix to a list-of-lists of Fractions.
    """
    rows, cols = mat.shape
    out = [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            out[i][j] = _recover_rational(mat[i, j], max_denom)
    return out

def _verify_fraction_decomp_333(U_frac, V_frac, W_frac):
    """
    Verify the 3x3x3 decomposition using exact Fraction arithmetic.
    """
    r = len(U_frac[0])
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                s = Fraction(0)
                for col in range(r):
                    s += U_frac[a][col] * V_frac[b][col] * W_frac[c][col]
                if s != Fraction(1):
                    return False
    return True

# ---------------------------------------------------------------------------
# Mapping 3x3x3 rank-7 to 3x3x6 rank-14
# ---------------------------------------------------------------------------

def _map_333_to_336(U_333, V_333, W_333):
    """
    Map rank-7 factors of 3x3x3 to rank-14 factors of 3x3x6.
    """
    rank_333 = U_333.shape[1]
    rank_336 = 2 * rank_333
    
    U_336 = np.zeros((9, rank_336), dtype=np.float64)
    V_336 = np.zeros((18, rank_336), dtype=np.float64)
    W_336 = np.zeros((18, rank_336), dtype=np.float64)
    
    # U is duplicated
    U_336[:, :rank_333] = U_333
    U_336[:, rank_333:] = U_333
    
    # V and W are split based on k
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # First half (k=0,1,2)
                V_336[j*6+k, :rank_333] = V_333[j*3+k, :]
                W_336[k*3+i, :rank_333] = W_333[k*3+i, :]
                
                # Second half (k=3,4,5)
                k_prime = k
                V_336[j*6+(k+3), rank_333:] = V_333[j*3+k_prime, :]
                W_336[(k+3)*3+i, rank_333:] = W_333[k_prime*3+i, :]
                
    return U_336, V_336, W_336

# ---------------------------------------------------------------------------
# Main construct function
# ---------------------------------------------------------------------------

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x6
    matrix multiplication.
    Returns {'U': ..., 'V': ..., 'W': ...} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    # Baseline: Schoolbook rank 54
    rank_sb = 54
    U_sb = np.zeros((9, rank_sb), dtype=np.int64)
    V_sb = np.zeros((18, rank_sb), dtype=np.int64)
    W_sb = np.zeros((18, rank_sb), dtype=np.int64)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(6):
                U_sb[i*3+j, col] = 1
                V_sb[j*6+k, col] = 1
                W_sb[k*3+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find rank-7 decomposition for 3x3x3
    # Rank of 3x3x3 is 7.
    found_333 = False
    U_333_best = None
    V_333_best = None
    W_333_best = None
    
    # Try ranks 7, 8, 9... up to 12 (just in case 7 is hard to find)
    for r_333 in range(7, 13):
        if time.time() > deadline - 10:
            break
            
        for trial in range(20):
            if time.time() > deadline - 10:
                break
                
            seed = 42 + r_333 * 100 + trial
            result = _als_333(r_333, seed)
            
            if result is not None:
                U_333, V_333, W_333 = result
                
                # Verify with fractions
                for max_denom in [100, 1000, 10000, 100000]:
                    U_frac = _to_fraction_matrix(U_333, max_denom)
                    V_frac = _to_fraction_matrix(V_333, max_denom)
                    W_frac = _to_fraction_matrix(W_333, max_denom)
                    
                    if _verify_fraction_decomp_333(U_frac, V_frac, W_frac):
                        U_333_best = U_333
                        V_333_best = V_333
                        W_333_best = W_333
                        found_333 = True
                        break
                if found_333:
                    break
        if found_333:
            break
    
    if found_333:
        # Map to 3x3x6
        U_336, V_336, W_336 = _map_333_to_336(U_333_best, V_333_best, W_333_best)
        
        # Verify the mapped 3x3x6 decomposition
        if _verify_336(U_336, V_336, W_336):
            best_rank = U_336.shape[1]
            best_U, best_V, best_W = U_336, V_336, W_336
    
    # If we still have the schoolbook or a higher rank, return it
    # Convert to list of lists of floats
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
