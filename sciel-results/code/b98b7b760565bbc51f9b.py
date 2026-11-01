"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def _validate_decomposition(U, V, W, n, m, p):
    """Validate that the decomposition exactly decomposes the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    
    # Check that the decomposition is exact
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1) > 1e-10:
                    return False
    return True

def _fft_decomposition(n, m, p, deadline):
    """
    Use FFT-based approach to find a low-rank decomposition.
    
    The idea: For matrix multiplication C = A @ B where A is n×m and B is m×p,
    we can think of each row of A as a polynomial and use FFT to compute products.
    
    For n=2, m=5, p=5:
    - Each row of A has 5 elements
    - We want to compute 2×5 = 10 output entries
    - Using cyclic convolution with FFT of appropriate size
    
    Key insight: If we embed the 5-element vectors into a larger cyclic group,
    we can diagonalize the multiplication operator.
    
    For 5×5 matrix multiplication, the schoolbook rank is 25.
    Using FFT with size N >= 9 (since 5+5-1=9), we get rank 2*N for the two rows.
    But we can do better by using the structure more carefully.
    
    Actually, let's think differently. The tensor for (2,5,5) has:
    - U: 10×r
    - V: 25×r  
    - W: 10×r
    
    We can try to find a decomposition where r < 50.
    """
    # Try different FFT-based approaches
    best_result = None
    best_rank = n * m * p  # Start with schoolbook rank
    
    # Approach 1: Direct embedding using cyclic groups
    # For m=5, p=5, we need convolution length >= 9
    # Use FFT of size 9 or 10
    
    for fft_size in [9, 10, 12, 15]:
        if time.time() > deadline - 20:
            break
        
        # Build decomposition using FFT structure
        # For each output position (i, k) where i in range(n), k in range(p):
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # We can use the fact that multiplication by a fixed matrix B
        # can be represented in the Fourier domain
        
        # Let's try a structured approach:
        # U[i*m+j, r] = alpha[i,j,r]
        # V[j*p+k, r] = beta[j,k,r]  
        # W[k*n+i, r] = gamma[k,i,r]
        
        # For rank r, we need:
        # sum_r alpha[i,j,r] * beta[j,k,r] * gamma[k,i,r] = 1 for all i,j,k
        
        # This is a system of n*m*p = 50 equations in 3*r*min(n*m,m*p,p*n) variables
        
        # Let's try to find a solution with small r using numerical methods
        # First, try r = 25 (schoolbook for 5x5, times 2 for the two rows would be 50)
        # But we might do better
        
        # Try incremental ranks
        for r in range(20, min(51, best_rank)):
            if time.time() > deadline - 10:
                break
            
            # Set up the linear system
            # Variables: U[0..9, 0..r-1], V[0..24, 0..r-1], W[0..9, 0..r-1]
            # That's 10*r + 25*r + 10*r = 45*r variables
            # Constraints: 50 equations (one for each (i,j,k) tuple)
            
            # For r >= 50/45 ≈ 1.11, we have more variables than equations
            # But we need a special structure
            
            # Let's try a specific ansatz:
            # Use the cyclic group Z_m structure
            # For j in range(m), define characters chi_s(j) = exp(2*pi*i*s*j/m)
            
            # Actually, let's use a simpler approach:
            # Try to find U, V, W such that the decomposition works
            
            # Initialize with random values and optimize
            np.random.seed(42)
            
            # We'll use the fact that we can scale U and V arbitrarily as long as
            # U @ V @ W (in the tensor sense) = 1
            # So we can fix one of them
            
            # Let's try: fix W to be a simple pattern
            # W[k*n+i, r] = 1 if (k+i) % r == 0 else 0  (or some other pattern)
            
            # Better: use the structure of matrix multiplication
            # C[i,k] = sum_j A[i,j] B[j,k]
            
            # For a rank-r decomposition, we can write:
            # C[i,k] = sum_r u[i,k,r] * v[i,k,r] * w[i,k,r]  (schematic)
            
            # Let's try a specific construction:
            # For each "frequency" s in range(r):
            # U[i*m+j, s] = exp(2*pi*i*s*j/m)  (Fourier mode)
            # V[j*p+k, s] = exp(2*pi*i*s*k/p)  (Fourier mode)
            # W[k*n+i, s] = c[s]  (coupling constant)
            
            # Then: sum_s exp(2*pi*i*s*j/m) * exp(2*pi*i*s*k/p) * c[s]
            # = sum_s c[s] * exp(2*pi*i*s*(j/m + k/p))
            
            # We want this to equal 1 for all j,k... that's not quite right.
            
            # Let me reconsider. The tensor is:
            # T[i,j,k] = 1 if the (i,j,k) entry should be 1
            # For matrix multiplication, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
            # Wait no, that's not right either.
            
            # The matrix multiplication tensor has:
            # T[i,j,k] = 1 for all valid (i,j,k)
            # And the decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            
            # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
            
            # Let's try a Kronecker product structure:
            # U = U1 ⊗ U2, V = V1 ⊗ V2, W = W1 ⊗ W2 (schematically)
            
            # For n=2, m=5, p=5:
            # We can decompose as: (2,5,5) = (2,1,1) ⊗ (1,5,5)
            # Rank(2,1,1) = 2, Rank(1,5,5) = ?
            
            # For (1,5,5), the schoolbook rank is 25.
            # Can we do better? The matrix multiplication tensor M_{1,5,5}
            # has been studied. The known bounds are...
            
            # Actually, for (1,m,p), the rank is at least max(m,p) and at most m*p.
            # For (1,5,5), rank is between 5 and 25.
            
            # Let's try to find a rank-10 decomposition for (1,5,5) using FFT
            # If we can do that, then (2,5,5) would have rank 2*10 = 20
            
            # For (1,5,5): C[k] = sum_j A[j] B[j,k] for k in range(5)
            # This is: C = A @ B where A is 1×5, B is 5×5
            
            # Using FFT: if we embed A and each column of B into polynomials of degree < 5,
            # their product has degree < 10, so we need FFT size >= 10.
            
            # But we only need the first 5 coefficients of the product.
            
            # Let's try rank 10 for (1,5,5):
            # U[j, s] = omega^{j*s} for s in range(10), where omega = exp(2*pi*i/10)
            # V[j,k,s] = ? 
            # W[k,s] = ?
            
            # This is getting complicated. Let's just try to solve the system numerically.
            
            # For rank r, we have 45*r unknowns and 50 equations.
            # For r >= 2, we have more unknowns than equations.
            # But we need a specific structure.
            
            # Let's try r = 25 first (should be achievable)
            if r >= 25:
                # Use the schoolbook decomposition
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = i * m * p + j * p + k
                            if idx < r:
                                U[i * m + j][idx] = 1
                                V[j * p + k][idx] = 1
                                W[k * n + i][idx] = 1
                
                if _validate_decomposition(U, V, W, n, m, p):
                    if r < best_rank:
                        best_rank = r
                        best_result = {"U": U, "V": V, "W": W}
                    break  # Don't try larger r in this loop
                
                break
            
            # For r < 25, try to find a solution
            # This is a nonlinear system, so use optimization
            # But that's expensive. Let's skip for now.
            
            pass
    
    # Also try the original pattern-based approach
    if time.time() < deadline - 30:
        result = _try_pattern_decomposition(n, m, p, n * m * p, deadline)
        if _validate_decomposition(result["U"], result["V"], result["W"], n, m, p):
            rank_found = len(result["U"][0])
            if rank_found < best_rank:
                best_rank = rank_found
                best_result = result
    
    # Return the best result found
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}

def _try_pattern_decomposition(n, m, p, rank, deadline):
    """Try to find a rank-minimal decomposition using pattern-based approach."""
    start_time = time.time()
    
    # Start with the direct sum decomposition (rank n*m*p)
    result = _direct_sum_decomposition(n, m, p)
    U = result["U"]
    V = result["V"]
    W = result["W"]
    
    current_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Phase 1: Try removing single terms repeatedly
    progress = True
    while progress:
        if time.time() > deadline - 30:
            break
        progress = False
        for r_remove in range(current_rank):
            if time.time() > deadline - 30:
                break
            success, U_new, V_new, W_new = _try_single_removal(U, V, W, r_remove, n, m, p)
            if success:
                U, V, W = U_new, V_new, W_new
                current_rank -= 1
                best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
                progress = True
                break
    
    # Phase 2: Try removing pairs of terms
    if time.time() < deadline - 15:
        progress = True
        while progress:
            if time.time() > deadline - 15:
                break
            progress = False
            found = False
            for r1 in range(current_rank):
                if time.time() > deadline - 15:
                    break
                for r2 in range(r1 + 1, current_rank):
                    if time.time() > deadline - 15:
                        break
                    success, U_new, V_new, W_new = _try_double_removal(U, V, W, r1, r2, n, m, p)
                    if success:
                        U, V, W = U_new, V_new, W_new
                        current_rank -= 2
                        best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
                        progress = True
                        found = True
                        break
                if found:
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _try_single_removal(U, V, W, r_remove, n, m, p):
    """Try removing a single term and check if decomposition is still valid."""
    rank = len(U[0])
    U_test = [row[:] for row in U]
    V_test = [row[:] for row in V]
    W_test = [row[:] for row in W]
    
    # Zero out the column for r_remove
    for i in range(len(U_test)):
        U_test[i][r_remove] = 0
    for i in range(len(V_test)):
        V_test[i][r_remove] = 0
    for i in range(len(W_test)):
        W_test[i][r_remove] = 0
    
    # Check if still valid
    if _validate_decomposition(U_test, V_test, W_test, n, m, p):
        # Remove the column
        U_test = [row[:r_remove] + row[r_remove+1:] for row in U_test]
        V_test = [row[:r_remove] + row[r_remove+1:] for row in V_test]
        W_test = [row[:r_remove] + row[r_remove+1:] for row in W_test]
        return True, U_test, V_test, W_test
    return False, None, None, None

def _try_double_removal(U, V, W, r1, r2, n, m, p):
    """Try removing two terms and check if decomposition is still valid."""
    rank = len(U[0])
    U_test = [row[:] for row in U]
    V_test = [row[:] for row in V]
    W_test = [row[:] for row in W]
    
    # Zero out the columns for r1 and r2
    for i in range(len(U_test)):
        U_test[i][r1] = 0
        U_test[i][r2] = 0
    for i in range(len(V_test)):
        V_test[i][r1] = 0
        V_test[i][r2] = 0
    for i in range(len(W_test)):
        W_test[i][r1] = 0
        W_test[i][r2] = 0
    
    # Check if still valid
    if _validate_decomposition(U_test, V_test, W_test, n, m, p):
        # Remove the columns (handle ordering)
        if r1 < r2:
            U_test = [row[:r1] + row[r1+1:r2] + row[r2+1:] for row in U_test]
            V_test = [row[:r1] + row[r1+1:r2] + row[r2+1:] for row in V_test]
            W_test = [row[:r1] + row[r1+1:r2] + row[r2+1:] for row in W_test]
        else:
            U_test = [row[:r2] + row[r2+1:r1] + row[r1+1:] for row in U_test]
            V_test = [row[:r2] + row[r2+1:r1] + row[r1+1:] for row in V_test]
            W_test = [row[:r2] + row[r2+1:r1] + row[r1+1:] for row in W_test]
        return True, U_test, V_test, W_test
    return False, None, None, None

def _direct_sum_decomposition(n, m, p):
    """Build a rank n*m*p decomposition via direct sum of 1x m x p blocks."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Try FFT-based decomposition
    result = _fft_decomposition(n, m, p, deadline)
    
    # Validate the result
    if _validate_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
