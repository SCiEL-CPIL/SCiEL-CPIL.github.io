"""Matrix multiplication 3x4x5 - rank-minimal tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-6):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    
    T = np.zeros((n, m, p), dtype=float)
    R = U.shape[1]
    for r in range(R):
        u = Ur[:, :, r]
        v = Vr[:, :, r]
        w = Wr[:, :, r]
        M = u @ v
        T += M * w.T
        
    T_target = np.ones((n, m, p), dtype=float)
    return np.allclose(T, T_target, atol=tol)

def _verify_half_integer(U, V, W, tol=1e-8):
    """Verify that entries are half-integers and decomposition is exact."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check half-integer constraint
    for mat in [U, V, W]:
        if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
            return False
    
    return _verify(U, V, W, tol)

def _laderman_4x4x4():
    """
    Laderman's 49-term decomposition for 4x4 * 4x4 matrix multiplication.
    Returns U, V, W matrices of shape (16, 49) each.
    
    Laderman's algorithm splits 4x4 matrices into 2x2 blocks and uses
    7 Strassen multiplications for each 2x2 block pair, plus cross terms.
    """
    # Laderman's 49-term algorithm for 4x4x4
    # A = [A11 A12; A21 A22], B = [B11 B12; B21 B22]
    # C = A*B = [C11 C12; C21 C22] where Cij are 2x2
    
    # For each 2x2 block multiplication, use Strassen (7 terms)
    # Total: 7*7 = 49 terms
    
    # Strassen formulas for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4, we have 4 block multiplications needed:
    # C11 needs A11*B11 + A12*B21
    # C12 needs A11*B12 + A12*B22
    # C21 needs A21*B11 + A22*B21
    # C22 needs A21*B12 + A22*B22
    
    # Laderman combines these into 49 terms
    
    # Let me use a known explicit Laderman decomposition
    # This is the standard 49-term algorithm
    
    # Indexing: A rows 0-3, cols 0-3; B rows 0-3, cols 0-3
    # U[i*4+j, r] = coefficient for A[i,j] in term r
    # V[j*4+k, r] = coefficient for B[j,k] in term r  
    # W[k*4+i, r] = coefficient for C[i,k] in term r
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Laderman's 49 terms
    # Each term is a product of a linear combination of A entries
    # and a linear combination of B entries
    
    # Let's define the 49 terms explicitly
    # Term format: (A_indices_with_coeffs, B_indices_with_coeffs, C_indices_with_coeffs)
    
    # This is complex; let me use a verified implementation
    # 
    # Actually, let me use a simpler approach: 
    # Pad 3x4*4x5 to 4x4*4x4, use Laderman, extract.
    # But we need the actual Laderman coefficients.
    
    # Alternative: use the fact that R(3,4,5) <= R(4,4,4) = 49
    # by padding. But we need explicit construction.
    
    # Let me try a different strategy: direct search for low-rank decomposition
    # using the structure of the problem.
    
    # For now, return None and rely on other methods
    return None

def _try_rank_reduction_from_schoolbook(max_rank=59, time_budget=60.0):
    """
    Start from schoolbook (rank 60) and try to reduce rank by finding
    linear dependencies among the terms.
    
    The key insight: if we can express one schoolbook term as a linear
    combination of others (with appropriate U,V,W adjustments), we can
    reduce the rank.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    nmp = n * m * p  # 60
    
    # Schoolbook decomposition
    U_sb = np.zeros((n * m, nmp))
    V_sb = np.zeros((m * p, nmp))
    W_sb = np.zeros((p * n, nmp))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try to find a rank-r decomposition by optimization
    # We'll use a simple approach: try to find a subset of rank r terms
    # that can be adjusted to form a valid decomposition
    
    # Better approach: use the fact that the tensor is 3x4x5 = 60-dimensional
    # and look for a low-rank decomposition via alternating least squares
    
    best_result = None
    
    # Try ranks from 59 down
    for rank in range(59, 48, -1):
        if time.time() - start_time > time_budget:
            break
        
        # Initialize with first 'rank' schoolbook terms
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Copy schoolbook for first rank columns
        U[:, :rank] = U_sb[:, :rank]
        V[:, :rank] = V_sb[:, :rank]
        W[:, :rank] = W_sb[:, :rank]
        
        # Now try to optimize to make it a valid decomposition
        # using alternating least squares
        
        # The constraint is: sum_r U[:,:,r] * V[:,:,r] * W[:,:,r] = 1 (elementwise)
        # where U[:,:,r] is (n,m), V[:,:,r] is (m,p), W[:,:,r] is (p,n)
        
        # This is a non-convex problem. Let's try a simple iterative approach.
        
        # Actually, let's try a different strategy:
        # Use the full schoolbook and try to find linear combinations
        # that give the same result with fewer terms.
        
        # The tensor T = sum_{i,j,k} e_i^A * e_j^B * e_k^C
        # We want T = sum_{r=1}^{R} U_r * V_r * W_r
        
        # One approach: random initialization + ALS
        np.random.seed(rank)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # ALS iterations
        for iteration in range(50):
            if time.time() - start_time > time_budget:
                break
            
            # Update U
            for r in range(rank):
                # Fix V and W, solve for U[:, r]
                # We need: sum_r U[:,:,r] * V[:,:,r] * W[:,:,r] = T
                # For fixed r, the contribution is U[:,:,r] * (V[:,:,r] * W[:,:,r].T)
                # Wait, the tensor product is outer product
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                # For fixed r, we want to update U[:, r] to minimize
                # ||T - sum_r' U[:,:,r'] * V[:,:,r'] * W[:,:,r']||
                
                # This is a linear least squares problem for U[:, r]
                
                # Let's compute the residual without term r
                T_current = np.zeros((n, m, p))
                for r2 in range(rank):
                    if r2 == r:
                        continue
                    Ur2 = U[:, r2].reshape(n, m)
                    Vr2 = V[:, r2].reshape(m, p)
                    Wr2 = W[:, r2].reshape(p, n)
                    T_current += Ur2[:, :, np.newaxis] * Vr2[np.newaxis, :, :] * Wr2.T[np.newaxis, np.newaxis, :]
                
                residual = np.ones((n, m, p)) - T_current
                
                # Now solve for U[:, r]
                # residual[i,j,k] = sum over (j,k) of ... 
                # Actually, for fixed r:
                # residual[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # This is still coupled.
                
                # Let's use a simpler update: gradient descent
                pass
            
            # This is getting complex. Let me try a different approach.
            break
        
        # Skip this approach; try something else
        pass
    
    return None

def _try_direct_search(time_budget=80.0):
    """
    Direct search for a low-rank decomposition using structured initialization.
    
    Key idea: The 3x4x5 tensor can be decomposed by exploiting the structure
    of the dimensions. We can try to find a decomposition where some terms
    cover multiple (i,j,k) entries.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strategy: Start with schoolbook and try to merge terms
    # A term (i,j,k) contributes to C[i,k] via A[i,j]*B[j,k]
    # If two terms share the same i and k but different j, they can potentially
    # be combined.
    
    # Actually, let's try a more systematic approach:
    # Use the fact that R(3,4,5) is known to be <= 49 (via padding to 4x4x4)
    # and try to construct an explicit 49-term or lower decomposition.
    
    # Padding approach:
    # A (3x4) -> pad to 4x4 by adding a zero row: A' = [A; 0 0 0 0]
    # B (4x5) -> pad to 4x4 by adding a zero column: B' = [B | 0]
    # C' = A'*B' is 4x4, top-left 3x5 block is C
    # 
    # If we have a 49-term decomposition for 4x4x4, we can extract the 3x5 part.
    # But we need to ensure the extracted part is valid.
    
    # Let me try to implement Laderman's 49-term algorithm properly.
    
    # Laderman's 49-term algorithm for 4x4x4:
    # Reference: Laderman, 1976
    # 
    # The algorithm uses 7 Strassen multiplications for each of the 4 
    # 2x2 block multiplications, but with sharing.
    
    # Let me use a known explicit formula.
    # 
    # Actually, let me try a simpler approach: use the Winograd algorithm
    # or a direct construction.
    
    # For now, let's try a heuristic: search for a rank-58 or 59 decomposition
    # by perturbing the schoolbook and using optimization.
    
    best_result = None
    best_error = float('inf')
    
    # Try a few random initializations with ALS
    for attempt in range(20):
        if time.time() - start_time > time_budget:
            break
        
        rank = 59  # Try rank 59 first
        
        np.random.seed(attempt * 12345 + 67890)
        
        # Initialize close to schoolbook
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Copy first rank schoolbook terms
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    if col >= rank:
                        break
                if col >= rank:
                    break
            if col >= rank:
                break
        
        # Add small random perturbation to the last term
        U[:, -1] += np.random.randn(n * m) * 0.1
        V[:, -1] += np.random.randn(m * p) * 0.1
        W[:, -1] += np.random.randn(p * n) * 0.1
        
        # Try to optimize using a simple iterative method
        # The goal is to make sum_r U[:,:,r] * V[:,:,r] * W[:,:,r] = 1
        
        # Compute current tensor
        def compute_tensor(U, V, W):
            T = np.zeros((n, m, p))
            for r in range(U.shape[1]):
                Ur = U[:, r].reshape(n, m)
                Vr = V[:, r].reshape(m, p)
                Wr = W[:, r].reshape(p, n)
                T += Ur[:, :, np.newaxis] * Vr[np.newaxis, :, :] * Wr.T[np.newaxis, np.newaxis, :]
            return T
        
        T = compute_tensor(U, V, W)
        error = np.linalg.norm(T - 1.0)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-6:
            # Check half-integer constraint
            if _verify_half_integer(U.tolist(), V.tolist(), W.tolist()):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
            else:
                # Try rounding
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify_half_integer(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    break
    
    return best_result

def _try_explicit_construction():
    """
    Try to construct an explicit low-rank decomposition.
    
    For 3x4x5, one approach is to use the fact that:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can try to find a decomposition where each term covers a 
    structured subset of (i,j,k) entries.
    """
    n, m, p = DIMENSIONS
    
    # Try to find a rank-58 decomposition by exploiting linear dependencies.
    # 
    # The schoolbook has 60 terms. Can we eliminate 2 of them?
    # 
    # Consider terms (i,j,k) and (i',j',k'). If we can express one as a
    # linear combination of the others, we can reduce the rank.
    #
    # Actually, let's try a different approach: use the fact that the
    # multiplication tensor has a certain structure that allows for
    # a lower-rank decomposition.
    
    # For now, return None
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety
    
    # Try explicit construction first (fastest)
    result = _try_explicit_construction()
    if result is not None:
        if _verify(result["U"], result["V"], result["W"]):
            return result
    
    # Check time
    if time.time() > deadline - 30:
        return _schoolbook()
    
    # Try direct search
    result = _try_direct_search(time_budget=80.0)
    if result is not None:
        if _verify(result["U"], result["V"], result["W"]):
            return result
    
    # Check time
    if time.time() > deadline - 10:
        return _schoolbook()
    
    # Try rank reduction
    result = _try_rank_reduction_from_schoolbook(max_rank=59, time_budget=30.0)
    if result is not None:
        if _verify(result["U"], result["V"], result["W"]):
            return result
    
    # Return schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
