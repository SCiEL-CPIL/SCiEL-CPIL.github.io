"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# Deadline guard: evaluator kills after 150s, use 120s as safety margin
_DEADLINE = time.time() + 115.0

def _remaining_time():
    return _DEADLINE - time.time()

def _is_valid_decomposition(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Check if U, V, W form a valid tensor decomposition for matrix multiplication.
    
    The tensor T[i,j,k] = delta_{i+k, j} in index-free form:
    T_{(i,j),(j,k),(k,i)} = 1 for all i,j,k (this is the matmul tensor)
    
    Actually, the matmul tensor is: T[i,j,k] where we multiply A[i,j] * B[j,k] -> C[i,k]
    So the tensor has entries T[(i,j),(j,k),(i,k)] = 1
    
    In the factorization: T = sum_r U[:,r] * V[:,r] * W[:,r]
    where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    The index convention:
    - U indexed by (i*m + j) for A[i,j]
    - V indexed by (j*p + k) for B[j,k]  
    - W indexed by (k*n + i) for C[i,k]  [trace convention: W[k*n+i, r]]
    
    So T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all i,j,k
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Compute the reconstructed tensor
    # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We need T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all i,j,k
    # And T should be 0 elsewhere
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check that non-required entries are zero
    # This is expensive for full check, so we sample
    # Actually for exact decomposition with integer/half-integer entries,
    # if the required entries are exactly 1 and the decomposition is valid,
    # the others should be 0. But let's do a full check for correctness.
    # For 3x3x3, the tensor has 27*27*27 = 19683 entries, but we only need to check
    # entries that aren't of the form (i*m+j, j*p+k, k*n+i)
    
    # Actually, let's just check a subset for speed, or do full check
    # Full check:
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Check if this is a "required" position
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                
                is_required = (i_a == i_c) and (j_a == j_b)
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if is_required:
                    if abs(val - 1.0) > tol:
                        return False
                else:
                    if abs(val) > tol:
                        return False
    return True

def _is_integer_or_half_integer(arr, tol=1e-9):
    """Check if all entries are integer or half-integer (i.e., 2*x is integer)."""
    arr = np.array(arr, dtype=float)
    doubled = 2.0 * arr
    rounded = np.round(doubled)
    return np.all(np.abs(doubled - rounded) < tol)

def _round_to_half_integer(arr, tol=1e-6):
    """Round entries to nearest half-integer."""
    arr = np.array(arr, dtype=float)
    return np.round(2.0 * arr) / 2.0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with the known schoolbook rank-27 decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bounds: for 3x3x3, rank is at least 19 (Landsberg)
    # Upper bound: 23 (Strassen-like improvements)
    # The best known upper bound for 3x3 matrix multiplication is 23
    
    # Strategy: Use a structured search based on symmetry
    # The matmul tensor for 3x3 has S3 x S3 symmetry (permuting rows/cols of A, B, C)
    
    # Approach: Try rank 23 with a structured pattern
    # The rank-23 decomposition can be derived from the identity:
    # (a+b)(c+d) = ac + ad + bc + bd - (a-b)(c-d)/2 ... no, that's not quite right
    
    # Actually, for 3x3 matrix multiplication, we can use the following approach:
    # Group the 27 terms and find linear dependencies.
    
    # Let's try a direct approach: assume a specific structure and solve for coefficients
    # The key insight: we can write products like (A_ii + A_ij)(B_jj + B_jk) etc.
    
    # For now, let's implement a rank-23 construction based on known results.
    # The rank-23 decomposition for 3x3 matmul uses the following idea:
    # Instead of computing each C[i,k] = sum_j A[i,j]*B[j,k] separately,
    # we use shared intermediate computations.
    
    # One known approach uses the identity for 2x2 blocks and extends to 3x3.
    
    # Let me try a different approach: use the fact that we can compute
    # (x+y+z)(a+b+c) with fewer multiplications by exploiting linearity.
    
    # Actually, the simplest correct improvement is to verify if rank 23 is achievable
    # with a specific structured pattern.
    
    # Let's try: for each pair of rows in A and columns in B, we can share computations.
    
    # A concrete rank-23 construction:
    # We have 27 "naive" multiplications. We need to eliminate 4 of them.
    # Use identities like:
    # A[i,j]*B[j,k] = (A[i,j] + A[i,j+1]) * B[j,k] - A[i,j+1]*B[j,k]
    # This doesn't directly help unless we can share the (A[i,j]+A[i,j+1])*B[j,k] term.
    
    # Let's try a more systematic approach: use random initialization and gradient-based
    # optimization to find a low-rank decomposition.
    
    # But first, let's try a known structured rank-23 decomposition.
    
    # The following is a valid rank-23 decomposition based on the work of:
    # "The rank of the matrix multiplication tensor" by Landsberg (2012)
    
    # Actually, let me just try to optimize from the schoolbook decomposition
    # by merging some terms.
    
    # Alternative: use the known fact that rank(3x3 matmul) <= 23 and construct it.
    
    # Here's a concrete construction:
    # We compute all 27 products but express 4 of them as linear combinations.
    # 
    # For C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # For C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # etc.
    #
    # Key idea: A[0,0]*B[0,0] appears in C[0,0], A[0,1]*B[0,0] doesn't appear directly...
    # 
    # Let me think differently. The tensor T has 27 nonzero entries (one for each (i,j,k)).
    # We want to write T = sum_{r=1}^{R} u_r * v_r * w_r where each (u_r, v_r, w_r)
    # is a rank-1 tensor.
    
    # For the schoolbook, each term is exactly one of the 27 basis tensors.
    # To reduce rank, we need some u_r, v_r, w_r to be superpositions.
    
    # Let's try: u_r = e_{i,j} + e_{i',j'} (sum of two basis vectors in A-space)
    # v_r = e_{j,k} + e_{j',k'} (sum of two basis vectors in B-space)
    # w_r = e_{k,i} + e_{k',i'} (sum of two basis vectors in C-space)
    # 
    # Then u_r * v_r * w_r = (e_{i,j} + e_{i',j'})(e_{j,k} + e_{j',k'})(e_{k,i} + e_{k',i'})
    # = e_{i,j}e_{j,k}e_{k,i} + e_{i,j}e_{j',k'}e_{k,i} + e_{i',j'}e_{j,k}e_{k',i'} + e_{i',j'}e_{j',k'}e_{k',i'}
    #
    # The first and last terms are "correct" tensor entries if the indices match up.
    # The cross terms are spurious and need to cancel with other terms.
    
    # This is getting complex. Let me just do a numerical optimization.
    
    # Try rank 23:
    for target_rank in [23, 24, 25, 26]:
        if _remaining_time() < 10:
            break
            
        # Initialize from schoolbook and try to merge terms
        # Start with rank 27 and try to eliminate terms
        
        # Use a structured initialization
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # First target_rank - 4 terms are the "main" diagonal terms
        # Last 4 terms are "correction" terms
        
        # Main terms: for each (i,j,k) where i==j==k (diagonal)
        # Actually, let's use a different initialization
        
        # Initialize with random signs and small perturbations
        # to break symmetry
        
        # Use a structured initialization based on the 27 terms
        # Assign 23 of the 27 terms as "main" and express 4 as combinations
        
        # Let's use the following strategy:
        # Terms 0-22 are the first 23 of the 27 schoolbook terms
        # Terms 23-26 (4 missing) are expressed via the correction terms
        
        # Actually, this is hard to do analytically. Let's use numerical optimization.
        
        # Initialize U, V, W with the schoolbook pattern for the first min(27, target_rank) terms
        num_init = min(27, target_rank)
        for idx in range(num_init):
            i = idx // (m * p)
            jk = idx % (m * p)
            j = jk // p
            k = jk % p
            U[i * m + j, idx] = 1
            V[j * p + k, idx] = 1
            W[k * n + i, idx] = 1
        
        # For the remaining terms (if target_rank < 27), initialize with small random values
        for r in range(num_init, target_rank):
            U[:, r] = np.random.randn(n * m) * 0.1
            V[:, r] = np.random.randn(m * p) * 0.1
            W[:, r] = np.random.randn(p * n) * 0.1
        
        # Optimize using gradient descent on the loss:
        # L = sum over all (a,b,c) of (T_reconstructed[a,b,c] - T_target[a,b,c])^2
        
        # Precompute target tensor
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1.0
        
        # Gradient descent
        lr = 0.01
        patience = 2000
        best_loss = np.inf
        
        for step in range(patience):
            if _remaining_time() < 5:
                break
            
            # Compute reconstructed tensor
            # T_rec[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For efficiency, compute as: T_rec = sum_r outer(U[:,r], outer(V[:,r], W[:,r]))
            
            T_rec = np.zeros((n * m, m * p, p * n))
            for r in range(target_rank):
                T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            loss = np.sum((T_rec - T_target) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if best_loss < 1e-10:
                break
            
            # Compute gradients
            # dL/dU[a,r] = 2 * sum_{b,c} (T_rec[a,b,c] - T_target[a,b,c]) * V[b,r] * W[c,r]
            # dL/dV[b,r] = 2 * sum_{a,c} (T_rec[a,b,c] - T_target[a,b,c]) * U[a,r] * W[c,r]
            # dL/dW[c,r] = 2 * sum_{a,b} (T_rec[a,b,c] - T_target[a,b,c]) * U[a,r] * V[b,r]
            
            diff = T_rec - T_target
            
            dU = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
            dV = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
            dW = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Adaptive learning rate
            if loss > best_loss * 1.1:
                lr *= 0.5
            elif loss < best_loss * 0.9:
                lr = min(lr * 1.1, 0.1)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Check if we found a valid decomposition
        if best_loss < 1e-8:
            # Round to half-integers
            U_rounded = _round_to_half_integer(best_U)
            V_rounded = _round_to_half_integer(best_V)
            W_rounded = _round_to_half_integer(best_W)
            
            if _is_valid_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                best_rank = target_rank
                break
        
        # Also try with the unrounded version
        if _is_valid_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-8):
            if _is_integer_or_half_integer(best_U) and _is_integer_or_half_integer(best_V) and _is_integer_or_half_integer(best_W):
                best_U = _round_to_half_integer(best_U)
                best_V = _round_to_half_integer(best_V)
                best_W = _round_to_half_integer(best_W)
                best_rank = target_rank
                break
    
    # If we found a better decomposition, use it
    if best_rank < n * m * p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
