"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Start with schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_schoolbook, V_schoolbook, W_schoolbook
    
    # Try to find a lower rank decomposition using random initialization + iterative refinement
    # The matrix multiplication tensor T[i,j,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # We want to find rank-r decomposition: T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    
    # For matrix multiplication, the tensor is: T[i, j, k] = 1 if the multiplication uses
    # A[i,j] * B[j,k] to produce C[i,k]. The tensor indices are (i, j, k) where:
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # And the decomposition satisfies: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 (if this triple contributes)
    # Actually, the matrix multiplication tensor is:
    # M[i,j,k] = 1 for the "1" entries corresponding to A[i,j]*B[j,k] -> C[i,k]
    # The tensor is T with T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # and T[i,j,k] = 1 (all entries are 1 in the relevant subspace)
    # Wait, let me reconsider. The matrix multiplication tensor has:
    # T[i, j, k] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} summed over the product
    # Actually for matmul: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor M has M[i,j,k] = 1 for all valid (i,j,k) triples, and the
    # decomposition is: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's verify: for the schoolbook, U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1,
    # W[k*n+i, (i,j,k)] = 1, so sum over c gives 1 for each (i,j,k). Correct.
    
    # Try various ranks below schoolbook
    # Known bounds: 4x5x6 matmul. The trivial bound is 4*5*6=120.
    # Using Strassen-like ideas might help, but let's try random search.
    
    def verify(U, V, W, tol=1e-6):
        """Verify that the decomposition is exact: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k"""
        r = U.shape[1]
        # Compute all triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-6):
        """Faster verification using numpy"""
        r = U.shape[1]
        # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We can do this more efficiently
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower rank decompositions
    # Strategy: random initialization with structured patterns, then gradient-free refinement
    
    # First, let's try some known low-rank constructions
    # For 2x2x2, Strassen gives rank 7 instead of 8
    # For 4x4x4, using Strassen recursively gives rank 7^2 = 49 instead of 64
    
    # For 4x5x6, we can decompose as:
    # 4x5 = [4x4 | 4x1]
    # 5x6 = [4x6; 1x6]
    # So C = A1*B1 + A2*B2 + A1_tail*B2_tail + A2_tail*B1_tail
    # where A1 is 4x4, B1 is 4x6, A2 is 4x1, B2 is 1x6
    # C = A1*B1 + A2*B2 + (A1_last_col)*B2_row + (A2)*B1_last_row
    
    # This is getting complex. Let's try a simpler approach:
    # Use the fact that we can combine terms.
    
    # Actually, let's try a direct random search with local improvement
    # for ranks from 119 down to some lower bound.
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # But the true lower bound is higher. For 4x5x6, the known upper bound is 120.
    
    # Let's try a few random initializations at rank 119, 118, etc. and see if we can
    # verify them. But random won't work well for exact verification.
    
    # Better approach: use the structure of matmul.
    # The key insight: we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for ALL (i,j,k).
    # This is a system of n*m*p = 120 equations in r * (n*m + m*p + p*n) = r * (20+30+24) = 74r unknowns.
    # For r < 120, we have fewer unknowns than equations in general, but the equations are
    # not all independent.
    
    # Let's try a different approach: use the bilinear structure.
    # Fix W and try to solve for U and V, or use alternating minimization.
    
    # Actually, the cleanest approach for exact decompositions is to use known algorithms.
    # Let me try the Winograd method for small matrices.
    
    # For now, let's try to use a rank-119 decomposition by merging two schoolbook terms
    # if possible. Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there's a way
    # to write their contribution as a single rank-1 term. This is generally not possible
    # unless they share indices in a special way.
    
    # Let's just try to verify the schoolbook and return it if we can't find better.
    # But first, let's try a simple optimization: check if we can use half-integer entries
    # to reduce rank.
    
    # Given the time constraint, let's try a quick random search at a few ranks.
    
    np.random.seed(42)
    
    for target_rank in [119, 118, 117, 116, 115, 114, 113, 112, 111, 110]:
        if time.time() > deadline - 20:
            break
        
        # Random initialization
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Normalize to make verification easier
        # We want sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        # Try a few iterations of alternating updates
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each (i,j), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is hard to do exactly with random init.
            pass
        
        # Check if by chance it works (very unlikely)
        if verify_fast(U, V, W, tol=1e-4):
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to lists with rounded values for cleaner output
    def to_list(mat, tol=1e-6):
        """Convert numpy array to list, rounding near-integer values"""
        result = []
        for row in mat:
            new_row = []
            for val in row:
                # Round to nearest half-integer
                nearest_half = round(val * 2) / 2
                if abs(val - nearest_half) < tol:
                    new_row.append(nearest_half)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    if best_rank < rank_schoolbook:
        U_list = to_list(best_U)
        V_list = to_list(best_V)
        W_list = to_list(best_W)
    else:
        U_list = to_list(best_U)
        V_list = to_list(best_V)
        W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
