"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # First, build the known rank-44 decomposition
    rank_44 = _build_rank_44(n, m, p)
    
    # Try to improve via random search for lower rank
    best_rank = 44
    best_decomp = rank_44
    
    # Try ranks below 44
    for target_rank in [43, 42, 41, 40]:
        if time.time() > deadline - 30:
            break
        
        success = _try_random_search(n, m, p, target_rank, deadline)
        if success:
            best_rank = target_rank
            best_decomp = success
            break
    
    return best_decomp

def _build_rank_44(n, m, p):
    """Build the known rank-44 decomposition using Strassen blocks."""
    total_rank = 44
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def _add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # Strassen 2x2x2 decomposition
        # m1: (a+d)(e+h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m2: (c+d)e
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1
        
        # m3: a(f-h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m4: d(g-e)
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += -1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        col += 1
        
        # m5: (a+b)h
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        col += 1
        
        # m6: (c-a)(e+f)
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+0)][col] += -1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m7: (b-d)(g+h)
        U[(i_off+0)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        col += 1

    def _add_schoolbook_2x1x4(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + (j_off+0)][col] = 1
                V[(j_off+0)*p + (k_off+k)][col] = 1
                W[(k_off+k)*n + (i_off+i)][col] = 1
                col += 1

    # Block 1: k in 0..3
    _add_strassen_block(0, 0, 0)
    _add_strassen_block(0, 0, 2)
    _add_schoolbook_2x1x4(0, 2, 0)

    # Block 2: k in 4..7
    _add_strassen_block(0, 0, 4)
    _add_strassen_block(0, 0, 6)
    _add_schoolbook_2x1x4(0, 2, 4)

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    # Convert to numpy arrays
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Build the multiplication tensor
    # T[i,j,k] = 1 if this is a valid multiplication term
    # The tensor represents: C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For each rank component r, we have:
    # U[i,j,r] * V[j,k,r] * W[k,i,r] should sum to the multiplication tensor
    
    # Let's verify by checking that the decomposition reproduces matrix multiplication
    # Create random test matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C using the decomposition
    C_computed = np.zeros((n, p))
    for r in range(rank):
        # U[:, :, r] is n*m x 1, reshape to n x m
        U_r = U_arr[:, r].reshape(n, m)
        # V[:, :, r] is m*p x 1, reshape to m x p
        V_r = V_arr[:, r].reshape(m, p)
        # W[:, :, r] is p*n x 1, reshape to p x n
        W_r = W_arr[:, r].reshape(p, n)
        
        # The decomposition should give: C += U_r * V_r * W_r (element-wise product summed)
        # Actually, the correct interpretation is:
        # C[i,k] += sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But this is not quite right. Let me think about this more carefully.
        
        # The tensor T has indices (i, j, k) where T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # But W has index (k*n+i), so W[k*n+i, r]
        
        # Let me just verify by direct computation
        pass
    
    # Direct verification: check that the tensor is correctly decomposed
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] = 1 for all valid (i,j,k) in the multiplication
    
    # Actually, the multiplication tensor has T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Let's verify a few random entries
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        
        # Expected value: 1 (since all terms are valid)
        expected = 1.0
        
        # Computed value
        computed = 0.0
        for r in range(rank):
            computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
        
        if abs(computed - expected) > 1e-10:
            return False
    
    return True

def _try_random_search(n, m, p, target_rank, deadline):
    """Try to find a decomposition of the given rank via random search."""
    # This is a placeholder - a proper random search would be complex
    # For now, return None to indicate failure
    return None

def _try_structured_improvement(n, m, p, deadline):
    """Try structured improvements to the rank-44 decomposition."""
    # For 2x3x8, we can try different block decompositions
    # Current: 2 Strassen blocks (7 each) + 2 schoolbook 2x1x4 (8 each) = 14+16 = 30? No...
    # Actually current is: 2*(7+7+8) = 2*22 = 44
    
    # Let's try: Can we use a 2x2x4 block more efficiently?
    # A 2x2x4 matrix multiplication can be done in rank 16 (schoolbook) 
    # or potentially lower with Strassen-like methods.
    
    # For 2x2x4: we can split into two 2x2x2 blocks, each rank 7, total 14
    # So 2x2x4 has rank 14 (using two Strassen blocks)
    
    # For 2x3x8: split 3 into 2+1, split 8 into 4+4
    # Block 1: 2x2x4 (rank 14 using Strassen)
    # Block 2: 2x1x4 (rank 8 schoolbook)
    # Block 3: 2x2x4 (rank 14 using Strassen) 
    # Block 4: 2x1x4 (rank 8 schoolbook)
    # Total: 14+8+14+8 = 44 (same as current)
    
    # Alternative: Can we do 2x3x4 better?
    # 2x3x4: split 3 into 2+1
    # 2x2x4 (rank 14) + 2x1x4 (rank 8) = 22
    # So 2x3x8 = 2 * 22 = 44
    
    # Let's try a different approach: use the fact that we have 3 columns
    # Maybe we can share some computations across the 3 columns?
    
    # For now, let's just verify the current decomposition is valid and return it
    return None

# EVOLVE-BLOCK-END
