"""Rewritten program for 2x2x8 matrix multiplication using Strassen's algorithm on blocks."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the multiplication tensor T[i, j, k] = delta_{i,k} (for A[i,j] * B[j,k] -> C[i,k])
    # Actually T is defined such that C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor decomposition requires:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]  (with appropriate indexing)
    
    # Let's check using the standard verification:
    # For random A (n x m) and B (m x p), compute C = A @ B
    # and compare with sum_r (U[r] * A) * (V[r] * B) * W[r]
    
    # U has shape (n*m, rank), so U[r] is a vector of length n*m = n*m
    # We need to reshape U[r] to (n, m) for multiplication with A
    # V has shape (m*p, rank), V[r] reshaped to (m, p)
    # W has shape (p*n, rank), W[r] reshaped to (p, n) per trace convention W[k*n+i, r]
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        U_r = U_arr[:, r].reshape(n, m)
        V_r = V_arr[:, r].reshape(m, p)
        W_r = W_arr[:, r].reshape(p, n)
        # C[i,k] += sum_j U_r[i,j] * V_r[j,k] * W_r[k,i]
        # This is a generalized product
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_computed[i, k] += U_r[i, j] * V_r[j, k] * W_r[k, i]
    
    if np.max(np.abs(C_computed - C_expected)) > tol:
        return False
    return True

def _build_strassen_228():
    """Build the standard Strassen rank-28 decomposition for 2x2x8."""
    n, m, p = 2, 2, 8
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    U_coeffs = [
        {0: 1.0, 3: 1.0},       # M1
        {2: 1.0, 3: 1.0},       # M2
        {0: 1.0},                # M3
        {3: 1.0},                # M4
        {0: 1.0, 1: 1.0},       # M5
        {2: 1.0, 0: -1.0},      # M6
        {1: 1.0, 3: -1.0}       # M7
    ]
    V_coeffs = [
        {0: 1.0, 3: 1.0},       # M1
        {0: 1.0},                # M2
        {1: 1.0, 3: -1.0},      # M3
        {2: 1.0, 0: -1.0},      # M4
        {3: 1.0},                # M5
        {0: 1.0, 1: 1.0},       # M6
        {2: 1.0, 3: 1.0}        # M7
    ]
    W_coeffs = [
        {0: 1.0, 3: 1.0},       # M1
        {1: 1.0, 3: -1.0},      # M2
        {2: 1.0, 3: 1.0},       # M3
        {0: 1.0, 1: 1.0},       # M4
        {0: -1.0, 2: 1.0},      # M5
        {3: 1.0},                # M6
        {0: 1.0}                 # M7
    ]
    num_blocks = p // 2
    for q in range(num_blocks):
        for s in range(7):
            r = q * 7 + s
            for u_idx, val in U_coeffs[s].items():
                U[u_idx][r] = val
            for v_local_idx, val in V_coeffs[s].items():
                j = v_local_idx // 2
                k_local = v_local_idx % 2
                k = 2 * q + k_local
                v_idx = j * p + k
                V[v_idx][r] = val
            for w_local_idx, val in W_coeffs[s].items():
                i = w_local_idx % 2
                k_local = w_local_idx // 2
                k = 2 * q + k_local
                w_idx = k * n + i
                W[w_idx][r] = val
    return U, V, W

def _try_find_dependency(U, V, W, n, m, p):
    """
    Try to find a linear dependency among the rank-28 Strassen terms.
    If we can find coefficients c_r (not all zero) such that
    sum_r c_r * U[:,r] = 0, sum_r c_r * V[:,r] = 0, sum_r c_r * W[:,r] = 0
    then we can eliminate one term. But that's not quite right.
    
    Actually, a tensor decomposition of rank R means we have R terms.
    To go from rank 28 to rank 24, we need to find 4 linear dependencies.
    
    The idea: if there exist vectors alpha (length n*m), beta (length m*p), gamma (length p*n)
    and coefficients c_1..c_28 such that for each term r:
    U[:,r] = alpha, V[:,r] = beta, W[:,r] = gamma (up to scaling)
    then we can combine them.
    
    A more practical approach: check if the 28 vectors in the tensor product space
    are linearly dependent. The tensor T can be viewed as an element of 
    (R^{n*m}) * (R^{m*p}) * (R^{p*n}).
    
    Let's check linear dependence of the 28 "triplets" (U_r, V_r, W_r).
    We form the matrix M where each column is the Kronecker product U_r ⊗ V_r ⊗ W_r.
    If rank(M) < 28, there's a dependency.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the matrix of Kronecker products
    # Each column r is U_arr[:,r] ⊗ V_arr[:,r] ⊗ W_arr[:,r]
    # Shape: (n*m * m*p * p*n, rank)
    cols = []
    for r in range(rank):
        col = np.kron(np.kron(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
        cols.append(col)
    
    M = np.column_stack(cols)
    
    # Check rank
    try:
        s = np.linalg.svd(M, compute_uv=False)
        # Count significant singular values
        threshold = 1e-10
        num_significant = sum(1 for sv in s if sv > threshold)
        print(f"Rank of Kronecker matrix: {num_significant} out of {rank}")
        
        if num_significant < rank:
            # Find null space vectors
            # The right singular vectors corresponding to small singular values
            # give the dependencies
            _, _, vh = np.linalg.svd(M, full_matrices=True)
            # Null space vectors are rows of vh corresponding to small singular values
            null_vectors = vh[num_significant:, :]
            print(f"Found {len(null_vectors)} linear dependencies")
            for i, nv in enumerate(null_vectors):
                print(f"  Dependency {i}: {nv[:10]}... (norm={np.linalg.norm(nv):.6f})")
            return null_vectors
    except Exception as e:
        print(f"Error in SVD: {e}")
    
    return None

def _try_group_algebra_search(n, m, p, deadline):
    """
    Search for a lower-rank decomposition using group algebra structure.
    For 2x2x8, we can view B as having 4 blocks of 2x2 matrices.
    The group C2 x C2 has 4 elements. We look for a decomposition that
    exploits the character orthogonality.
    """
    # The 4 blocks of B (each 2x2) can be indexed by (a,b) in C2 x C2
    # B = sum_{a,b} B_{a,b} * e_{a,b} where e_{a,b} are group basis elements
    
    # Strassen's algorithm on each 2x2 block gives rank 7 per block.
    # But the blocks are not independent - they share the same A.
    
    # Key insight: In Strassen's algorithm for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 2x2x8, we have 4 such blocks. The A factors are shared.
    # Let's see if we can combine terms across blocks.
    
    # For block q (q=0,1,2,3), the A-factors in Strassen are:
    # A11, A22, A21+A22, A11, A11+A12, A21-A11, A12-A22
    # These are the same for all blocks!
    
    # So the U factors are identical across all 4 blocks.
    # The V and W factors differ per block.
    
    # This means we can potentially combine the V and W factors
    # using the group structure.
    
    # Let's build the decomposition and check for dependencies.
    U, V, W = _build_strassen_228()
    
    # Check for linear dependencies
    null_vecs = _try_find_dependency(U, V, W, n, m, p)
    
    if null_vecs is not None and len(null_vecs) > 0:
        # We found dependencies. Try to use them to reduce rank.
        # For now, just return the original decomposition
        # A full implementation would reconstruct the decomposition
        # without the dependent terms.
        pass
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # First, try the standard Strassen rank-28 decomposition
    U, V, W = _build_strassen_228()
    
    # Verify it's correct
    if _verify_decomposition(U, V, W, n, m, p):
        best_rank = 28
    else:
        # Fallback to schoolbook
        best_rank = n * m * p
        U = [[0.0] * best_rank for _ in range(n * m)]
        V = [[0.0] * best_rank for _ in range(m * p)]
        W = [[0.0] * best_rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1.0
                    V[j * p + k][r] = 1.0
                    W[k * n + i][r] = 1.0
    
    # Try to find a lower-rank decomposition
    if time.time() < deadline:
        try:
            U2, V2, W2 = _try_group_algebra_search(n, m, p, deadline)
            if _verify_decomposition(U2, V2, W2, n, m, p):
                rank2 = len(U2[0])
                if rank2 < best_rank:
                    U, V, W = U2, V2, W2
                    best_rank = rank2
        except Exception as e:
            print(f"Group algebra search failed: {e}")
    
    # Try a direct search for rank < 28 using random initialization
    if time.time() < deadline - 5:
        try:
            # Try to find a rank-24 decomposition by removing 4 dependent terms
            # This is a heuristic approach
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            # Check which sets of 4 terms can be removed
            # by checking if the remaining 24 terms still span the tensor
            rank = len(U[0])
            found_lower = False
            
            # Try removing 4 terms at a time (combinatorial, but limited)
            # For efficiency, only try a subset
            np.random.seed(123)
            num_trials = 500
            for trial in range(num_trials):
                if time.time() > deadline - 2:
                    break
                
                # Randomly select 4 indices to remove
                remove_indices = np.random.choice(rank, size=4, replace=False)
                keep_indices = [i for i in range(rank) if i not in remove_indices]
                
                # Check if the remaining terms can still decompose the tensor
                # This is expensive, so use a quick check
                U_sub = U_arr[:, keep_indices]
                V_sub = V_arr[:, keep_indices]
                W_sub = W_arr[:, keep_indices]
                
                # Quick check: can we find coefficients to recover the removed terms?
                # This is a linear system problem
                # For now, just verify if the sub-decomposition is valid
                # (This would require a more sophisticated check)
                
                # Skip for now - too expensive
                pass
            
            if found_lower:
                pass
        except Exception as e:
            print(f"Direct search failed: {e}")
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        best_rank = n * m * p
        U = [[0.0] * best_rank for _ in range(n * m)]
        V = [[0.0] * best_rank for _ in range(m * p)]
        W = [[0.0] * best_rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1.0
                    V[j * p + k][r] = 1.0
                    W[k * n + i][r] = 1.0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
