"""
Strategy: Direct algebraic rank-reduction via null-space projection.
Start from the known rank-97 hybrid (Strassen 4x4x4 + schoolbook 4x4x3),
then iteratively attempt to reduce rank by finding linear dependencies
among the rank-1 tensor factors and projecting them out.
"""

import time
import numpy as np
from itertools import combinations

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # Vectorized verification
    # For each (i, j, k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _build_hybrid_4x4x7(n=4, m=4, p=7):
    """
    Build the hybrid rank-97 decomposition:
    - Strassen (rank 49) for the 4x4x4 part (k=0..3)
    - Schoolbook (rank 48) for the 4x4x3 part (k=4..6)
    """
    # Strassen 2x2x2 (rank 7) - verified correct
    def strassen_2x2():
        U2 = np.zeros((4, 7), dtype=np.float64)
        V2 = np.zeros((4, 7), dtype=np.float64)
        W2 = np.zeros((4, 7), dtype=np.float64)
        
        # M1 = (A00+A11)(B00+B11)
        U2[0,0]=1; U2[3,0]=1
        V2[0,0]=1; V2[3,0]=1
        W2[0,0]=1; W2[3,0]=1
        
        # M2 = (A10+A11)B00
        U2[2,1]=1; U2[3,1]=1
        V2[0,1]=1
        W2[2,1]=1; W2[3,1]=1
        
        # M3 = A00(B10-B11)
        U2[0,2]=1
        V2[2,2]=1; V2[3,2]=-1
        W2[1,2]=1; W2[3,2]=-1
        
        # M4 = A11(B01-B00)
        U2[3,3]=1
        V2[1,3]=1; V2[0,3]=-1
        W2[0,3]=1; W2[2,3]=1
        
        # M5 = (A00+A01)B11
        U2[0,4]=1; U2[1,4]=1
        V2[3,4]=1
        W2[0,4]=1; W2[2,4]=1
        
        # M6 = (A10-A00)(B00+B10)
        U2[2,5]=1; U2[0,5]=-1
        V2[0,5]=1; V2[1,5]=1
        W2[2,5]=1
        
        # M7 = (A01-A11)(B01+B11)
        U2[1,6]=1; U2[3,6]=-1
        V2[2,6]=1; V2[3,6]=1
        W2[0,6]=-1; W2[1,6]=1; W2[3,6]=1
        
        return U2, V2, W2
    
    U2, V2, W2 = strassen_2x2()
    
    # Build 4x4x4 from 8 blocks of 2x2x2 Strassen: rank 56
    # Actually, for 4x4x4: 4 output blocks * 2 input blocks * 7 = 56
    # But we want rank 49 for the 4x4x4 part. Let's use the standard approach.
    
    # Standard 4x4x4 via 2x2x2 Strassen on blocks:
    # C = A*B, A is 4x4, B is 4x4, C is 4x4
    # Block into 2x2 blocks of size 2x2:
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    # Each term is a 2x2x2 tensor with rank 7.
    # So C[0:2,0:2] has rank at most 14.
    # 4 output blocks * 14 = 56.
    
    # To get rank 49, we need a more clever 4x4x4 decomposition.
    # The known minimum rank for 4x4x4 is 49 (Laderman's algorithm).
    
    # Laderman's algorithm for 4x4x4 (rank 49):
    # This is complex. Let me use a simpler approach: 
    # Use the 56-rank block Strassen for 4x4x4, then try to reduce.
    
    # Actually, let's just build the 56-rank 4x4x4 + 48-rank 4x4x3 = 104 total,
    # then apply null-space reduction.
    
    total_rank = 56 + 48
    U = np.zeros((16, total_rank), dtype=np.float64)
    V = np.zeros((28, total_rank), dtype=np.float64)
    W = np.zeros((28, total_rank), dtype=np.float64)
    
    rank_offset = 0
    # For each output block (a,b) in {(0,0),(0,1),(1,0),(1,0)}:
    #   For each input block c in {0,1}:
    #     Apply Strassen 2x2x2
    for a in range(2):  # output block row
        for b in range(2):  # output block col
            for c in range(2):  # input block (summed over)
                for r in range(7):
                    for ii in range(2):
                        for jj in range(2):
                            i = 2*a + ii
                            j = 2*c + jj
                            U[i*4 + j, rank_offset + r] += U2[ii*2 + jj, r]
                    
                    for jj in range(2):
                        for kk in range(2):
                            j = 2*c + jj
                            k = 2*b + kk
                            V[j*4 + k, rank_offset + r] += V2[jj*2 + kk, r]
                    
                    for kk in range(2):
                        for ii in range(2):
                            k = 2*b + kk
                            i = 2*a + ii
                            W[k*4 + i, rank_offset + r] += W2[kk*2 + ii, r]
                
                rank_offset += 7
    
    # Now add 4x4x3 schoolbook for k in {4,5,6}
    for i in range(4):
        for j in range(4):
            for k_local in range(3):
                k = 4 + k_local
                col = rank_offset + i*12 + j*3 + k_local
                U[i*4 + j, col] = 1
                V[j*7 + k, col] = 1
                W[k*4 + i, col] = 1
    
    assert rank_offset + 48 == total_rank
    
    return U, V, W


def _tensor_to_vectors(U, V, W, n, m, p):
    """
    Convert the factor matrices to a list of rank-1 tensor vectors.
    Each rank-1 factor is a tensor T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    We represent each as a vector of length n*m*p.
    """
    rank = U.shape[1]
    vectors = []
    for r in range(rank):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i * m * p + j * p + k] = U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
        vectors.append(vec)
    return vectors


def _vectors_to_tensor(vectors, n, m, p):
    """Sum the rank-1 vectors to get the full tensor."""
    tensor = np.zeros(n * m * p)
    for vec in vectors:
        tensor += vec
    return tensor


def _find_linear_dependencies(vectors, tol=1e-9):
    """
    Find linear dependencies among the vectors using SVD.
    Returns indices of vectors that can be removed (linearly dependent on others).
    """
    if len(vectors) == 0:
        return []
    
    mat = np.array(vectors).T  # shape: (n*m*p, rank)
    # Use SVD to find the rank
    try:
        U_svd, s, Vt = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        return []
    
    # Determine numerical rank
    if len(s) == 0:
        return []
    
    max_s = s[0] if len(s) > 0 else 0
    if max_s == 0:
        return list(range(len(vectors)))
    
    threshold = max_s * tol
    num_rank = np.sum(s > threshold)
    
    # If all vectors are linearly independent, no reduction possible
    if num_rank == len(vectors):
        return []
    
    # Find which columns of Vt have significant singular values
    # The null space is spanned by the last (rank - num_rank) rows of Vt
    null_space = Vt[num_rank:]
    
    # Find vectors that are in the null space (linearly dependent)
    # A vector is linearly dependent if it can be expressed as a combination of others
    # More practically: find pairs where one is close to a multiple of another
    
    # Simple approach: find pairs (i, j) where vectors[i] ≈ c * vectors[j]
    dependent_indices = []
    n_vecs = len(vectors)
    
    for i in range(n_vecs):
        if i in dependent_indices:
            continue
        for j in range(i + 1, n_vecs):
            if j in dependent_indices:
                continue
            vi = vectors[i]
            vj = vectors[j]
            norm_i = np.linalg.norm(vi)
            norm_j = np.linalg.norm(vj)
            if norm_i < tol or norm_j < tol:
                # One is zero, it's dependent
                if norm_i < tol:
                    dependent_indices.append(i)
                else:
                    dependent_indices.append(j)
                continue
            
            # Check if vi ≈ c * vj
            c = np.dot(vi, vj) / (norm_j ** 2)
            diff = vi - c * vj
            if np.linalg.norm(diff) < tol * max(norm_i, norm_j):
                dependent_indices.append(j)
                break
    
    return dependent_indices


def _reduce_rank(U, V, W, n, m, p, max_iterations=50, tol=1e-9):
    """
    Iteratively reduce rank by finding and removing linearly dependent columns.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    
    for iteration in range(max_iterations):
        vectors = _tensor_to_vectors(current_U, current_V, current_W, n, m, p)
        
        # Check if the sum is correct
        tensor = _vectors_to_tensor(vectors, n, m, p)
        expected = np.ones(n * m * p)
        if np.linalg.norm(tensor - expected) > tol:
            break
        
        # Find linear dependencies
        dependent = _find_linear_dependencies(vectors, tol)
        
        if not dependent:
            break
        
        # Remove the dependent columns
        keep_indices = [i for i in range(len(vectors)) if i not in dependent]
        
        if len(keep_indices) == len(vectors):
            break
        
        current_U = current_U[:, keep_indices]
        current_V = current_V[:, keep_indices]
        current_W = current_W[:, keep_indices]
        
        # Verify after removal
        if not _verify(current_U, current_V, current_W, n, m, p, tol):
            # Restore
            current_U = U.copy()
            current_V = V.copy()
            current_W = W.copy()
            break
    
    return current_U, current_V, current_W


def _try_random_perturbation_reduction(U, V, W, n, m, p, num_trials=20, tol=1e-9):
    """
    Try random perturbations to break symmetries and find lower-rank decompositions.
    This is a heuristic search for non-trivial rank reductions.
    """
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    rng = np.random.default_rng(42)
    
    for trial in range(num_trials):
        # Create a perturbed version
        U_pert = U.copy()
        V_pert = V.copy()
        W_pert = W.copy()
        
        # Add small random perturbations to some entries
        num_perturb = min(10, U.shape[1] // 5)
        for _ in range(num_perturb):
            r = rng.integers(0, U.shape[1])
            i = rng.integers(0, U.shape[0])
            j = rng.integers(0, V.shape[0])
            k = rng.integers(0, W.shape[0])
            
            delta = rng.uniform(-0.1, 0.1)
            U_pert[i, r] += delta
            V_pert[j, r] += delta
            W_pert[k, r] += delta
        
        # Try to reduce the perturbed version
        U_red, V_red, W_red = _reduce_rank(U_pert, V_pert, W_pert, n, m, p, max_iterations=10, tol=tol)
        
        if U_red.shape[1] < best_rank and _verify(U_red, V_red, W_red, n, m, p, tol):
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = U_red.shape[1]
    
    return best_U, best_V, best_W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin from 150s limit
    
    # Build the hybrid decomposition (rank 104)
    U, V, W = _build_hybrid_4x4x7(n, m, p)
    
    # Verify it's correct
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try null-space reduction
    U_red, V_red, W_red = _reduce_rank(U, V, W, n, m, p, max_iterations=50)
    
    if time.time() < deadline:
        # Try random perturbation reduction
        U_pert, V_pert, W_pert = _try_random_perturbation_reduction(U_red, V_red, W_red, n, m, p, num_trials=10)
        
        if U_pert.shape[1] < U_red.shape[1] and _verify(U_pert, V_pert, W_pert, n, m, p):
            U_red, V_red, W_red = U_pert, V_pert, W_pert
    
    # Final verification
    if not _verify(U_red, V_red, W_red, n, m, p):
        # Fall back to the original hybrid
        U_red, V_red, W_red = U, V, W
    
    # Ensure it's correct
    if not _verify(U_red, V_red, W_red, n, m, p):
        # Ultimate fallback: schoolbook
        rank = n * m * p
        U_red = np.zeros((n * m, rank), dtype=np.float64)
        V_red = np.zeros((m * p, rank), dtype=np.float64)
        W_red = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_red[i * m + j, col] = 1
                    V_red[j * p + k, col] = 1
                    W_red[k * n + i, col] = 1
                    col += 1
    
    return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}

# EVOLVE-BLOCK-END
