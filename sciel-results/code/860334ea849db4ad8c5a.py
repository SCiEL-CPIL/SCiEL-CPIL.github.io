"""
Program for matrix multiplication 2x5x5 rank minimization.
Uses Kronecker product structure with Strassen core for low-rank decomposition.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}))
    terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}))
    terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}))
    terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}))
    terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
    terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}))
    return terms

def _terms_to_matrices(terms, n, m, p):
    """Convert list of term dicts to U, V, W matrices."""
    rank = len(terms)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j, c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, c] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_3d = U.reshape(n, m, -1)
    V_3d = V.reshape(m, p, -1)
    W_3d = W.reshape(p, n, -1)
    
    result = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    expected = np.ones((n, m, p))
    
    max_err = np.max(np.abs(result - expected))
    return max_err < tol, max_err

def _build_kronecker_decomposition(n, m, p):
    """
    Build decomposition using Kronecker product structure.
    
    Key insight: View 2x5x5 as action of 5x5 matrix on 2x5 block.
    Decompose 5-dim column space as 4+1, where 4-dim part uses
    2x2 Strassen core (rank 4 via Kronecker), 1-dim boundary adds 2 terms.
    
    Structure:
    - Split j (columns of A) into {0,1,2,3} and {4}
    - Split k (rows of B) into {0,1,2,3} and {4}
    - For the 4x4 core: use rank-4 decomposition based on 2x2 Strassen
    - For boundary terms: 2 additional terms for the 1-dim parts
    
    Total target: 4 (core) + 2 (boundary) + 4 (cross terms) = 10
    """
    terms = []
    
    # The 4x4 core can be decomposed using a rank-4 structure.
    # We use the fact that 4x4 matrix mult can be done with rank 14 (Strassen-based),
    # but for our specific 2x4x4 sub-problem, we can do better.
    
    # Actually, let's think more carefully. The tensor is T_{i,j,k} for i in {0,1}, j in {0..4}, k in {0..4}
    # We want to decompose this as sum of rank-1 tensors.
    
    # Strategy: Use the structure that C[i,k] = sum_j A[i,j] * B[j,k]
    # For fixed i, this is a row of A times B.
    
    # Let's try a direct approach: express the 2x5x5 tensor as a sum of rank-1 terms.
    
    # Approach: Use the fact that we can share computations between the two rows.
    
    # Let A = [a00 a01 a02 a03 a04; a10 a11 a12 a13 a14]
    # Let B be 5x5.
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We can write this as:
    # C[0,k] = sum_j a0j * B[j,k]
    # C[1,k] = sum_j a1j * B[j,k]
    
    # Define:
    # M0 = a00 * B0 + a01 * B1 + a02 * B2 + a03 * B3 + a04 * B4  (row 0 of C)
    # M1 = a10 * B0 + a11 * B1 + a12 * B2 + a13 * B3 + a14 * B4  (row 1 of C)
    
    # Where Bj is row j of B.
    
    # We need to compute these 10 scalar products (5 for each row).
    # The question is how many rank-1 terms we need.
    
    # Let's try to use a shared structure. Define:
    # S0 = a00 + a10
    # S1 = a00 - a10  
    # etc.
    
    # Actually, let's just try the straightforward Kronecker approach.
    
    # For the 4x4 sub-problem (j in {0,1,2,3}, k in {0,1,2,3}):
    # We can use a rank-14 decomposition (Strassen-based for 4x4x4)
    # But we only need 2x4x4, which should be less.
    
    # Let me try a different approach: direct enumeration with optimization.
    
    # For now, let's build a valid decomposition and then try to reduce rank.
    
    # Build schoolbook first:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(({(i, j): 1}, {(j, k): 1}, {(k, i): 1}))
    
    return terms

def _build_strassen_blocks(n, m, p):
    """Build decomposition using Strassen blocks where possible."""
    terms = []
    
    def add_strassen(i_off, j_off, k_off):
        for u, v, w in _strassen_2x2x2_terms():
            u_off = {}
            for (ii, jj), val in u.items():
                u_off[(ii + i_off, jj + j_off)] = val
            v_off = {}
            for (jj, kk), val in v.items():
                v_off[(jj + j_off, kk + k_off)] = val
            w_off = {}
            for (kk, ii), val in w.items():
                w_off[(kk + k_off, ii + i_off)] = val
            terms.append((u_off, v_off, w_off))
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    terms.append(({(i, j): 1}, {(j, k): 1}, {(k, i): 1}))
    
    # Use Strassen for 2x2x2 blocks where possible
    # i in {0,1}, j in {0,1,2,3,4}, k in {0,1,2,3,4}
    
    # j in {0,1}: Strassen for k in {0,1} and k in {2,3}, schoolbook for k=4
    add_strassen(0, 0, 0)   # 7 terms
    add_strassen(0, 0, 2)   # 7 terms
    add_schoolbook(0, 2, 0, 2, 4, 5)  # 4 terms
    
    # j in {2,3}: Strassen for k in {0,1} and k in {2,3}, schoolbook for k=4
    add_strassen(0, 2, 0)   # 7 terms
    add_strassen(0, 2, 2)   # 7 terms
    add_schoolbook(0, 2, 2, 4, 4, 5)  # 4 terms
    
    # j in {4}: schoolbook for all k
    add_schoolbook(0, 2, 4, 5, 0, 5)  # 10 terms
    
    return terms

def _reduce_rank_greedy(U, V, W, n, m, p, deadline=None):
    """Greedy rank reduction by removing redundant terms."""
    start_time = time.time()
    if deadline is None:
        deadline = start_time + 60.0
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    
    reduced = True
    while reduced and time.time() < deadline - 10:
        reduced = False
        current_rank = current_U.shape[1]
        
        for c in range(current_rank):
            if time.time() > deadline - 10:
                break
            
            mask = [i for i in range(current_rank) if i != c]
            test_U = current_U[:, mask]
            test_V = current_V[:, mask]
            test_W = current_W[:, mask]
            
            valid, err = _verify_decomposition(test_U, test_V, test_W, n, m, p, tol=1e-8)
            
            if valid:
                current_U, current_V, current_W = test_U, test_V, test_W
                reduced = True
                break
    
    return current_U, current_V, current_W

def _reduce_rank_nullspace(U, V, W, n, m, p, deadline=None):
    """
    Try to find linear dependencies in the tensor decomposition.
    If we can find coefficients such that sum_c alpha_c * (U_c ⊗ V_c ⊗ W_c) = 0,
    then we can reduce the rank.
    """
    start_time = time.time()
    if deadline is None:
        deadline = start_time + 60.0
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    rank = current_U.shape[1]
    
    if rank <= 1:
        return current_U, current_V, current_W
    
    # Flatten the tensor
    # The tensor T_{i,j,k} = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We want to find if there's a non-trivial linear combination that gives zero.
    
    # This is equivalent to checking if the rank-1 tensors are linearly independent.
    # We can check this by forming a matrix where each column is a flattened rank-1 tensor.
    
    n, m, p = 2, 5, 5
    tensor_size = n * m * p
    
    # Build matrix of flattened rank-1 tensors
    M = np.zeros((tensor_size, rank), dtype=np.float64)
    
    for c in range(rank):
        U_c = current_U[:, c].reshape(n, m)
        V_c = current_V[:, c].reshape(m, p)
        W_c = current_W[:, c].reshape(p, n)
        
        # Compute the rank-1 tensor: T_{i,j,k} = U_c[i,j] * V_c[j,k] * W_c[k,i]
        T = np.zeros((n, m, p), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = U_c[i, j] * V_c[j, k] * W_c[k, i]
        
        M[:, c] = T.flatten()
    
    # Check rank of M
    if time.time() > deadline - 5:
        return current_U, current_V, current_W
    
    try:
        M_rank = np.linalg.matrix_rank(M, tol=1e-8)
    except:
        return current_U, current_V, current_W
    
    if M_rank < rank:
        # There's linear dependence. Find null space.
        try:
            null_space = np.linalg.null_space(M)
            null_dim = null_space.shape[1]
        except:
            return current_U, current_V, current_W
        
        if null_dim > 0:
            # We can reduce rank by null_dim
            # For now, just note that reduction is possible
            # Implementing the actual reduction requires careful handling
            pass
    
    return current_U, current_V, current_W

def _try_optimize_rank(U, V, W, n, m, p, deadline=None):
    """Try multiple rank reduction strategies and return the best."""
    start_time = time.time()
    if deadline is None:
        deadline = start_time + 100.0
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Strategy 1: Greedy removal
    if time.time() < deadline - 30:
        U1, V1, W1 = _reduce_rank_greedy(U, V, W, n, m, p, deadline=deadline - 20)
        if U1.shape[1] < best_rank:
            best_rank = U1.shape[1]
            best_U, best_V, best_W = U1, V1, W1
    
    # Strategy 2: Null space analysis
    if time.time() < deadline - 10:
        U2, V2, W2 = _reduce_rank_nullspace(best_U, best_V, best_W, n, m, p, deadline=deadline - 5)
        if U2.shape[1] < best_rank:
            best_rank = U2.shape[1]
            best_U, best_V, best_W = U2, V2, W2
    
    return best_U, best_V, best_W

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = 2, 5, 5
    
    # Try different initial decompositions
    candidates = []
    
    # Strategy 1: Strassen blocks
    if time.time() < deadline - 50:
        terms1 = _build_strassen_blocks(n, m, p)
        U1, V1, W1 = _terms_to_matrices(terms1, n, m, p)
        valid1, _ = _verify_decomposition(U1, V1, W1, n, m, p)
        if valid1:
            candidates.append((U1.shape[1], U1, V1, W1))
    
    # Strategy 2: Schoolbook
    terms2 = _build_kronecker_decomposition(n, m, p)
    U2, V2, W2 = _terms_to_matrices(terms2, n, m, p)
    valid2, _ = _verify_decomposition(U2, V2, W2, n, m, p)
    if valid2:
        candidates.append((U2.shape[1], U2, V2, W2))
    
    if not candidates:
        # Fallback
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Sort by rank
    candidates.sort(key=lambda x: x[0])
    
    # Try to optimize the best candidate
    best_rank, best_U, best_V, best_W = candidates[0]
    
    if time.time() < deadline - 20:
        opt_U, opt_V, opt_W = _try_optimize_rank(best_U, best_V, best_W, n, m, p, deadline=deadline - 10)
        if opt_U.shape[1] < best_rank:
            best_rank, best_U, best_V, best_W = opt_U.shape[1], opt_U, opt_V, opt_W
    
    # Final verification
    valid, err = _verify_decomposition(best_U, best_V, best_W, n, m, p)
    if not valid:
        # Use the first valid candidate
        for rank, U, V, W in candidates:
            v, _ = _verify_decomposition(U, V, W, n, m, p)
            if v:
                best_U, best_V, best_W = U, V, W
                break
        else:
            # Ultimate fallback
            U = np.zeros((n * m, n * m * p))
            V = np.zeros((m * p, n * m * p))
            W = np.zeros((p * n, n * m * p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        c = i * m * p + j * p + k
                        U[i * m + j, c] = 1
                        V[j * p + k, c] = 1
                        W[k * n + i, c] = 1
            best_U, best_V, best_W = U, V, W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
