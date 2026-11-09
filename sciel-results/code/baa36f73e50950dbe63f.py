"""Rank-minimal tensor decomposition search for 3x4x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]
N, M, P = DIMENSIONS
A_DIM = N * M  # 12
B_DIM = M * P  # 24
C_DIM = P * N  # 18
TENSOR_SIZE = A_DIM * B_DIM * C_DIM  # 576
SCHOOLBOOK_RANK = N * M * P  # 72

def _build_tensor():
    """Build the matrix multiplication tensor T[a,b,c] = 1 iff a=M+i, b=M+j, c=i*N+k."""
    T = np.zeros((A_DIM, B_DIM, C_DIM), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i * M + j, j * P + k, k * N + i] = 1.0
    return T

def _verify(U, V, W):
    """Verify that sum_r U[:,r] V[:,r] W[:,r] == T (broadcast outer product sum)."""
    T = _build_tensor()
    # Sum over rank dimension: U[a,r]*V[b,r]*W[c,r]
    # Efficient: for each r, outer(U[:,r], V[:,r]) * W[:,r] broadcast
    # U shape (A, R), V shape (B, R), W shape (C, R)
    # result[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # = sum_r (U[:,r] outer V[:,r]) * W[:,r]
    # Use einsum for clarity
    R = U.shape[1]
    result = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(result, T, atol=1e-8, rtol=1e-10)

def _rank_of_matrix(M, tol=1e-10):
    """Numerical rank via SVD."""
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol * max(s[0], 1.0)))

def _random_decomposition(rank, rng, seed_offset=0):
    """Generate a random decomposition of the given rank."""
    s = seed_offset + rank
    rng = random.Random(s)
    U = np.array([[rng.gauss(0,1) for _ in range(rank)] for _ in range(A_DIM)])
    V = np.array([[rng.gauss(0,1) for _ in range(rank)] for _ in range(B_DIM)])
    W = np.array([[rng.gauss(0,1) for _ in range(rank)] for _ in range(C_DIM)])
    return U, V, W

def _check_decomposition(U, V, W):
    """Check if U,V,W exactly decompose the tensor."""
    T = _build_tensor()
    result = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(result, T, atol=1e-8, rtol=1e-10)

def _try_rank(rank, max_trials, rng, deadline):
    """Try to find a valid decomposition of the given rank."""
    for trial in range(max_trials):
        if time.time() > deadline:
            return None
        U, V, W = _random_decomposition(rank, rng, trial)
        if _check_decomposition(U, V, W):
            return U, V, W
    return None

def _structured_search(rank, max_trials, rng, deadline):
    """
    Structured search: try decompositions where the tensor entries are 
    from a restricted set {0, ±1, ±1/2} which are more likely to be exact.
    """
    # The matrix multiplication tensor has entries 0 or 1.
    # A rank-r decomposition requires sum_r U[a,r]V[b,r]W[c,r] = T[a,b,c].
    # For the 72 non-zero entries (i,j,k), we need sum_r U[iM+j, r] V[jP+k, r] W[kN+i, r] = 1
    # For the 576-72=504 zero entries, we need the sum = 0.
    
    # Strategy: use structured patterns
    # For rank close to 72, we can try to find a permutation-like structure.
    
    # Approach: Start with the schoolbook (rank 72) and try to merge terms.
    # Actually, let's try a different approach: 
    # The tensor is the "multiplication tensor" which has known lower bounds.
    # For 3x4x6, the Strassen-like or other algorithms might give rank < 72.
    
    # Let's try: for each trial, create a sparse random decomposition
    # with entries from {-1, 0, 1}
    for trial in range(max_trials):
        if time.time() > deadline:
            return None
        s = trial + rank * 1000
        rng_local = random.Random(s)
        
        # Sparse random: each entry is 0 with prob 0.7, else ±1
        U = np.zeros((A_DIM, rank))
        V = np.zeros((B_DIM, rank))
        W = np.zeros((C_DIM, rank))
        for r in range(rank):
            for a in range(A_DIM):
                if rng_local.random() < 0.3:
                    U[a, r] = rng_local.choice([-1, 1])
            for b in range(B_DIM):
                if rng_local.random() < 0.3:
                    V[b, r] = rng_local.choice([-1, 1])
            for c in range(C_DIM):
                if rng_local.random() < 0.3:
                    W[c, r] = rng_local.choice([-1, 1])
        
        if _check_decomposition(U, V, W):
            return U, V, W
    return None

def _greedy_construct(rank, deadline):
    """
    Greedy construction: build the decomposition term by term.
    Start with the residual tensor and try to subtract rank-1 tensors.
    """
    T = _build_tensor()
    residual = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    for r in range(rank):
        if time.time() > deadline:
            return None
        if np.allclose(residual, 0, atol=1e-10):
            break
        # Try to find a rank-1 tensor that reduces the residual
        # A rank-1 tensor is u outer v outer w
        # We want to minimize ||residual - u⊗v⊗w||
        # This is hard in general, but we can try structured candidates.
        
        # Try all combinations of unit vectors for u, v, w
        # That's A_DIM * B_DIM * C_DIM = 12*24*18 = 5184 candidates
        # For each, compute the residual after subtraction
        best_score = np.linalg.norm(residual)
        best_u, best_v, best_w = None, None, None
        
        # Try each (a, b, c) as a potential support
        for a in range(A_DIM):
            for b in range(B_DIM):
                for c in range(C_DIM):
                    # Candidate: u = e_a, v = e_b, w = e_c
                    # residual_new = residual - e_a ⊗ e_b ⊗ e_c
                    # This only changes residual[a,b,c] by -1
                    if abs(residual[a, b, c]) > 1e-10:
                        # Check if this reduces the norm
                        old_val = residual[a, b, c]
                        new_val = old_val - 1.0
                        norm_change = new_val**2 - old_val**2
                        if norm_change < -1e-10:
                            score = best_score + norm_change
                            if score < best_score:
                                best_score = score
                                best_u, best_v, best_w = a, b, c
        
        if best_u is None:
            # Try with -1
            for a in range(A_DIM):
                for b in range(B_DIM):
                    for c in range(C_DIM):
                        if abs(residual[a, b, c]) > 1e-10:
                            old_val = residual[a, b, c]
                            new_val = old_val + 1.0
                            norm_change = new_val**2 - old_val**2
                            if norm_change < -1e-10:
                                score = best_score + norm_change
                                if score < best_score:
                                    best_score = score
                                    best_u, best_v, best_w = a, b, c
                                    # Mark as negative
                                    best_u = -best_u - 1  # encode sign
        
        if best_u is None:
            return None
        
        # Decode
        if best_u < 0:
            a = -best_u - 1
            sign = -1
        else:
            a = best_u
            sign = 1
        
        # Subtract
        residual[a, best_v, best_w] -= sign
        U_list.append((a, sign))
        V_list.append(best_v)
        W_list.append(best_w)
    
    if not np.allclose(residual, 0, atol=1e-10):
        return None
    
    # Build matrices
    actual_rank = len(U_list)
    U = np.zeros((A_DIM, actual_rank))
    V = np.zeros((B_DIM, actual_rank))
    W = np.zeros((C_DIM, actual_rank))
    for r in range(actual_rank):
        U[U_list[r][0], r] = U_list[r][1]
        V[V_list[r], r] = 1
        W[W_list[r], r] = 1
    
    return U, V, W

def _schoolbook():
    """Return the schoolbook decomposition (rank 72)."""
    rank = SCHOOLBOOK_RANK
    U = np.zeros((A_DIM, rank))
    V = np.zeros((B_DIM, rank))
    W = np.zeros((C_DIM, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _schoolbook()
    
    # Try to find lower rank decompositions
    # Known lower bound for 3x4x6: at least 72? No, that's the trivial bound.
    # Actually, the tensor rank of matrix multiplication M_{n,m,p} is at least n*m*p / max(n,m,p)
    # and at most n*m*p. For 3,4,6: lower bound = 72/6 = 12, upper = 72.
    # But the true rank is likely much higher.
    
    # Let's try ranks from low to high
    # Start with a reasonable guess: try ranks 50-71
    rng = random.Random(42)
    
    best_result = (U_fb, V_fb, W_fb)
    best_rank = SCHOOLBOOK_RANK
    
    # Try structured search for ranks below 72
    for rank in range(60, 72):
        if time.time() > deadline:
            break
        result = _structured_search(rank, 200, rng, deadline)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = rank
            break
    
    # If we found something better, try even lower
    if best_rank < 72:
        for rank in range(50, best_rank):
            if time.time() > deadline:
                break
            result = _structured_search(rank, 100, rng, deadline)
            if result is not None:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = rank
                break
    
    # Try greedy construction for various ranks
    for rank in range(65, 72):
        if time.time() > deadline:
            break
        result = _greedy_construct(rank, deadline)
        if result is not None:
            U, V, W = result
            if U.shape[1] < best_rank:
                best_result = (U, V, W)
                best_rank = U.shape[1]
    
    # Final check: make sure we return a valid decomposition
    U, V, W = best_result
    if not _check_decomposition(U, V, W):
        # Fallback to schoolbook
        U, V, W = _schoolbook()
    
    # Convert to lists of lists (integer/half-integer entries)
    U_list = [[float(U[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i, j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i, j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
