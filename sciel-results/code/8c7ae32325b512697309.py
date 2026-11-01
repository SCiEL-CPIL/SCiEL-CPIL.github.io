import time
import numpy as np
from numpy.linalg import svd

DIMENSIONS = [2, 4, 7]

def _build_trivial():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1
                V[b_idx, r] = 1
                W[c_idx, r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    R = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i * m + j, j * p + k, k * n + i] = 1
    return np.allclose(T, Expected, atol=1e-9)

def _try_separable_block(n, m, p):
    # Try decomposition where each term is a product of rank-1 in each dimension
    # This is the trivial one, but we can try to merge
    # For 2x4x7, the trivial is 56. Let's see if we can do better.
    # The matrix multiplication tensor M_{2x4x7} has rank <= 56.
    # Known lower bounds: max(nm, mp, pn) = max(8,28,14) = 28
    # Upper bounds from various constructions...
    # For small matrices, the trivial decomposition is often optimal or near-optimal.
    # Let's try a structured decomposition based on splitting.
    
    # Try: decompose into terms where each term handles a subset
    # Since m=4 is the middle, and 4 = 2*2, maybe we can use this structure
    # But this is speculative. For now, let's try a few hand-crafted decompositions.
    
    # Try rank 49 (n*m*p - 7)
    # Try rank 42
    # Try rank 35
    
    # For now, just return the trivial one
    return _build_trivial()

def _try_random_search(n, m, p, max_rank, num_trials=50, time_limit=20):
    start = time.time()
    best = None
    
    dim_a = n * m  # 8
    dim_b = m * p  # 28
    dim_c = p * n  # 14
    
    # Try to find a decomposition with rank less than n*m*p
    for target_rank in range(max_rank - 1, max_rank // 2, -1):
        if time.time() - start > time_limit:
            break
            
        for trial in range(num_trials):
            if time.time() - start > time_limit:
                break
                
            # Generate random sparse matrices
            U = np.zeros((dim_a, target_rank))
            V = np.zeros((dim_b, target_rank))
            W = np.zeros((dim_c, target_rank))
            
            # Each column should be sparse - maybe rank-1 like
            for r in range(target_rank):
                # Choose random indices
                ua = np.random.choice(dim_a, size=1, replace=False)
                vb = np.random.choice(dim_b, size=1, replace=False)
                wc = np.random.choice(dim_c, size=1, replace=False)
                
                U[ua[0], r] = 1
                V[vb[0], r] = 1
                W[wc[0], r] = 1
            
            if _verify(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                return best
    
    return best

def _structured_decomposition_2x4x7():
    """
    Try to find a better decomposition using the structure of 2x4x7.
    n=2, m=4, p=7
    a_idx = i*m + j where i in [0,1], j in [0,3]
    b_idx = j*p + k where j in [0,3], k in [0,6]
    c_idx = k*n + i where k in [0,6], i in [0,1]
    
    Key observation: The index j appears in both a and b, and i appears in a and c.
    This creates a "chain" structure.
    
    For the trivial decomposition, each (i,j,k) triple gets its own rank-1 term.
    Can we merge terms that share structure?
    
    Consider: for fixed i and j, the k-index goes from 0 to 6.
    The term for (i,j,k) is: e_{im+j} ⊗ e_{jp+k} ⊗ e_{kn+i}
    
    We can write: sum_k e_{im+j} ⊗ e_{jp+k} ⊗ e_{kn+i}
    = e_{im+j} ⊗ (sum_k e_{jp+k} ⊗ e_{kn+i})
    
    But this doesn't directly help with rank reduction.
    
    Let's try: for each i (2 choices), j (4 choices), the k-sum has 7 terms.
    If we could represent sum_k e_{jp+k} ⊗ e_{kn+i} with fewer terms...
    
    The matrix S_{i,j} with entries S_{(jp+k), (kn+i)} = 1 if same k, else 0
    This is a 28x14 matrix with 7 non-zero entries.
    Its rank is at most 7, likely 7.
    
    So the total rank would still be at least 2*4*7 = 56 if each (i,j) block has rank 7.
    
    However, different (i,j) blocks share structure, so there might be overlap.
    
    For now, let's try a specific construction:
    Split j into j1 + 2*j2 where j1 in [0,1], j2 in [0,1]
    Split k into k1 + 2*k2 where k1 in [0,1], k2 in [0,3] (but 7 = 2*3+1)
    
    This is getting complex. Let me try a different approach:
    Use the fact that for matrix multiplication, there are known bounds.
    For 2x4x7, the rank is likely 56 or close to it.
    """
    n, m, p = 2, 4, 7
    
    # Let's try to construct a decomposition with rank 49 by merging some terms
    # or try rank 42, 35, etc.
    
    # For now, return trivial as fallback
    return _build_trivial()

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    trivial = _build_trivial()
    
    # First, try structured approaches
    result = _structured_decomposition_2x4x7()
    
    # Verify and use if valid
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        if result["U"].shape[1] < trivial["U"].shape[1]:
            return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
    
    # Try random search for better rank
    if time.time() - start_time < deadline:
        best_rank = trivial["U"].shape[1]
        random_result = _try_random_search(n, m, p, best_rank, 
                                           num_trials=20, 
                                           time_limit=min(30, deadline - (time.time() - start_time)))
        if random_result is not None:
            if _verify(random_result["U"], random_result["V"], random_result["W"], n, m, p):
                if random_result["U"].shape[1] < best_rank:
                    trivial = random_result
    
    # Return the best found
    if time.time() - start_time > deadline - 10:
        # Just return trivial to be safe
        pass
    
    return {"U": trivial["U"].tolist(), "V": trivial["V"].tolist(), "W": trivial["W"].tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}, Sum diff: {diff.sum()}")
