import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]


def _build_expected(n, m, p):
    """Build the expected multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _verify(U, V, W, n, m, p):
    """Check that U, V, W decompose the matrix multiplication tensor."""
    R = U.shape[1]
    if R == 0:
        return False
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = _build_expected(n, m, p)
    return np.allclose(T, Expected, atol=1e-9)


def _trivial_decomposition(n, m, p):
    """Trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
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


def _strassen_222():
    """
    Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) such that
    sum_r U[:,r] outer V[:,r] outer W[:,r] = T_{2,2,2}
    
    Indices: a = i*2+j, b = j*2+k, c = k*2+i
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2: (A10+A11)B00 -> C10 += M2
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1
    
    # M3: A00(B10-B11) -> C01 += M3
    U[0, 2] += 1
    V[2, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1
    
    # M4: A11(B01-B00) -> C11 += M4
    U[3, 3] += 1
    V[1, 3] += 1; V[0, 3] -= 1
    W[3, 3] += 1
    
    # M5: (A00+A01)B11 -> C01 += M5
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[1, 4] += 1
    
    # M6: (A10-A00)(B00+B01) -> C10 += M6
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[2, 5] += 1
    
    # M7: (A01-A11)(B10+B11) -> C00 += M7
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    
    return U, V, W


def _block_decomposition(n, m, p):
    """
    Block decomposition for 2x4x7 using Strassen on 2x2x2 sub-blocks.
    Split m=4 into {0,1} and {2,3}.
    For each group, split p=7 into {0,1}, {2,3}, {4,5}, {6}.
    Use Strassen (rank 7) for 2x2x2 blocks, trivial (rank 4) for 2x2x1.
    Total: 2 * (3*7 + 4) = 50
    """
    n, m, p = 2, 4, 7
    total_rank = 50
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    r_offset = 0
    
    for j_group in range(2):
        j_base = j_group * 2
        k_groups = [(0, 2), (2, 4), (4, 6), (6, 7)]
        
        for k_start, k_end in k_groups:
            k_len = k_end - k_start
            if k_len == 2:
                U_s, V_s, W_s = _strassen_222()
                for r in range(7):
                    for ia in range(4):
                        i = ia // 2
                        j_local = ia % 2
                        j = j_base + j_local
                        a_idx = i * m + j
                        U[a_idx, r_offset + r] += U_s[ia, r]
                    for ib in range(4):
                        j_local = ib // 2
                        k_local = ib % 2
                        j = j_base + j_local
                        k = k_start + k_local
                        b_idx = j * p + k
                        V[b_idx, r_offset + r] += V_s[ib, r]
                    for ic in range(4):
                        k_local = ic // 2
                        i = ic % 2
                        k = k_start + k_local
                        c_idx = k * n + i
                        W[c_idx, r_offset + r] += W_s[ic, r]
                r_offset += 7
            elif k_len == 1:
                k = k_start
                for i in range(2):
                    for j_local in range(2):
                        j = j_base + j_local
                        a_idx = i * m + j
                        b_idx = j * p + k
                        c_idx = k * n + i
                        U[a_idx, r_offset] = 1
                        V[b_idx, r_offset] = 1
                        W[c_idx, r_offset] = 1
                        r_offset += 1
    
    return {"U": U, "V": V, "W": W}


def _lift_1xn_p(n, p, time_budget=30):
    """
    Find a low-rank decomposition for 1xn x p matrix multiplication.
    Here n=4 (middle dim), p=7.
    Tensor T[a,b,c] = 1 iff a=j, b=j*p+k, c=k (with i=0 fixed, n=1 effectively).
    Actually for 1x4x7: a in {0,1,2,3} (j), b in {0..27} (j*7+k), c in {0..6} (k).
    T[a,b,c] = 1 iff b = a*p+c and c < p.
    
    This is the "embedding" tensor for computing 1x4 @ 4x7 -> 1x7.
    Rank is at most 28 (trivial). Can we do better?
    
    Actually, the 1x4x7 tensor has a special structure:
    T[j, j*7+k, k] = 1 for j in 0..3, k in 0..6.
    
    This is equivalent to computing y = x @ M where x is 1x4, M is 4x7, y is 1x7.
    y[k] = sum_j x[j] * M[j,k]
    
    As a tensor, this has 28 non-zero entries. The rank of this tensor...
    Let me think. The tensor can be written as sum over (j,k) of e_j (outer) e_{j*7+k} (outer) e_k.
    That's 28 terms, so rank <= 28.
    
    Can we do better? The tensor is actually the "structure tensor" of the multiplication.
    For 1xn x p, the rank is at least n*p (since each output y[k] depends on all n inputs).
    Actually, I think the rank is exactly n*p for the trivial case, but maybe we can use
    some algebraic structure.
    
    For 1x4x7, let me just use the trivial decomposition with rank 28, but let's see if
    we can find something better via a simple optimization.
    
    Actually, let's think about it differently. The tensor T satisfies:
    T = sum_{j,k} e_j (outer) e_{j*7+k} (outer) e_k
    
    This is already a sum of 28 rank-1 tensors. The question is whether there's a shorter
    sum. 
    
    For small cases like 1x2x2, the rank is 4 (trivial) or can it be 3?
    T[j, j*2+k, k] = 1. So T[0,0,0]=1, T[0,1,1]=1, T[1,2,0]=1, T[1,3,1]=1.
    That's 4 non-zeros. Rank is 4? Or can we combine?
    
    I'll just use trivial for now and focus on the structural approach for 2x4x7.
    """
    # Trivial decomposition for 1x4x7
    n_local, p_local = 4, 7
    rank = n_local * p_local
    U = np.zeros((n_local, rank))
    V = np.zeros((n_local * p_local, rank))
    W = np.zeros((p_local, rank))
    r = 0
    for j in range(n_local):
        for k in range(p_local):
            a_idx = j
            b_idx = j * p_local + k
            c_idx = k
            U[a_idx, r] = 1
            V[b_idx, r] = 1
            W[c_idx, r] = 1
            r += 1
    return U, V, W, rank


def _structural_2x4x7(n, m, p, time_budget=60):
    """
    Structural decomposition for 2x4x7.
    
    Key idea: C = A @ B where A is 2x4, B is 4x7, C is 2x7.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can write this as:
    C[0,k] = sum_j A[0,j]*B[j,k]
    C[1,k] = sum_j A[1,j]*B[j,k]
    
    Define combinations:
    For each j, define:
      S_j = A[0,j] + A[1,j]
      D_j = A[0,j] - A[1,j]
    
    And for B:
      T_jk = B[j,k]
    
    Then:
    C[0,k] = sum_j A[0,j]*B[j,k] = sum_j ((S_j + D_j)/2) * B[j,k]
    C[1,k] = sum_j A[1,j]*B[j,k] = sum_j ((S_j - D_j)/2) * B[j,k]
    
    This doesn't directly help reduce rank.
    
    Better approach: Group the j indices.
    Split j into two groups: {0,1} and {2,3}.
    
    For group {0,1}:
    C[0,k] += A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] += A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    This is a 2x2x7 sub-problem. We can use Strassen-like for pairs of k.
    
    For the 2x2x7 part with j in {0,1}, k in 0..6:
    Split k into {0,1}, {2,3}, {4,5}, {6}.
    
    For each pair {k1,k2}, we have a 2x2x2 sub-problem which has rank 7 (Strassen).
    For the single k=6, we have 2x2x1 which has rank 4.
    
    Total for one group: 3*7 + 4 = 25.
    Two groups: 50.
    
    This is the same as _block_decomposition. Let me try a different split.
    
    What if we split k differently? E.g., {0,1,2}, {3,4,5}, {6}?
    2x2x3 sub-problem: what's its rank?
    
    For 2x2x3: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in 0,1,2.
    This has 2*2*3 = 12 non-zero entries in the tensor.
    Trivial rank is 12.
    
    Can we do better? The 2x2x3 tensor:
    T[i, j, k] = 1 if (i,j,k) is a valid triple.
    Valid triples: (0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),(1,0,0),(1,0,1),(1,0,2),(1,1,0),(1,1,1),(1,1,2)
    
    That's 12 entries. Rank at most 12.
    
    Using Strassen on 2x2x2 part: split k into {0,1} and {2}.
    2x2x2: rank 7, 2x2x1: rank 4. Total: 11.
    
    So 2x2x3 has rank <= 11.
    
    For 2x2x7: split k into {0,1,2}, {3,4,5}, {6}.
    2x2x3: rank 11, 2x2x3: rank 11, 2x2x1: rank 4. Total: 26.
    
    Two groups: 52. Worse than 50.
    
    What about splitting m into 1+3?
    Group 1: j=0, 2x1x7 -> rank 14 (trivial)
    Group 2: j in {1,2,3}, 2x3x7 -> ?
    
    2x3x7: split k into pairs and singles.
    This is getting complicated. Let me try a different structural approach.
    
    Alternative: Use the fact that we can precompute sums/differences of A and B
    to share computation across different k values.
    
    Define for A (2x4):
    a00, a01, a02, a03 (row 0)
    a10, a11, a12, a13 (row 1)
    
    For B (4x7):
    b0k, b1k, b2k, b3k for k=0..6
    
    C[0,k] = a00*b0k + a01*b1k + a02*b2k + a03*b3k
    C[1,k] = a10*b0k + a11*b1k + a12*b2k + a13*b3k
    
    We need to compute 14 output values using some number of products.
    
    A product is of the form (linear combination of A entries) * (linear combination of B entries).
    
    This is essentially finding a bilinear algorithm for 2x4x7 matrix multiplication.
    
    The minimum number of products is the rank of the tensor.
    
    Let me try a heuristic: for each k, we need 2 outputs. Each output is a dot product
    of a 4-vector (row of A) with a 4-vector (column of B).
    
    We can share products across k values if we define linear combinations of B columns.
    
    For example, define:
    B_sum_01[k] = B[0,k] + B[1,k]  (for each k)
    B_diff_01[k] = B[0,k] - B[1,k]
    
    And for A:
    A_sum_0 = A[0,0] + A[1,0]
    A_diff_0 = A[0,0] - A[1,0]
    etc.
    
    But this doesn't obviously reduce the count.
    
    Let me just go with the block decomposition of rank 50 and try to optimize further
    with a local search.
    """
    # Start with block decomposition
    result = _block_decomposition(n, m, p)
    U = result["U"].copy()
    V = result["V"].copy()
    W = result["W"].copy()
    rank = U.shape[1]
    
    start_time = time.time()
    best_rank = rank
    
    # Try to remove redundant terms or merge
    # For now, just return the block decomposition
    return {"U": U, "V": V, "W": W}


def _randomized_search(n, m, p, target_rank, time_budget=30):
    """
    Try to find a decomposition of rank target_rank using randomized linear algebra.
    This is a heuristic approach.
    """
    start_time = time.time()
    Expected = _build_expected(n, m, p)
    num_a = n * m
    num_b = m * p
    num_c = p * n
    
    # Flatten the tensor
    T_flat = Expected.reshape(num_a, num_b * num_c)
    
    # Try to find a low-rank approximation
    # This is a heuristic, may not find exact solutions
    for _ in range(100):
        if time.time() - start_time > time_budget:
            break
        # Random initialization
        U = np.random.randn(num_a, target_rank) * 0.1
        V = np.random.randn(num_b, target_rank) * 0.1
        W = np.random.randn(num_c, target_rank) * 0.1
        
        # Simple gradient-free optimization
        for iteration in range(1000):
            if time.time() - start_time > time_budget:
                break
            
            # Compute current tensor
            T_current = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T_current - Expected)
            
            if error < 1e-10:
                return {"U": U, "V": V, "W": W}
            
            # Perturb
            mask = np.random.rand(target_rank) < 0.1
            if mask.any():
                U[:, mask] += np.random.randn(num_a, mask.sum()) * 0.01
                V[:, mask] += np.random.randn(num_b, mask.sum()) * 0.01
                W[:, mask] += np.random.randn(num_c, mask.sum()) * 0.01
    
    return None


def construct():
    start_time = time.time()
    deadline = 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with trivial
    trivial = _trivial_decomposition(n, m, p)
    best_rank = trivial["U"].shape[1]
    best_result = trivial
    
    # Try block decomposition (should give rank 50)
    if time.time() - start_time < deadline - 20:
        block = _block_decomposition(n, m, p)
        if _verify(block["U"], block["V"], block["W"], n, m, p):
            if block["U"].shape[1] < best_rank:
                best_rank = block["U"].shape[1]
                best_result = block
    
    # Try structural approach
    if time.time() - start_time < deadline - 10:
        structural = _structural_2x4x7(n, m, p, time_budget=deadline - time.time() - 10)
        if structural is not None and _verify(structural["U"], structural["V"], structural["W"], n, m, p):
            if structural["U"].shape[1] < best_rank:
                best_rank = structural["U"].shape[1]
                best_result = structural
    
    # Try randomized search for even lower rank
    if time.time() - start_time < deadline - 5:
        for target_rank in range(best_rank - 1, max(best_rank - 5, 40), -1):
            if time.time() - start_time > deadline - 5:
                break
            result = _randomized_search(n, m, p, target_rank, time_budget=min(5, deadline - time.time() - 5))
            if result is not None and _verify(result["U"], result["V"], result["W"], n, m, p):
                best_rank = target_rank
                best_result = result
                break
    
    if time.time() - start_time > deadline - 2:
        return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}
    
    return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}


if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = _build_expected(n, m, p)
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}, Sum diff: {diff.sum()}")
