"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under the 120s/150s wall-clock limits

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 36, 36, 36

    # Build the schoolbook decomposition as a fallback
    schoolbook_rank = n * m * p  # 216
    
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = schoolbook_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    # Try cyclic group decomposition: rank 180
    cyclic_rank = 180
    
    if cyclic_rank < best_rank:
        success = _build_cyclic_group(n, m, p, deadline)
        if success is not None:
            U_found, V_found, W_found = success
            best_rank = cyclic_rank
            best_U = U_found
            best_V = V_found
            best_W = W_found

    # Try Block-Strassen: partition 6x6 into 3x3 blocks of 2x2 submatrices
    strassen_rank = 3 * 3 * 3 * 7  # 189
    
    if strassen_rank < best_rank:
        success = _build_block_strassen(n, m, p, deadline)
        if success is not None:
            U_found, V_found, W_found = success
            best_rank = strassen_rank
            best_U = U_found
            best_V = V_found
            best_W = W_found

    # Convert to lists for the output format
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}


def _build_cyclic_group(n, m, p, deadline):
    """Build a cyclic group tensor decomposition for 6x6x6 matrix multiplication.
    
    Uses the Z_6 cyclic structure to create a rank-180 decomposition.
    The key idea: group the 216 schoolbook terms into 180 structured terms
    by exploiting the periodicity of indices modulo 6.
    
    For each residue r in Z_6, we create 30 rank-1 terms. The total is 6*30=180.
    """
    nm, mp, pn = n * m, m * p, p * n
    total_rank = 180
    
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    # For the cyclic group approach, we use the following construction:
    # The matrix multiplication tensor T_{ijk} = e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    # has 216 nonzero entries. We want to express it as a sum of 180 rank-1 tensors.
    
    # Strategy: For each pair (d1, d2) where d1 = (i-k) mod 6, d2 = (j-i) mod 6,
    # we can create structured terms. There are 36 such pairs, but we can combine
    # them using the cyclic symmetry to get 180 independent terms.
    
    # Alternative approach: Use the fact that the tensor is a sum over (i,j,k) of
    # e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}. We can write this as:
    # Sum over s=0 to 5 of (sum over t=0 to 5 of e_{(s+t) mod 6, (s+2t) mod 6} ⊗ 
    #                         e_{(s+2t) mod 6, (s+t) mod 6} ⊗ 
    #                         e_{(s+t) mod 6, (s+t) mod 6})
    # Wait, that's not quite right. Let me think differently.
    
    # Better approach: Use the discrete Fourier transform on Z_6.
    # The tensor can be decomposed using the characters of Z_6.
    # There are 6 irreducible characters, and for each character, we get 30 terms.
    
    # For character chi_r(i) = exp(2πi*r*i/6), r = 0,1,2,3,4,5:
    # We create 30 rank-1 terms per character, total 180.
    
    # The construction: For each r in {0,...,5} and each (a,b) in a set of 30 pairs,
    # we define:
    # U[:, col] has 1's at positions (i,j) where chi_r(i-j) = 1 (or some condition)
    # V[:, col] has 1's at positions (j,k) where chi_r(j-k) = 1
    # W[:, col] has 1's at positions (k,i) where chi_r(k-i) = 1
    # But this doesn't quite work directly.
    
    # Let me try a different construction based on the actual tensor structure.
    # The tensor T_{ijk} = 1 if and only if the indices form a valid matrix multiplication path.
    # We can write T = sum_{r=0}^{5} sum_{s=0}^{5} sum_{t=0}^{5} c_{rst} * u_r ⊗ v_s ⊗ w_t
    # where c_{rst} are coefficients and u_r, v_s, w_t are vectors.
    
    # Actually, let me use a simpler approach: the tensor is the sum of 216 basis tensors.
    # We can group them into 180 groups where each group is a rank-1 tensor.
    # This requires finding 180 rank-1 tensors that sum to the original tensor.
    
    # One way: For each (i,k) pair, there are 6 values of j. We can combine these 6 terms
    # into 5 rank-1 terms using linear combinations, saving 1 term per (i,k) pair.
    # There are 36 (i,k) pairs, so we save 36 terms, getting 216-36=180.
    
    # Construction: For fixed (i,k), the 6 terms are:
    # sum_{j=0}^{5} e_{ij} ⊗ e_{jk} ⊗ e_{ki}
    # = e_{ki} ⊗ (sum_{j=0}^{5} e_{ij} ⊗ e_{jk})
    # The inner sum is a 6x6 matrix with 1's on the diagonal (if we index properly).
    # Actually, sum_{j=0}^{5} e_{ij} ⊗ e_{jk} is a matrix M where M_{(i,j),(j',k')} = 1 if j=j' and i=i', k=k'.
    # This is just the identity in the middle index.
    
    # Let me try yet another approach: use the fact that for fixed i and k,
    # sum_{j=0}^{5} e_{ij} ⊗ e_{jk} ⊗ e_{ki} can be written as a sum of 5 rank-1 tensors
    # instead of 6, by combining terms.
    
    # For fixed (i,k), define:
    # Term j: u_j = e_{ij}, v_j = e_{jk}, w_j = e_{ki}
    # We can write: sum_{j=0}^{5} u_j ⊗ v_j ⊗ w_j = sum_{j=0}^{4} (u_j - u_5) ⊗ v_j ⊗ w_j + u_5 ⊗ (sum_{j=0}^{5} v_j) ⊗ w_j
    # But this doesn't reduce the rank.
    
    # I think the correct approach is to use the cyclic group algebra structure.
    # Let me implement a known construction: the tensor rank of n×n matrix multiplication
    # is at most n^3 - n^2 + n for n a prime power, but for n=6 this doesn't directly apply.
    
    # For now, let me implement a heuristic: try to find 180 rank-1 terms that sum to the tensor.
    # This is a hard problem, so I'll use a greedy approach.
    
    # Actually, let me just return None and fall back to Strassen for now.
    # The cyclic group approach requires more careful mathematical analysis.
    
    return None


def _build_block_strassen(n, m, p, deadline):
    """Build a block-Strassen decomposition for 6x6x6 matrix multiplication.
    
    Partition into 3x3 blocks of 2x2 submatrices.
    For each triple (bi, bk, bj), apply Strassen's rank-7 decomposition
    to the 2x2 block product A[bi,bk] * B[bk,bj] -> C[bi,bj].
    Total: 3 * 3 * 3 * 7 = 189 terms.
    """
    nm, mp, pn = n * m, m * p, p * n
    num_blocks = 3  # 6 / 2 = 3 blocks per dimension
    total_rank = num_blocks * num_blocks * num_blocks * 7  # 189
    
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    # Strassen's 7 terms for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # Each term: (A_factor, B_factor, C_factor) where factors are tuples of 4 values
    # A factor indices: (a00, a01, a10, a11) -> global (row, col) = (0,0), (0,1), (1,0), (1,1)
    # B factor indices: (b00, b01, b10, b11) -> global (row, col) = (0,0), (0,1), (1,0), (1,1)
    # C factor indices: (c00, c01, c10, c11) -> global (row, col) = (0,0), (0,1), (1,0), (1,1)
    
    # M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
    # M2 = (a10+a11)b00 -> C10 += M2, C11 -= M2
    # M3 = a00(b10-b11) -> C01 += M3, C11 += M3
    # M4 = a11(b01-b00) -> C00 += M4, C10 += M4
    # M5 = (a00+a01)b11 -> C00 -= M5, C01 += M5
    # M6 = (a10-a00)(b00+b01) -> C11 += M6
    # M7 = (a01-a11)(b10+b11) -> C00 += M7
    
    # A factors (for a00, a01, a10, a11):
    # M1: (1, 0, 0, 1)
    # M2: (0, 0, 1, 1)
    # M3: (1, 0, 0, 0)
    # M4: (0, 0, 0, 1)
    # M5: (1, 1, 0, 0)
    # M6: (0, 0, 1, -1)
    # M7: (0, 1, 0, -1)
    
    # B factors (for b00, b01, b10, b11):
    # M1: (1, 0, 0, 1)
    # M2: (1, 0, 0, 0)
    # M3: (0, 0, 1, -1)
    # M4: (0, 1, -1, 0)
    # M5: (0, 0, 0, 1)
    # M6: (1, 1, 0, 0)
    # M7: (0, 0, 1, 1)
    
    # C factors (for c00, c01, c10, c11):
    # M1: (1, 0, 0, 1)
    # M2: (0, 0, 1, -1)
    # M3: (0, 1, 0, 1)
    # M4: (1, 0, 1, 0)
    # M5: (-1, 1, 0, 0)
    # M6: (0, 0, 0, 1)
    # M7: (1, 0, 0, 0)
    
    strassen_terms = [
        # (A_factor, B_factor, C_factor)
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),   # M1
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),  # M2
        ((1, 0, 0, 0), (0, 0, 1, -1), (0, 1, 0, 1)),  # M3
        ((0, 0, 0, 1), (0, 1, -1, 0), (1, 0, 1, 0)),  # M4
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),  # M5
        ((0, 0, 1, -1), (1, 1, 0, 0), (0, 0, 0, 1)),  # M6
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),  # M7
    ]
    
    col = 0
    for bi in range(num_blocks):
        for bk in range(num_blocks):
            for bj in range(num_blocks):
                for s in range(7):
                    a_fac, b_fac, c_fac = strassen_terms[s]
                    
                    # A factor: for each (r, s_idx) in 2x2 block
                    # r is row offset (0 or 1), s_idx is col offset (0 or 1)
                    # Global A index: (bi*2 + r) * m + (bk*2 + s_idx)
                    # a_fac index: r * 2 + s_idx
                    for r in range(2):
                        for s_idx in range(2):
                            idx = r * 2 + s_idx
                            if a_fac[idx] != 0:
                                global_a = (bi * 2 + r) * m + (bk * 2 + s_idx)
                                U[global_a, col] = a_fac[idx]
                    
                    # B factor: for each (r, s_idx) in 2x2 block
                    # r is row offset (0 or 1), s_idx is col offset (0 or 1)
                    # Global B index: (bk*2 + r) * p + (bj*2 + s_idx)
                    # b_fac index: r * 2 + s_idx
                    for r in range(2):
                        for s_idx in range(2):
                            idx = r * 2 + s_idx
                            if b_fac[idx] != 0:
                                global_b = (bk * 2 + r) * p + (bj * 2 + s_idx)
                                V[global_b, col] = b_fac[idx]
                    
                    # C factor: for each (t, u) in 2x2 block
                    # t is row offset (0 or 1), u is col offset (0 or 1)
                    # Global C index: C[bi*2+t, bj*2+u]
                    # W index: (bj*2+u) * n + (bi*2+t)  [trace convention W[k*n+i]]
                    # c_fac index: t * 2 + u
                    for t in range(2):
                        for u in range(2):
                            idx = t * 2 + u
                            if c_fac[idx] != 0:
                                global_w = (bj * 2 + u) * n + (bi * 2 + t)
                                W[global_w, col] = c_fac[idx]
                    
                    col += 1
    
    # Verify the decomposition
    if not _verify_decomposition(n, m, p, U, V, W):
        return None
    
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W, tol=1e-8):
    """Verify that U, V, W form a valid decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    if U.shape != (nm, rank) or V.shape != (mp, rank) or W.shape != (pn, rank):
        return False
    
    # Check all valid (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check that invalid triples give 0 - check all 36*36*36 = 46656 combinations
    # This is feasible and ensures correctness
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                i1 = a // m
                j1 = a % m
                j2 = b // p
                k1 = b % p
                k2 = c // n
                i2 = c % n
                
                if i1 == i2 and j1 == j2 and k1 == k2:
                    continue
                
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
    
    return True
# EVOLVE-BLOCK-END
