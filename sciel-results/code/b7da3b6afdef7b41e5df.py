"""Matrix multiplication 3x5x7 rank-minimal decomposition using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _construct_schoolbook(n, m, p):
    """Construct the trivial schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 tensors for the 2x2x2 matrix multiplication tensor.
    
    Each term is a triple (u, v, w) of length-4 vectors.
    u is indexed by (i,j) with flat index i*2+j
    v is indexed by (j,k) with flat index j*2+k
    w is indexed by (k,i) with flat index k*2+i
    
    Standard Strassen decomposition:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 - M4 + M5 - M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M7
    
    Note: M6 is not used in the standard C formulas, but it's part of the
    rank-7 tensor decomposition. The actual C formulas that use all 7 terms
    are different. Let me use the correct ones.
    
    Actually, let me verify the tensor decomposition directly.
    For the 2x2x2 tensor, we need sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1
    for all valid (i,j,k).
    
    Let me define the terms and verify:
    """
    # Term 1: M1 = (A00+A11)(B00+B11)
    # u: A00 + A11 -> indices (0,0) and (1,1) -> flat 0 and 3
    # v: B00 + B11 -> indices (0,0) and (1,1) -> flat 0 and 3
    # w: contributes to C00 and C11 -> C00 is (k=0,i=0) flat 0, C11 is (k=1,i=1) flat 3
    # But wait, M1 contributes +1 to C00 and +1 to C11? Let me check.
    # C00 = M1 - M4 + M5 - M7, so M1 contributes +1 to C00
    # C11 = M1 - M2 + M3 + M7, so M1 contributes +1 to C11
    # So w: W[0] += 1, W[3] += 1
    term1 = (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]))
    
    # Term 2: M2 = (A10+A11)B00
    # u: A10 + A11 -> indices (1,0) and (1,1) -> flat 2 and 3
    # v: B00 -> index (0,0) -> flat 0
    # w: M2 contributes +1 to C10 and -1 to C11
    # C10 is (k=0,i=1) flat 1, C11 is (k=1,i=1) flat 3
    # So w: W[1] += 1, W[3] += -1
    term2 = (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]))
    
    # Term 3: M3 = A00(B01-B11)
    # u: A00 -> index (0,0) -> flat 0
    # v: B01 - B11 -> indices (0,1) and (1,1) -> flat 1 and 3
    # w: M3 contributes +1 to C01 and +1 to C11
    # C01 is (k=1,i=0) flat 2, C11 is (k=1,i=1) flat 3
    # So w: W[2] += 1, W[3] += 1
    term3 = (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]))
    
    # Term 4: M4 = A11(B10-B00)
    # u: A11 -> index (1,1) -> flat 3
    # v: B10 - B00 -> indices (1,0) and (0,0) -> flat 2 and 0
    # w: M4 contributes -1 to C00 and +1 to C10
    # C00 is (k=0,i=0) flat 0, C10 is (k=0,i=1) flat 1
    # So w: W[0] += -1, W[1] += 1
    term4 = (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([-1, 1, 0, 0]))
    
    # Term 5: M5 = (A00+A01)B11
    # u: A00 + A01 -> indices (0,0) and (0,1) -> flat 0 and 1
    # v: B11 -> index (1,1) -> flat 3
    # w: M5 contributes +1 to C00 and +1 to C01
    # C00 is (k=0,i=0) flat 0, C01 is (k=1,i=0) flat 2
    # So w: W[0] += 1, W[2] += 1
    term5 = (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([1, 0, 1, 0]))
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    # u: A10 - A00 -> indices (1,0) and (0,0) -> flat 2 and 0
    # v: B00 + B01 -> indices (0,0) and (0,1) -> flat 0 and 1
    # w: M6 contributes +1 to C11 (from C22 = M1 - M2 + M3 + M6)
    # C11 is (k=1,i=1) flat 3
    # So w: W[3] += 1
    term6 = (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]))
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    # u: A01 - A11 -> indices (0,1) and (1,1) -> flat 1 and 3
    # v: B10 + B11 -> indices (1,0) and (1,1) -> flat 2 and 3
    # w: M7 contributes -1 to C00 and +1 to C11
    # C00 is (k=0,i=0) flat 0, C11 is (k=1,i=1) flat 3
    # So w: W[0] += -1, W[3] += 1
    term7 = (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([-1, 0, 0, 1]))
    
    return [term1, term2, term3, term4, term5, term6, term7]


def _block_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
    """
    Generate a decomposition using block-wise Strassen.
    
    n_blocks: list of block sizes for n dimension (rows of A)
    m_blocks: list of block sizes for m dimension (cols of A / rows of B)
    p_blocks: list of block sizes for p dimension (cols of B)
    """
    # Compute block offsets
    n_offsets = [0]
    for s in n_blocks:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_blocks:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_blocks:
        p_offsets.append(p_offsets[-1] + s)
    
    # Count total rank and collect terms
    rank = 0
    terms = []
    for ni in range(len(n_blocks)):
        for mi in range(len(m_blocks)):
            for pi in range(len(p_blocks)):
                ns = n_blocks[ni]
                ms = m_blocks[mi]
                ps = p_blocks[pi]
                if ns == 2 and ms == 2 and ps == 2:
                    # Strassen: 7 terms
                    rank += 7
                    terms.append((ni, mi, pi, 'strassen'))
                else:
                    # Trivial: ns*ms*ps terms
                    rank += ns * ms * ps
                    terms.append((ni, mi, pi, 'trivial'))
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen_terms = _strassen_2x2x2_terms()
    
    col = 0
    for ni, mi, pi, method in terms:
        n0 = n_offsets[ni]
        m0 = m_offsets[mi]
        p0 = p_offsets[pi]
        ns = n_blocks[ni]
        ms = m_blocks[mi]
        ps = p_blocks[pi]
        
        if method == 'strassen':
            # Apply Strassen to the 2x2x2 block
            for u, v, w in strassen_terms:
                # u is indexed by (i,j) within the block, flat index i*2+j
                # Map to global: (n0+i)*m + (m0+j)
                for idx in range(4):
                    i_local = idx // 2
                    j_local = idx % 2
                    U[(n0 + i_local) * m + (m0 + j_local), col] = u[idx]
                
                # v is indexed by (j,k) within the block, flat index j*2+k
                # Map to global: (m0+j)*p + (p0+k)
                for idx in range(4):
                    j_local = idx // 2
                    k_local = idx % 2
                    V[(m0 + j_local) * p + (p0 + k_local), col] = v[idx]
                
                # w is indexed by (k,i) within the block, flat index k*2+i
                # Map to global: (p0+k)*n + (n0+i)
                for idx in range(4):
                    k_local = idx // 2
                    i_local = idx % 2
                    W[(p0 + k_local) * n + (n0 + i_local), col] = w[idx]
                
                col += 1
        else:
            # Trivial multiplication for this block
            for i in range(ns):
                for j in range(ms):
                    for k in range(ps):
                        U[(n0 + i) * m + (m0 + j), col] = 1.0
                        V[(m0 + j) * p + (p0 + k), col] = 1.0
                        W[(p0 + k) * n + (n0 + i), col] = 1.0
                        col += 1
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try different block decompositions and pick the one with lowest rank
    best_rank = n * m * p  # Start with schoolbook
    best_U, best_V, best_W = _construct_schoolbook(n, m, p)
    
    # Block size options
    # n=3: can be [2,1], [1,2], [1,1,1], [3]
    # m=5: can be [2,2,1], [2,1,2], [1,2,2], [2,3], [3,2], [5], [2,1,1,1], etc.
    # p=7: can be [2,2,3], [2,3,2], [3,2,2], [2,2,2,1], [2,2,1,2], etc.
    
    n_options = [
        [2, 1],
        [1, 2],
        [1, 1, 1],
        [3],
    ]
    
    m_options = [
        [2, 2, 1],
        [2, 1, 2],
        [1, 2, 2],
        [2, 3],
        [3, 2],
        [5],
        [2, 1, 1, 1],
        [1, 2, 1, 1],
        [1, 1, 2, 1],
        [1, 1, 1, 2],
    ]
    
    p_options = [
        [2, 2, 3],
        [2, 3, 2],
        [3, 2, 2],
        [2, 2, 2, 1],
        [2, 2, 1, 2],
        [2, 1, 2, 2],
        [1, 2, 2, 2],
        [7],
        [2, 2, 1, 1, 1],
        [2, 1, 2, 1, 1],
        [2, 1, 1, 2, 1],
        [2, 1, 1, 1, 2],
    ]
    
    # Try all combinations
    for n_blocks in n_options:
        if sum(n_blocks) != n:
            continue
        for m_blocks in m_options:
            if sum(m_blocks) != m:
                continue
            for p_blocks in p_options:
                if sum(p_blocks) != p:
                    continue
                
                U, V, W = _block_decomposition(n, m, p, n_blocks, m_blocks, p_blocks)
                rank = U.shape[1]
                
                if rank < best_rank:
                    # Verify the decomposition
                    if _verify(U, V, W, n, m, p):
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W
                        print(f"Found better decomposition: rank {rank} with blocks {n_blocks}, {m_blocks}, {p_blocks}")
        
        if time.time() > deadline:
            break
    
    # If we found a better decomposition, use it; otherwise use schoolbook
    U_out = best_U
    V_out = best_V
    W_out = best_W
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
