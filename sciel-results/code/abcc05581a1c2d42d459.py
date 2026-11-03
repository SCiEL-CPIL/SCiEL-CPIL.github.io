"""
Rank-minimal exact tensor decomposition for 2x4 by 4x6 matrix multiplication.

Strategy:
  1. Embed the 2x4x6 multiplication into a 4x4x4 multiplication via zero-padding.
  2. Use the known rank-17 (Strassen-like) decomposition for 4x4x4.
  3. Project the resulting factor matrices onto the non-zero subspace defined by the
     original input dimensions to obtain an exact decomposition of the 2x4x6 problem.
  4. If projection yields rank <= 17, great. Otherwise, try to reduce further using
     alternating least squares on the projected factors.
  5. Fallback: use a structured block decomposition (Strassen 2x2 blocks) with rank 42.

The key insight: the 2x4x6 tensor is a sub-tensor of a 4x4x4 tensor. If we have a rank-R
decomposition of the 4x4x4 tensor, we can restrict it to the non-zero coordinates of the
2x4x6 tensor. This may yield a rank-R decomposition of the 2x4x6 tensor, or we may be able
to reduce the rank further by exploiting the sparsity/structure.

Actually, a cleaner approach: directly construct a low-rank decomposition using the
structure of the problem. The 2x4x6 multiplication can be viewed as:
  C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]  for i in {0,1}, k in {0,1,2,3,4,5}

This is equivalent to 6 separate 2x4 by 4x1 multiplications, or 4 separate 2x1 by 1x6
multiplications, etc.

A better structural approach: use the fact that we can decompose the 4x4 middle index
into 2x2 blocks. Let A = [A0 | A1] where A0, A1 are 2x2, and B = [B0; B1] where B0, B1
are 2x6. Then C = A0*B0 + A1*B1.

Each A0*B0 is a 2x2 by 2x6 multiplication. We can use a rank-7 Strassen-like decomposition
for 2x2x2, applied to each of the 6 columns of B0 (and similarly for B1). But we can share
the Strassen structure across columns to reduce the total rank.

Actually, the most promising approach is to use the known result that the tensor rank of
2x4x6 multiplication is at most 28 (or possibly lower). Let's try to construct a rank-28
or lower decomposition using a combination of Strassen blocks and sharing.

Let me try a different approach: use the embedding into 4x4x4 and the rank-17 algorithm.
"""
import time
import numpy as np

n, m, p = 2, 4, 6
_START = time.time()
_DEADLINE = _START + 140.0  # 10s safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _build_tensor():
    """Build the 2x4x4x6 multiplication tensor."""
    T = np.zeros((n * m, m * p, n * p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _res(U, V, W, T):
    """Compute residual norm of decomposition."""
    return np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)


def _check(U, V, W, T, tol=1e-7):
    if _elapsed() > _DEADLINE:
        return False
    return _res(U, V, W, T) < tol


def _strassen_4x4_rank17():
    """
    Rank-17 decomposition for 4x4x4 matrix multiplication.
    This is a known result (Strassen-like).
    
    For 4x4x4, we can use a 2x2 block Strassen where each block is 2x2,
    and each 2x2x2 uses rank 7. But we can share some products.
    
    Actually, the rank of 4x4x4 is known to be at most 49 (schoolbook) but can be
    reduced. A rank-17 decomposition is not standard for 4x4x4. Let me reconsider.
    
    The rank of 2x2x2 is 7 (Strassen). For 4x4x4, we can view it as 16 blocks of
    2x2x2, but with sharing. A known upper bound is 49 (schoolbook), and it can be
    reduced to about 35-40 with block Strassen.
    
    Let me use a different approach: directly construct a low-rank decomposition
    for 2x4x6 using the structure.
    """
    pass


def _schoolbook_decomposition(T):
    """Schoolbook decomposition with rank n*m*p = 48."""
    R0 = n * m * p
    Uf = np.zeros((n * m, R0))
    Vf = np.zeros((m * p, R0))
    Wf = np.zeros((p * n, R0))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Uf[i * m + j, c] = 1
                Vf[j * p + k, c] = 1
                Wf[k * n + i, c] = 1
                c += 1
    return Uf, Vf, Wf


def _block_strassen_decomposition(T):
    """
    Block Strassen decomposition using 2x2 blocks.
    
    A is 2x4, B is 4x6.
    Split A = [A0 | A1] where A0, A1 are 2x2.
    Split B = [B0; B1] where B0, B1 are 2x6.
    
    C = A0*B0 + A1*B1
    
    Each A0*B0 is 2x2 by 2x6. We can decompose this using Strassen for each pair of
    columns, but share the Strassen factors.
    
    For 2x2 by 2x6:
    C0 = A0 * B0 where A0 is 2x2, B0 is 2x6.
    C0[i,k] = A0[i,0]*B0[0,k] + A0[i,1]*B0[1,k]
    
    Using Strassen for 2x2x2, applied to each column k of B0:
    We need 7 products per column, but we can share the A-side factors.
    
    Actually, let's use a simpler approach: for each of the 6 columns, use a rank-7
    Strassen decomposition. But share the A-side linear combinations.
    
    For A0 (2x2) and B0[:,k] (2x1), the Strassen decomposition is:
    M1 = (A0[0,0]+A0[1,1]) * (B0[0,k]+B0[1,k])
    M2 = (A0[1,0]+A0[1,1]) * B0[0,k]
    M3 = A0[0,0] * (B0[1,k]-B0[0,k])
    M4 = A0[1,1] * (B0[0,k]-B0[1,k])
    M5 = (A0[0,0]+A0[0,1]) * B0[1,k]
    M6 = (A0[1,0]-A0[0,0]) * (B0[0,k]+B0[1,k])
    M7 = (A0[0,1]-A0[1,1]) * (B0[0,k]+B0[1,k])
    
    Wait, this is getting complicated. Let me use the standard 2x2x2 Strassen and
    apply it to each (2x2) by (2x1) subproblem, sharing where possible.
    
    For the full 2x4 by 4x6:
    - 4 columns of A (j=0,1,2,3)
    - 6 rows of B (k=0,1,2,3,4,5)
    - 2 rows of A (i=0,1)
    
    Split into 2x2 blocks:
    A = [A00 A01 | A10 A11]  (2x4)
    B = [B00 B01 B02 B03 B04 B05;
         B10 B11 B12 B13 B14 B15]  (4x6)
    
    C = A0*B0 + A1*B1 where:
    A0 = [A00 A01] (2x2), A1 = [A10 A11] (2x2)
    B0 = [B00..B05; B10..B15] (2x6), B1 = [B20..B25; B30..B35] (2x6)
    
    For A0*B0 (2x2 by 2x6):
    Use Strassen: 7 products, but B0 has 6 columns.
    We can group B0 into 3 pairs of columns: (B0[:,0:2], B0[:,2:4], B0[:,4:6])
    Each 2x2 by 2x2 uses 7 Strassen products.
    Total for A0*B0: 3*7 = 21 products.
    Similarly for A1*B1: 21 products.
    Total: 42 products.
    
    This matches the rank-42 base decomposition.
    """
    # Use the rank-42 block Strassen decomposition
    U, V, W = _rank42_decomposition()
    return U, V, W


def _strassen_2x2_factors():
    """Return U, V, W factors for 2x2x2 Strassen decomposition (rank 7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A21+A22)B11
    U[1, 1] = 1; U[2, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[2, 1] = 1
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1
    return U, V, W


def _rank42_decomposition():
    """Construct the 42-rank decomposition using Strassen blocks."""
    Us, Vs, Ws = _strassen_2x2_factors()
    R = 42
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # For each 2x2 block of A (rows 0-1, cols j,j+1) and each 2x2 block of B
    # (rows j,j+1, cols k,k+1), use a Strassen 2x2x2 decomposition.
    
    # A is 2x4: rows i=0,1; cols j=0,1,2,3
    # B is 4x6: rows j=0,1,2,3; cols k=0,1,2,3,4,5
    # C is 2x6: rows i=0,1; cols k=0,1,2,3,4,5
    
    # Block A into 2 2x2 blocks: A0 = A[:,0:2], A1 = A[:,2:4]
    # Block B into 2 2x6 blocks: B0 = B[0:2,:], B1 = B[2:4,:]
    # C = A0*B0 + A1*B1
    
    # For A0*B0: A0 is 2x2, B0 is 2x6
    # Split B0 into 3 2x2 blocks: B0[:,0:2], B0[:,2:4], B0[:,4:6]
    # Each uses 7 Strassen products: 3*7 = 21
    
    # For A1*B1: similarly 21 products
    
    # Total: 42
    
    # U indices: i*m + j where i in {0,1}, j in {0,1,2,3}
    # V indices: j*p + k where j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # W indices: k*n + i where k in {0,1,2,3,4,5}, i in {0,1}
    
    # Strassen 2x2x2: U, V, W are 4x7
    # U indices correspond to A's 4 entries: (0,0), (0,1), (1,0), (1,1)
    # V indices correspond to B's 4 entries: (0,0), (0,1), (1,0), (1,1)
    # W indices correspond to C's 4 entries: (0,0), (0,1), (1,0), (1,1)
    
    # For A0*B0[:,0:2]:
    # A0 entries: (i=0,j=0), (i=0,j=1), (i=1,j=0), (i=1,j=1)
    #   -> U indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # B0[:,0:2] entries: (j=0,k=0), (j=0,k=1), (j=1,k=0), (j=1,k=1)
    #   -> V indices: 0*6+0=0, 0*6+1=1, 1*6+0=6, 1*6+1=7
    # C[:,0:2] entries: (i=0,k=0), (i=0,k=1), (i=1,k=0), (i=1,k=1)
    #   -> W indices: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3
    
    U_A0 = [0, 1, 4, 5]  # A0's 4 entries
    V_B0_0 = [0, 1, 6, 7]  # B0[:,0:2]'s 4 entries
    W_C_0 = [0, 1, 2, 3]  # C[:,0:2]'s 4 entries
    
    V_B0_1 = [2, 3, 8, 9]  # B0[:,2:4]
    W_C_1 = [4, 5, 6, 7]  # C[:,2:4]
    
    V_B0_2 = [4, 5, 10, 11]  # B0[:,4:6]
    W_C_2 = [8, 9, 10, 11]  # C[:,4:6]
    
    # For A1*B1:
    # A1 entries: (i=0,j=2), (i=0,j=3), (i=1,j=2), (i=1,j=3)
    #   -> U indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # B1 entries: (j=2,k=k), (j=2,k=k+1), (j=3,k=k), (j=3,k=k+1)
    
    U_A1 = [2, 3, 6, 7]  # A1's 4 entries
    
    V_B1_0 = [12, 13, 18, 19]  # B1[:,0:2]
    V_B1_1 = [14, 15, 20, 21]  # B1[:,2:4]
    V_B1_2 = [16, 17, 22, 23]  # B1[:,4:6]
    
    # C entries are the same as for A0*B0
    
    # Build the decomposition
    blocks = [
        (U_A0, V_B0_0, W_C_0, 0),
        (U_A0, V_B0_1, W_C_1, 7),
        (U_A0, V_B0_2, W_C_2, 14),
        (U_A1, V_B1_0, W_C_0, 21),
        (U_A1, V_B1_1, W_C_1, 28),
        (U_A1, V_B1_2, W_C_2, 35),
    ]
    
    for u_idx, v_idx, w_idx, offset in blocks:
        for r in range(7):
            for idx in range(4):
                U[u_idx[idx], offset + r] = Us[idx, r]
                V[v_idx[idx], offset + r] = Vs[idx, r]
                W[w_idx[idx], offset + r] = Ws[idx, r]
    
    return U, V, W


def _alternating_ls(U, V, W, T, iters=200, tol=1e-7):
    """Alternating least squares to drive residual to zero for fixed rank."""
    for it in range(iters):
        if _elapsed() > _DEADLINE - 5:
            break
        # Update U
        VW = np.einsum('br,cr->bcr', V, W)  # (m*p, n*p, R)
        for a in range(n * m):
            Aa = VW.transpose(2, 0, 1).reshape(VW.shape[2], -1)
            ta = T[a].reshape(-1)
            U[a] = np.linalg.lstsq(Aa, ta, rcond=None)[0]
        # Update V
        UW = np.einsum('ar,cr->acr', U, W)  # (n*m, n*p, R)
        for b in range(m * p):
            Ab = UW.transpose(2, 0, 1).reshape(UW.shape[2], -1)
            tb = T[:, b, :].reshape(-1)
            V[b] = np.linalg.lstsq(Ab, tb, rcond=None)[0]
        # Update W
        UV = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, R)
        for c in range(n * p):
            Ac = UV.transpose(2, 0, 1).reshape(UV.shape[2], -1)
            tc = T[:, :, c].reshape(-1)
            W[c] = np.linalg.lstsq(Ac, tc, rcond=None)[0]
        if _res(U, V, W, T) < tol:
            break
    return U, V, W


def _try_reduce_rank(U, V, W, T, target_rank, max_trials=10):
    """Try to reduce rank from current to target_rank using alternating LS."""
    if _elapsed() > _DEADLINE - 10:
        return None
    
    rng = np.random.default_rng(42)
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return (U, V, W)
    
    for trial in range(max_trials):
        if _elapsed() > _DEADLINE - 5:
            break
        
        # Initialize with random perturbation
        U_new = U[:, :target_rank].copy() + 0.1 * rng.standard_normal((n * m, target_rank))
        V_new = V[:, :target_rank].copy() + 0.1 * rng.standard_normal((m * p, target_rank))
        W_new = W[:, :target_rank].copy() + 0.1 * rng.standard_normal((p * n, target_rank))
        
        U_new, V_new, W_new = _alternating_ls(U_new, V_new, W_new, T, iters=150, tol=1e-7)
        
        if _res(U_new, V_new, W_new, T) < 1e-7:
            return (U_new, V_new, W_new)
    
    return None


def _embedding_approach(T):
    """
    Embed 2x4x6 into 4x4x4, use a low-rank decomposition of 4x4x4,
    then project onto the non-zero subspace.
    
    The 2x4x6 tensor T has non-zero entries at (i*m+j, j*p+k, k*n+i) for
    i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}.
    
    Embed into 4x4x4 by zero-padding:
    - First mode: 2*4=8 -> 4*4=16 (pad with zeros)
    - Second mode: 4*6=24 -> 4*4=16 (can't pad, 24 > 16)
    
    This doesn't work directly. Let me reconsider.
    
    Actually, 2x4 by 4x6 can be embedded into 4x4 by 4x4 by:
    - Pad A from 2x4 to 4x4
    - Pad B from 4x6 to 4x4 (truncate? No, this changes the problem)
    
    This approach doesn't work cleanly. Let me skip it and use the block Strassen
    with rank reduction.
    """
    return None


def construct():
    global T
    T = _build_tensor()
    
    # Start with rank-42 block Strassen decomposition
    U42, V42, W42 = _rank42_decomposition()
    best_rank = 42
    best = (U42.copy(), V42.copy(), W42.copy())
    
    if not _check(U42, V42, W42, T):
        # Fallback to schoolbook
        Uf, Vf, Wf = _schoolbook_decomposition(T)
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
    
    # Try to reduce rank
    # Lower bound: max matrix rank of flattenings
    M1 = T.reshape(n * m, -1)
    M2 = T.transpose(1, 0, 2).reshape(m * p, -1)
    M3 = T.transpose(2, 0, 1).reshape(n * p, -1)
    lower_bound = max(np.linalg.matrix_rank(M1), np.linalg.matrix_rank(M2), np.linalg.matrix_rank(M3))
    
    search_start = max(lower_bound, 20)
    
    for R in range(search_start, 42):
        if _elapsed() > _DEADLINE - 10:
            break
        
        result = _try_reduce_rank(U42, V42, W42, T, R, max_trials=10)
        if result is not None:
            best_rank = R
            best = result
            break
    
    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
