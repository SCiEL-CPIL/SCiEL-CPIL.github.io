"""Tensor decomposition search for 3x4x6 matrix multiplication (rank-minimal)."""

import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 6]
DEADLINE = 150.0  # seconds, with safety margin


def _elapsed():
    return time.time() - _START_TIME


def _verify(U, V, W, n, m, p):
    """Check if U@V^T == W^T in the tensor sense: sum_r U[i,j,r]*V[j,k,r] == W[k,i,r] summed over r."""
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r]
    # Must equal sum_r' W[k*n+i, r']
    # Actually the decomposition is: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] = delta(i,j,k) where the multiplication tensor has T[i,j,k] = 1 if (i,j,k) corresponds to the product term
    # For matmul: T[i,j,k] = 1 for all (i,j,k) in the trivial case
    # The factorization: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r[i*m+j] = U[i*m+j, r], v_r[j*p+k] = V[j*p+k, r], w_r[k*n+i] = W[k*n+i, r]
    # So T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * ... no wait.
    # Actually for the matrix multiplication tensor:
    # T[i,j,k] = 1 (the tensor has all 1s in the n x m x p box)
    # And the factorization is T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is length n*m, v_r is length m*p, w_r is length p*n
    # T[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    # But that's a 3-way outer product, not what we want.
    
    # Actually the standard tensor decomposition for matmul is:
    # C = A * B, where C[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor T has indices (i, j, k) and T[i,j,k] = 1
    # Rank-r decomposition: T[i,j,k] = sum_{r=1}^{rank} a_{i,j,r} * b_{j,k,r} * c_{k,i,r}
    # Wait, that's not right either. Let me think again.
    
    # The matrix multiplication tensor: for multiplying n×m by m×p to get n×p
    # T[i, j, k] = 1 for all i in [n], j in [m], k in [p]
    # A rank-r decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait no. Let me re-read the task.
    
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # The decomposition must satisfy: for the matrix multiplication tensor
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]? 
    # No, that would be a 3-way outer product which doesn't make sense for matmul.
    
    # Actually, I think the correct interpretation is:
    # The multiplication tensor T of order (n, m, p) with T[i,j,k] = 1
    # is decomposed as T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n}
    # But that's not standard. Let me reconsider.
    
    # Standard: C_{ik} = sum_j A_{ij} B_{jk}
    # Tensor T_{ijk} = 1 (all ones, n×m×p tensor)
    # Rank-r: T_{ijk} = sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}
    # So U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # And the check is: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Wait, that's a trilinear form, which is unusual. Let me verify with the trivial case:
    # U[i*m+j, r] = 1 if r == i*m+j, else 0
    # V[j*p+k, r] = 1 if r == j*p+k... but r goes up to n*m*p, and j*p+k < m*p <= n*m*p
    # Hmm, in the seed: U[i*m+j][r] = 1, V[j*p+k][r] = 1, W[k*n+i][r] = 1
    # with r = i*m*p + j*p + k (0-indexed)
    # So for a given (i,j,k), the term r = i*m*p + j*p + k has U=1, V=1, W=1
    # But for other (i',j',k') with the same r, U[i'*m+j'][r] would be 0 unless i'=i and j'=j
    # Wait no, r is unique for each (i,j,k) triple.
    # So sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for the correct r, 0 otherwise
    # Total sum = 1. Yes! That works.
    
    rank = U.shape[1]
    Umat = np.array(U, dtype=np.float64)
    Vmat = np.array(V, dtype=np.float64)
    Wmat = np.array(W, dtype=np.float64)
    
    # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
    Ur = Umat.reshape(n, m, rank)
    Vr = Vmat.reshape(m, p, rank)
    Wr = Wmat.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    
    # Check all entries are 1
    return np.allclose(T, 1.0, atol=1e-9)


_START_TIME = time.time()


def _make_trivial(n, m, p):
    """Create the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
    return U, V, W


def _strassen_2x2x2():
    """Return the 7-rank Strassen decomposition for 2x2 * 2x2.
    Returns U (4,7), V (4,7), W (4,7) where indices are (i*m+j, r), (j*p+k, r), (k*n+i, r)
    For 2x2: n=m=p=2, so U is (4,7), V is (4,7), W is (4,7)
    U index: i*2+j, V index: j*2+k, W index: k*2+i
    """
    # Strassen:
    # m1 = (a00-a11)*(b11+b10)  -> but let's use standard formulation
    # C00 = m1 - m4 + m6 - m7
    # C01 = m2 + m4 - m5 - m3
    # C10 = m1 + m3
    # C11 = m2 - m4 + m5 + m7
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b10-b11)
    # m4 = a11(b01-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b10)
    # m7 = (a01-a11)(b01+b11)
    
    # For the tensor decomposition, we need:
    # T[i,j,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
    
    # This is tricky. Let me use a different approach.
    # Actually, let me just use the known Strassen factors.
    # The Strassen algorithm computes C = AB where A, B are 2x2.
    # C00 = m1 - m4 + m6 - m7
    # C01 = m2 + m4 - m5 - m3  
    # C10 = m1 + m3
    # C11 = m2 - m4 + m5 + m7
    
    # Each mr is a product of two linear combinations.
    # m1 = (a00+a11)(b00+b11): 
    #   U part: a00+a11 -> U[0*2+0]=1, U[1*2+1]=1 (i.e., U[0]=1, U[3]=1)
    #   V part: b00+b11 -> V[0*2+0]=1, V[1*2+1]=1 (i.e., V[0]=1, V[3]=1)
    #   W part: contributes to C00 -> W[0*2+0]=1 (k=0,i=0)
    #   But m1 appears in C00 with +1 and C10 with +1
    #   So W[0*2+0]=1, W[1*2+0]=1 (k=0,i=0 and k=1,i=0)
    
    # Let me build this properly.
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # m1 = (a00+a11)(b00+b11), contributes to C00(+1), C10(+1)
    U[0, 0] = 1  # a00
    U[3, 0] = 1  # a11
    V[0, 0] = 1  # b00
    V[3, 0] = 1  # b11
    W[0, 0] = 1  # C00: k=0,i=0
    W[2, 0] = 1  # C10: k=1,i=0 (W index = k*2+i = 1*2+0 = 2)
    
    # m2 = (a10+a11)b00, contributes to C01(+1), C11(+1)
    U[2, 1] = 1  # a10 (i=1,j=0)
    U[3, 1] = 1  # a11 (i=1,j=1)
    V[0, 1] = 1  # b00 (j=0,k=0)
    W[1, 1] = 1  # C01: k=1,i=0 -> W[1*2+0]=W[2]? No wait.
    # W index = k*2 + i. C01 means i=0,k=1, so W[1*2+0]=W[2]
    # C11 means i=1,k=1, so W[1*2+1]=W[3]
    W[2, 1] = 1  # C01
    W[3, 1] = 1  # C11
    
    # m3 = a00(b10-b11), contributes to C01(-1), C10(+1)
    U[0, 2] = 1  # a00
    V[2, 2] = 1  # b10 (j=1,k=0)
    V[3, 2] = -1  # b11 (j=1,k=1)
    W[2, 2] = -1  # C01 (i=0,k=1)
    W[2, 2] = 1   # C10 (i=0,k=1)? No, C10 is i=1,k=0 -> W[0*2+1]=W[1]
    # Wait, I need to be more careful.
    # m3 appears in C01 with coefficient -1 and C10 with coefficient +1
    # C01: i=0, k=1 -> W index = 1*2+0 = 2
    # C10: i=1, k=0 -> W index = 0*2+1 = 1
    W[2, 2] = -1  # C01 contribution
    W[1, 2] = 1   # C10 contribution
    
    # m4 = a11(b01-b00), contributes to C00(-1), C01(+1), C11(-1)
    U[3, 3] = 1  # a11
    V[1, 3] = 1  # b01 (j=0,k=1)
    V[0, 3] = -1  # b00 (j=0,k=0)
    # C00: i=0,k=0 -> W[0]
    # C01: i=0,k=1 -> W[2]
    # C11: i=1,k=1 -> W[3]
    W[0, 3] = -1
    W[2, 3] = 1
    W[3, 3] = -1
    
    # m5 = (a00+a01)b11, contributes to C01(-1), C11(+1)
    U[0, 4] = 1  # a00
    U[1, 4] = 1  # a01 (i=0,j=1)
    V[3, 4] = 1  # b11
    W[2, 4] = -1  # C01
    W[3, 4] = 1   # C11
    
    # m6 = (a10-a00)(b00+b10), contributes to C00(+1)
    U[2, 5] = 1  # a10
    U[0, 5] = -1  # a00
    V[0, 5] = 1  # b00
    V[2, 5] = 1  # b10
    W[0, 5] = 1  # C00
    
    # m7 = (a01-a11)(b01+b11), contributes to C00(-1), C11(+1)
    U[1, 6] = 1  # a01
    U[3, 6] = -1  # a11
    V[1, 6] = 1  # b01
    V[3, 6] = 1  # b11
    W[0, 6] = -1  # C00
    W[3, 6] = 1   # C11
    
    return U, V, W


def _build_block_decomposition(n, m, p):
    """
    Build a block decomposition using Strassen where possible.
    Partition: n=3 -> 1+2, m=4 -> 2+2, p=6 -> 2+2+2
    Blocks:
      (1x2)*(2x2): rank 4 (trivial for 1x2*2x2 is 4, or use optimized)
      (1x2)*(2x2): rank 4
      (1x2)*(2x2): rank 4
      (2x2)*(2x2): rank 7 (Strassen)
      (2x2)*(2x2): rank 7
      (2x2)*(2x2): rank 7
    Total: 4+4+4+7+7+7 = 33
    """
    # Actually, for 1x2 * 2x2, the trivial rank is 1*2*2 = 4.
    # Can we do better? The tensor is 1x2x2, which has all ones.
    # Rank of 1x2x2 tensor: it's just a vector of 4 ones, so rank 1? No.
    # T[0,j,k] = 1 for j in {0,1}, k in {0,1}
    # Rank decomposition: T[0,j,k] = sum_r U[0,j,r] * V[j,k,r] * W[k,0,r]
    # Since i=0 always, W[k,0,r] = W[k,r] effectively.
    # This is equivalent to: 1 = sum_r U[0,j,r] * V[j,k,r] * W[k,r]
    # We can do: r=0: U[0,0,0]=1, V[0,0,0]=1, W[0,0]=1 -> gives 1 for (j=0,k=0)
    # But we need all (j,k) to give 1.
    # Actually with rank 4 (trivial), each (j,k) gets its own term.
    # Can we do rank 2? T[0,j,k] = sum_{r=1}^2 u_j^r v_{jk}^r w_k^r
    # For j=0,k=0: u0^1 v00^1 w0^1 + u0^2 v00^2 w0^2 = 1
    # For j=0,k=1: u0^1 v01^1 w1^1 + u0^2 v01^2 w1^2 = 1
    # For j=1,k=0: u1^1 v10^1 w0^1 + u1^2 v10^2 w0^2 = 1
    # For j=1,k=1: u1^1 v11^1 w1^1 + u1^2 v11^2 w1^2 = 1
    # This is a 2x2 matrix that equals J (all ones) as a sum of 2 rank-1 matrices with the special structure.
    # Actually, let's just use rank 4 for simplicity.
    
    rank_total = 4 + 4 + 4 + 7 + 7 + 7  # = 33
    
    U = np.zeros((n * m, rank_total), dtype=np.int64)
    V = np.zeros((m * p, rank_total), dtype=np.int64)
    W = np.zeros((p * n, rank_total), dtype=np.int64)
    
    r_offset = 0
    
    # Helper to place a block decomposition
    def place_block(i_start, j_start, k_start, bi, bj, bk, U_block, V_block, W_block, offset):
        """Place a block of size bi x bj x bk at positions (i_start, j_start, k_start)."""
        for ri in range(bi):
            for rj in range(bj):
                for r in range(U_block.shape[1]):
                    if U_block[ri * bj + rj, r] != 0:
                        U[(i_start + ri) * m + (j_start + rj), offset + r] = U_block[ri * bj + rj, r]
        for rj in range(bj):
            for rk in range(bk):
                for r in range(V_block.shape[1]):
                    if V_block[rj * bk + rk, r] != 0:
                        V[(j_start + rj) * p + (k_start + rk), offset + r] = V_block[rj * bk + rk, r]
        for rk in range(bk):
            for ri in range(bi):
                for r in range(W_block.shape[1]):
                    if W_block[rk * bi + ri, r] != 0:
                        W[(k_start + rk) * n + (i_start + ri), offset + r] = W_block[rk * bi + ri, r]
    
    # Block (0,0): 1x2 * 2x2, at i=0, j=0..1, k=0..1
    # Trivial rank 4
    U0, V0, W0 = _make_trivial(1, 2, 2)
    place_block(0, 0, 0, 1, 2, 2, U0, V0, W0, r_offset)
    r_offset += 4
    
    # Block (0,1): 1x2 * 2x2, at i=0, j=2..3, k=2..3
    U1, V1, W1 = _make_trivial(1, 2, 2)
    place_block(0, 2, 2, 1, 2, 2, U1, V1, W1, r_offset)
    r_offset += 4
    
    # Block (0,2): 1x2 * 2x2, at i=0, j=0..1, k=4..5
    # Wait, I need to be more careful with the partitioning.
    # m=4 -> j partitions: [0,1], [2,3]
    # p=6 -> k partitions: [0,1], [2,3], [4,5]
    # n=3 -> i partitions: [0], [1,2]
    
    # Block (i=0, j=[0,1], k=[0,1]): 1x2*2x2
    # Block (i=0, j=[0,1], k=[2,3]): 1x2*2x2
    # Block (i=0, j=[0,1], k=[4,5]): 1x2*2x2
    # Block (i=0, j=[2,3], k=[0,1]): 1x2*2x2
    # Block (i=0, j=[2,3], k=[2,3]): 1x2*2x2
    # Block (i=0, j=[2,3], k=[4,5]): 1x2*2x2
    # Block (i=[1,2], j=[0,1], k=[0,1]): 2x2*2x2
    # Block (i=[1,2], j=[0,1], k=[2,3]): 2x2*2x2
    # Block (i=[1,2], j=[0,1], k=[4,5]): 2x2*2x2
    # Block (i=[1,2], j=[2,3], k=[0,1]): 2x2*2x2
    # Block (i=[1,2], j=[2,3], k=[2,3]): 2x2*2x2
    # Block (i=[1,2], j=[2,3], k=[4,5]): 2x2*2x2
    
    # That's 6 blocks of 1x2*2x2 (rank 4 each) and 6 blocks of 2x2*2x2 (rank 7 each)
    # Total: 6*4 + 6*7 = 24 + 42 = 66
    
    # Let me redo this properly.
    U = np.zeros((n * m, 66), dtype=np.int64)
    V = np.zeros((m * p, 66), dtype=np.int64)
    W = np.zeros((p * n, 66), dtype=np.int64)
    
    r_offset = 0
    i_partitions = [(0, 1), (1, 3)]  # (start, end) exclusive
    j_partitions = [(0, 2), (2, 4)]
    k_partitions = [(0, 2), (2, 4), (4, 6)]
    
    for i_start, i_end in i_partitions:
        bi = i_end - i_start
        for j_start, j_end in j_partitions:
            bj = j_end - j_start
            for k_start, k_end in k_partitions:
                bk = k_end - k_start
                if bi == 1 and bj == 2 and bk == 2:
                    Ub, Vb, Wb = _make_trivial(1, 2, 2)
                elif bi == 2 and bj == 2 and bk == 2:
                    Ub, Vb, Wb = _strassen_2x2x2()
                else:
                    Ub, Vb, Wb = _make_trivial(bi, bj, bk)
                
                place_block(i_start, j_start, k_start, bi, bj, bk, Ub, Vb, Wb, r_offset)
                r_offset += Ub.shape[1]
    
    return U, V, W


def _local_search_merge(U, V, W, n, m, p, max_iters=1000):
    """Try to merge pairs of terms to reduce rank."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    for _ in range(max_iters):
        if _elapsed() > DEADLINE - 10:
            break
        merged = False
        for r1 in range(rank):
            if _elapsed() > DEADLINE - 10:
                break
            for r2 in range(r1 + 1, rank):
                # Try to see if term r2 can be absorbed into term r1
                # This is hard to do exactly, so skip for now
                pass
        if merged:
            rank -= 1
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
    
    return U, V, W, rank


# EVOLVE-BLOCK-START
def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = 3, 4, 6
    
    # Build block decomposition (should give rank 66)
    U, V, W = _build_block_decomposition(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fall back to trivial
        U, V, W = _make_trivial(n, m, p)
    
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
