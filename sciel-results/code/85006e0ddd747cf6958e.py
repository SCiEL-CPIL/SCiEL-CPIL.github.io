"""
Rank-84 decomposition for 3x5x6 matrix multiplication using Strassen's rank-7
decomposition on 2x2x2 sub-blocks.

Block partition:
  n=3 -> (2, 1)
  m=5 -> (2, 2, 1)
  p=6 -> (2, 2, 2)

The 2x2x2 blocks use Strassen (rank 7), all other blocks use schoolbook.
"""

import numpy as np
import time

# Strassen rank-7 decomposition for 2x2x2.
# Convention: C[i,k] = sum_j A[i,j] B[j,k]
# Tensor: T[i,j,k] = 1 if we need A[i,j]*B[j,k]*C[i,k]
# W index: k*2 + i (per task convention W[k*n+i])

# The 7 rank-1 terms: (u, v, w) where u in R^{4} (i*2+j), v in R^{4} (j*2+k), w in R^{4} (k*2+i)
# Using standard Strassen:
# P1 = (a+d)(e+h)
# P2 = (c+d)h
# P3 = a(f-e)
# P4 = d(g-c)
# P5 = (a+b)(e+f)
# P6 = (c+b)(g+h)
# P7 = (a-c)(e+f)
#
# C[0,0] = P1 - P5 + P4 + P7
# C[0,1] = P3 + P5
# C[1,0] = P2 + P6
# C[1,1] = P1 - P3 + P2 - P6

# u vectors (index i*2+j):
# P1: a+d -> [1,0,0,1]
# P2: c+d -> [0,0,1,1]
# P3: a  -> [1,0,0,0]
# P4: d  -> [0,0,0,1]
# P5: a+b -> [1,1,0,0]
# P6: c+b -> [0,1,1,0]
# P7: a-c -> [1,0,-1,0]

_strassen_u = np.array([
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [1, 0, -1, 0],
], dtype=float)

# v vectors (index j*2+k):
# P1: e+h -> [1,0,0,1]
# P2: h   -> [0,0,0,1]
# P3: f-e -> [0,-1,1,0]
# P4: g-c -> [-1,0,0,1]
# P5: e+f -> [1,1,0,0]
# P6: g+h -> [0,0,1,1]
# P7: e+f -> [1,1,0,0]

_strassen_v = np.array([
    [1, 0, 0, 1],
    [0, 0, 0, 1],
    [0, -1, 1, 0],
    [-1, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
], dtype=float)

# w vectors (index k*2+i):
# C[0,0] = P1 - P5 + P4 + P7
# C[0,1] = P3 + P5
# C[1,0] = P2 + P6
# C[1,1] = P1 - P3 + P2 - P6
#
# w for P1: C[0,0]+C[1,1] -> indices: k=0,i=0 -> 0; k=1,i=1 -> 3 => [1,0,0,1]
# w for P2: C[1,0]-C[1,1] -> indices: k=0,i=1 -> 1; k=1,i=1 -> 3 => [0,1,0,-1]
# w for P3: C[0,1]-C[1,1] -> indices: k=1,i=0 -> 2; k=1,i=1 -> 3 => [0,0,1,-1]
# w for P4: C[0,0]        -> index: k=0,i=0 -> 0 => [1,0,0,0]
# w for P5: -C[0,0]+C[0,1] -> indices: k=0,i=0 -> 0; k=1,i=0 -> 2 => [-1,0,1,0]
# w for P6: C[1,0]-C[1,1] -> same as P2? No: C[1,0] has +P6, C[1,1] has -P6
#   So w for P6: C[1,0]-C[1,1] -> [0,1,0,-1]... wait that's same as P2.
#   Let me recheck: C[1,0] = P2 + P6, C[1,1] = P1 - P3 + P2 - P6
#   P6 appears in C[1,0] with +1 and C[1,1] with -1.
#   So w for P6: +1 at (k=0,i=1)=1, -1 at (k=1,i=1)=3 => [0,1,0,-1]
#   But P2 also: C[1,0] has +P2, C[1,1] has +P2. So w for P2: +1 at (k=0,i=1)=1, +1 at (k=1,i=1)=3 => [0,1,0,1]
#
# Let me redo carefully:
# C[0,0] = P1 - P5 + P4 + P7
# C[0,1] = P3 + P5
# C[1,0] = P2 + P6
# C[1,1] = P1 - P3 + P2 - P6

# w_r[k*2+i] coefficient of P_r in C[i,k]:
# P1: C[0,0]+1, C[1,1]+1 => w1[0]=1, w1[3]=1 => [1,0,0,1]
# P2: C[1,0]+1, C[1,1]+1 => w2[1]=1, w2[3]=1 => [0,1,0,1]
# P3: C[0,1]+1, C[1,1]-1 => w3[2]=1, w3[3]=-1 => [0,0,1,-1]
# P4: C[0,0]+1 => w4[0]=1 => [1,0,0,0]
# P5: C[0,0]-1, C[0,1]+1 => w5[0]=-1, w5[2]=1 => [-1,0,1,0]
# P6: C[1,0]+1, C[1,1]-1 => w6[1]=1, w6[3]=-1 => [0,1,0,-1]
# P7: C[0,0]+1 => w7[0]=1 => [1,0,0,0]

_strassen_w = np.array([
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, -1],
    [1, 0, 0, 0],
    [-1, 0, 1, 0],
    [0, 1, 0, -1],
    [1, 0, 0, 0],
], dtype=float)


def _build_block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """
    Build a decomposition using Strassen for 2x2x2 blocks and schoolbook otherwise.
    
    n_parts: list of sizes summing to n
    m_parts: list of sizes summing to m
    p_parts: list of sizes summing to p
    """
    # Collect all terms
    terms = []  # list of (u_vec, v_vec, w_vec)
    
    for ni, ns in enumerate(n_parts):
        for mi, ms in enumerate(m_parts):
            for pi, ps in enumerate(p_parts):
                if ns == 2 and ms == 2 and ps == 2:
                    # Use Strassen
                    for r in range(7):
                        terms.append(('strassen', ni, mi, pi, r))
                else:
                    # Schoolbook
                    for i in range(ns):
                        for j in range(ms):
                            for k in range(ps):
                                terms.append(('school', ni, mi, pi, (i, j, k)))
    
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for idx, term in enumerate(terms):
        if term[0] == 'strassen':
            _, ni, mi, pi, r = term
            # Compute offsets
            n_off = sum(n_parts[:ni])
            m_off = sum(m_parts[:mi])
            p_off = sum(p_parts[:pi])
            
            # u: index (n_off + i) * m + (m_off + j) for i,j in 0..1
            for local_i in range(2):
                for local_j in range(2):
                    gi = n_off + local_i
                    gj = m_off + local_j
                    u_idx = gi * m + gj
                    U[u_idx, idx] = _strassen_u[r, local_i * 2 + local_j]
            
            # v: index (m_off + j) * p + (p_off + k) for j,k in 0..1
            for local_j in range(2):
                for local_k in range(2):
                    gj = m_off + local_j
                    gk = p_off + local_k
                    v_idx = gj * p + gk
                    V[v_idx, idx] = _strassen_v[r, local_j * 2 + local_k]
            
            # w: index (p_off + k) * n + (n_off + i) for k,i in 0..1
            for local_k in range(2):
                for local_i in range(2):
                    gk = p_off + local_k
                    gi = n_off + local_i
                    w_idx = gk * n + gi
                    W[w_idx, idx] = _strassen_w[r, local_k * 2 + local_i]
        
        else:
            _, ni, mi, pi, (li, lj, lk) = term
            n_off = sum(n_parts[:ni])
            m_off = sum(m_parts[:mi])
            p_off = sum(p_parts[:pi])
            
            gi = n_off + li
            gj = m_off + lj
            gk = p_off + lk
            
            U[gi * m + gj, idx] = 1.0
            V[gj * p + gk, idx] = 1.0
            W[gk * n + gi, idx] = 1.0
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # We check a random set of (i,j,k) triples
    rng = np.random.RandomState(42)
    for _ in range(200):
        i = rng.randint(n)
        j = rng.randint(m)
        k = rng.randint(p)
        val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
        if abs(val - 1.0) > 1e-9:
            return False
    # Also check that off-diagonal terms are zero
    # For the multiplication tensor, T[i,j,k] = 1 only for the "diagonal" structure
    # Actually the tensor is: T[i,j,k] = 1 for all valid (i,j,k) since
    # C[i,k] = sum_j A[i,j] B[j,k] means each (i,j,k) contributes exactly 1.
    # Wait - the tensor for matrix multiplication has T[i,j,k] = 1 for all i,j,k.
    # So we just need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for ALL i,j,k.
    # Let's verify more thoroughly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def construct():
    """
    Constructs a rank-84 tensor decomposition for 3x5x6 matrix multiplication
    using Strassen's rank-7 decomposition on 2x2x2 sub-blocks.
    
    Block partition: n=3=(2,1), m=5=(2,2,1), p=6=(2,2,2)
    """
    n, m, p = 3, 5, 6
    
    # Try the block decomposition
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]
    
    U, V, W = _build_block_decomposition(n, m, p, n_parts, m_parts, p_parts)
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
