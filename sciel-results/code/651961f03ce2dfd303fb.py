import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Strassen 2x2x2 rank 7 decomposition
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # U, V, W are 4x7 matrices (indices i,j in 0..1, k in 0..1)
    # We need: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k
    
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U[i,j,r] corresponds to A[i,j] coefficient in M_r
    # V[j,k,r] corresponds to B[j,k] coefficient in M_r
    # W[k,i,r] corresponds to C output coefficient (since C[i,k] gets contribution)
    # Actually W[k,i,r] means the r-th term contributes W[k,i,r] to C[i,k] from the k,i position...
    # Let me re-think: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For 2x2x2 subblock with offsets (i0,j0,k0):
    # T[i0+li, j0+lj, k0+lk] = sum_r U[(i0+li)*m+(j0+lj), r] * V[(j0+lj)*p+(k0+lk), r] * W[(k0+lk)*n+(i0+li), r]
    
    # Strassen patterns as (li,lj) -> U val, (lj,lk) -> V val, (lk,li) -> W val
    strassen = [
        # M1: (A11+A22)(B11+B22) -> C11 + C22
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        # M2: (A21+A22)B11 -> C21
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(0,1): 1}},
        # M3: A11(B12-B22) -> C12 + C22
        {'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(1,0): 1, (1,1): 1}},
        # M4: A22(B21-B11) -> C11 + C21
        {'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (0,1): 1}},
        # M5: (A11+A12)B22 -> C11 + C12
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (1,0): 1}},
        # M6: (A21-A11)(B11+B12) -> C22
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}},
        # M7: (A12-A22)(B21+B22) -> C11
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}},
    ]
    
    def build_strassen_block(n_off, m_off, p_off, rank_start, U, V, W):
        """Add a 2x2x2 Strassen block at offsets (n_off, m_off, p_off), starting at rank_start.
        Returns the new rank position."""
        r = rank_start
        for pat in strassen:
            for (li, lj), val in pat['U'].items():
                U[(n_off + li) * m + (m_off + lj)][r] = val
            for (lj, lk), val in pat['V'].items():
                V[(m_off + lj) * p + (p_off + lk)][r] = val
            for (lk, li), val in pat['W'].items():
                W[(p_off + lk) * n + (n_off + li)][r] = val
            r += 1
        return r
    
    def build_block(n_off, m_off, p_off, nb, mb, pb, rank_start, U, V, W):
        """Build a nb x mb x pb block. Use Strassen for 2x2x2, schoolbook otherwise."""
        if nb == 2 and mb == 2 and pb == 2:
            return build_strassen_block(n_off, m_off, p_off, rank_start, U, V, W)
        else:
            r = rank_start
            for li in range(nb):
                for lj in range(mb):
                    for lk in range(pb):
                        U[(n_off + li) * m + (m_off + lj)][r] = 1
                        V[(m_off + lj) * p + (p_off + lk)][r] = 1
                        W[(p_off + lk) * n + (n_off + li)][r] = 1
                        r += 1
            return r
    
    def compositions(d, max_part=2):
        """Generate compositions of d using parts 1..max_part."""
        if d == 0:
            yield []
            return
        for first in range(1, min(max_part, d) + 1):
            for rest in compositions(d - first, max_part):
                yield [first] + list(rest)
    
    def compute_rank(ns, ms, ps):
        rank = 0
        for a, b, c in zip(ns, ms, ps):
            if a == 2 and b == 2 and c == 2:
                rank += 7
            else:
                rank += a * b * c
        return rank
    
    # Try all compositions
    n_comps = list(compositions(n, 2))
    m_comps = list(compositions(m, 2))
    p_comps = list(compositions(p, 2))
    
    best_rank = n * m * p
    best_splits = None
    
    for nc in n_comps:
        for mc in m_comps:
            for pc in p_comps:
                # Check that sums match
                if sum(nc) != n or sum(mc) != m or sum(pc) != p:
                    continue
                r = compute_rank(nc, mc, pc)
                if r < best_rank:
                    best_rank = r
                    best_splits = (nc, mc, pc)
    
    if best_splits is None:
        best_splits = ([n], [m], [p])
    
    ns, ms, ps = best_splits
    
    # Build the decomposition
    rank = best_rank
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    n_off = 0
    m_off = 0
    p_off = 0
    r = 0
    
    for i_b in range(len(ns)):
        for j_b in range(len(ms)):
            for k_b in range(len(ps)):
                nb, mb, pb = ns[i_b], ms[j_b], ps[k_b]
                r = build_block(n_off, m_off, p_off, nb, mb, pb, r, U, V, W)
                p_off += pb
            p_off = 0
            m_off += mb
        m_off = 0
        n_off += nb
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r_idx in range(rank):
                    val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

# EVOLVE-BLOCK-END
