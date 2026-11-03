"""Matrix multiplication 3x5x5 with Strassen-based rank reduction."""
import time
import sys

DIMENSIONS = [3, 5, 5]

# Strassen 2x2x2 terms: each term has U (2x2), V (2x2), W (2x2) coefficients
# U indices: (i, j) for i in 0..1, j in 0..1
# V indices: (j, k) for j in 0..1, k in 0..1  
# W indices: (k, i) for k in 0..1, i in 0..1
STRASSEN_TERMS = [
    # M1 = (A11+A22)(B11+B22)
    ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
    # M2 = (A21+A22)B11
    ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
    # M3 = A11(B21-B22)
    ([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
    # M4 = A22(B12-B11)
    ([(1,1,1)], [(0,1,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
    # M5 = (A11+A12)B22
    ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
    # M6 = (A21-A11)(B11+B12)
    ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
    # M7 = (A12-A22)(B21+B22)
    ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
]

def strassen_block_rank():
    """Number of rank terms for a 2x2x2 Strassen block."""
    return 7

def schoolbook_block_rank(ns, ms, ps):
    """Number of rank terms for a schoolbook block of size ns x ms x ps."""
    return ns * ms * ps

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    # We'll use a recursive splitting strategy.
    # Split 3x5x5 into blocks and apply Strassen where possible.
    # 
    # Strategy: Split the 3 rows into 2+1, the 5 cols into 2+2+1 or 2+3, etc.
    # Let's try: n=3 = 2+1, m=5 = 2+2+1, p=5 = 2+2+1
    # For each combination of sub-blocks (ns, ms, ps):
    #   if ns==2 and ms==2 and ps==2: use Strassen (7 terms)
    #   else: use schoolbook (ns*ms*ps terms)
    
    # Let's enumerate all possible splittings and compute total rank.
    # n_splits: ways to split 3 into positive integers
    # m_splits: ways to split 5 into positive integers
    # p_splits: ways to split 5 into positive integers
    
    def all_compositions(total, max_parts=4):
        """Generate all compositions of total into at most max_parts positive integers."""
        if total <= 1:
            yield [total]
            return
        for first in range(1, total):
            for rest in all_compositions(total - first, max_parts - 1):
                yield [first] + rest
    
    def block_rank(ns, ms, ps):
        """Compute rank for a single block of size ns x ms x ps."""
        # If all dimensions are exactly 2, use Strassen
        if ns == 2 and ms == 2 and ps == 2:
            return 7
        # Could also handle other special cases, but for now just schoolbook
        return ns * ms * ps
    
    # Try different compositions and pick the one with minimum total rank
    best_rank = n * m * p  # default schoolbook
    best_splittings = (None, None, None)
    
    n_comps = list(all_compositions(n, 3))
    m_comps = list(all_compositions(m, 4))
    p_comps = list(all_compositions(p, 4))
    
    for nc in n_comps:
        for mc in m_comps:
            for pc in p_comps:
                total_rank = 0
                # Each combination of one part from each composition forms a block
                # Wait, that's not right. The blocks are determined by the split points.
                # If n is split into [n1, n2], m into [m1, m2], p into [p1, p2],
                # then we have 2x2x2 = 8 blocks, each of size (ni, mj, pk).
                for ni in nc:
                    for mj in mc:
                        for pk in pc:
                            total_rank += block_rank(ni, mj, pk)
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_splittings = (nc, mc, pc)
    
    nc, mc, pc = best_splittings
    if nc is None:
        nc, mc, pc = [n], [m], [p]
    
    # Now build the factor matrices
    rank = best_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Compute offsets for each part
    n_offsets = []
    off = 0
    for part in nc:
        n_offsets.append(off)
        off += part
    m_offsets = []
    off = 0
    for part in mc:
        m_offsets.append(off)
        off += part
    p_offsets = []
    off = 0
    for part in pc:
        p_offsets.append(off)
        off += part
    
    for ni_idx, ni in enumerate(nc):
        for mj_idx, mj in enumerate(mc):
            for pk_idx, pk in enumerate(pc):
                n_off = n_offsets[ni_idx]
                m_off = m_offsets[mj_idx]
                p_off = p_offsets[pk_idx]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen block
                    for u_coeffs, v_coeffs, w_coeffs in STRASSEN_TERMS:
                        for i, j, val in u_coeffs:
                            U[n_off + i][m_off + j][col] += val
                        for j, k, val in v_coeffs:
                            V[m_off + j][p_off + k][col] += val
                        for k, i, val in w_coeffs:
                            W[p_off + k][n_off + i][col] += val
                        col += 1
                else:
                    # Schoolbook block
                    for i in range(ni):
                        for j in range(mj):
                            for k in range(pk):
                                U[n_off + i][m_off + j][col] = 1
                                V[m_off + j][p_off + k][col] = 1
                                W[p_off + k][n_off + i][col] = 1
                                col += 1
    
    # Safety check: ensure we didn't exceed deadline
    if time.time() > deadline:
        # Return what we have so far (should be complete by now)
        pass
    
    return {"U": U, "V": V, "W": W}
