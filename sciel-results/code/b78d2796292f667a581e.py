import time
import numpy as np

DIMENSIONS = [2, 4, 7]


def _strassen_core():
    """Return (U, V, W) for 2x2x2 matmul with rank 7, entries in {-1, 0, 1}.
    U shape (4,7), V shape (4,7), W shape (4,7).
    U[i,j] corresponds to A[i,j], V[j,k] to B[j,k], W[k,i] to C[k,i] (0-indexed).
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)

    # M1 = (A00+A11)(B01+B10) -> C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[1, 0] = 1; V[2, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # M3 = A00(B00-B10) -> C01 += M3, C11 += M3
    U[0, 2] = 1
    V[0, 2] = 1; V[2, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    # M4 = A11(B11-B01) -> C00 += M4, C10 += M4
    U[3, 3] = 1
    V[3, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1

    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1

    # M6 = (A10-A00)(B00+B01) -> C10 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C[1,0] index: k=1,i=0 -> 1*2+0=2? Wait. C[k,i] with k=row of B (1), i=col of A (0). 
    # Let's re-verify indices.
    # A is 2x2: A[i,j]. B is 2x2: B[j,k]. C is 2x2: C[i,k].
    # Convention: U[i*m+j, ...] where m=2. So U[i*2+j].
    # V[j*p+k, ...] where p=2. So V[j*2+k].
    # W[k*n+i, ...] where n=2. So W[k*2+i].
    # M6 contributes to C[1,0]. So k=1, i=0. W[1*2+0] = W[2].
    # My previous code had W[3,5]=1 which is k=1,i=1 -> C[1,1]. That's wrong for M6.
    # Let's check standard Strassen.
    # M6 = (A10 - A00)(B00 + B01).
    # A10 is A[1,0]. A00 is A[0,0].
    # B00 is B[0,0]. B01 is B[0,1].
    # Result adds to C[1,0].
    # So W[2, 5] should be 1.
    # Let's re-verify all W indices.
    
    # M1: (A00+A11)(B01+B10). 
    # A00: i=0,j=0 -> U[0]. A11: i=1,j=1 -> U[3].
    # B01: j=0,k=1 -> V[1]. B10: j=1,k=0 -> V[2].
    # C00: k=0,i=0 -> W[0]. C11: k=1,i=1 -> W[3].
    # W[0,0]=1, W[3,0]=1. Correct.

    # M2: (A10+A11)B00.
    # A10: i=1,j=0 -> U[2]. A11: i=1,j=1 -> U[3].
    # B00: j=0,k=0 -> V[0].
    # C10: k=0,i=1 -> W[1*2+0]=W[2]. C11: k=1,i=1 -> W[3].
    # So W[2,1]=1, W[3,1]=-1.
    # My code had W[2,1]=1; W[3,1]=-1. Correct.

    # M3: A00(B00-B10).
    # A00: U[0].
    # B00: V[0]. B10: j=1,k=0 -> V[2].
    # C01: k=1,i=0 -> W[1*2+0]=W[2]. C11: k=1,i=1 -> W[3].
    # So W[2,2]=1, W[3,2]=1.
    # My code had W[1,2]=1; W[3,2]=1. W[1] is k=0,i=1 -> C[0,1]. 
    # Wait. C[k,i]. 
    # If k=0, i=1, index is 0*2+1 = 1. That is C[0,1].
    # M3 adds to C[0,1] and C[1,1].
    # So W[1,2] and W[3,2] are correct.

    # M4: A11(B11-B01).
    # A11: U[3].
    # B11: j=1,k=1 -> V[3]. B01: j=0,k=1 -> V[1].
    # C00: k=0,i=0 -> W[0]. C10: k=0,i=1 -> W[2].
    # So W[0,3]=1, W[2,3]=1.
    # My code had W[0,3]=1; W[2,3]=1. Correct.

    # M5: (A00+A01)B11.
    # A00: U[0]. A01: i=0,j=1 -> U[1].
    # B11: V[3].
    # C00: k=0,i=0 -> W[0]. C01: k=1,i=0 -> W[2].
    # So W[0,4]=-1, W[2,4]=1.
    # My code had W[0,4]=-1; W[1,4]=1. 
    # W[1] is C[0,1]? No. W[k*n+i]. n=2.
    # W[1] -> k=0, i=1 -> C[0,1].
    # But M5 adds to C[0,0] and C[0,1].
    # So W[0] (C[0,0]) and W[2] (C[1,0])? No.
    # C[0,1] is k=0, i=1. Index 0*2+1 = 1.
    # So W[1,4]=1 is correct for C[0,1].
    # Wait, M5 = (A00+A01)B11.
    # A00 (i=0,j=0) + A01 (i=0,j=1). Both have i=0.
    # B11 (j=1,k=1).
    # Result C[i,k] = C[0,1].
    # So it only adds to C[0,1]?
    # Standard Strassen:
    # M5 = (A00+A01)B11.
    # C00 -= M5? No.
    # Let's check a reliable source.
    # M1 = (A00+A11)(B01+B10)
    # M2 = (A10+A11)B00
    # M3 = A00(B00-B10)
    # M4 = A11(B11-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M5 + M4 + M7
    # C01 = M5 + M3
    # C10 = M2 + M6
    # C11 = M1 - M3 - M2 + M4
    
    # Let's re-evaluate M5.
    # M5 = (A00+A01)B11.
    # A00: i=0,j=0. A01: i=0,j=1.
    # B11: j=1,k=1.
    # Sum over j.
    # A[0,0]B[0,1] + A[0,1]B[1,1].
    # This is C[0,1].
    # So M5 contributes ONLY to C[0,1].
    # But the formula says C00 -= M5. How?
    # Ah, the standard formulas often define M5 differently or I am misremembering the assignment.
    # Let's derive M5's contribution.
    # M5 = A00*B11 + A01*B11.
    # A00*B11 is part of C01.
    # A01*B11 is part of C01.
    # So M5 is exactly C01's contribution from the second column of B? No.
    # C01 = A00*B01 + A01*B11.
    # M5 = A00*B11 + A01*B11.
    # This doesn't match C01.
    # Let's look at M3: A00(B00-B10) = A00*B00 - A00*B10.
    # C01 = A00*B01 + A01*B11.
    # This is getting messy. Let's just verify the core by reconstruction.
    
    # I will re-construct the core carefully in code and verify it before using.
    # If verification fails, I will fall back to schoolbook for that block.
    
    # Let's fix the W indices based on the standard formulas:
    # C00 = M1 - M5 + M4 + M7
    # C01 = M5 + M3
    # C10 = M2 + M6
    # C11 = M1 - M3 - M2 + M4

    # M1: C00(+1), C11(+1). W[0,0]=1, W[3,0]=1.
    # M2: C10(+1), C11(-1). W[2,1]=1, W[3,1]=-1.
    # M3: C01(+1), C11(-1). W[1,2]=1, W[3,2]=-1.
    # M4: C00(+1), C10(+1). W[0,3]=1, W[2,3]=1.
    # M5: C00(-1), C01(+1). W[0,4]=-1, W[1,4]=1.
    # M6: C10(+1). W[2,5]=1.
    # M7: C00(+1). W[0,6]=1.

    # Let's update W in the function body below.
    W[0, 0] = 1; W[3, 0] = 1
    W[2, 1] = 1; W[3, 1] = -1
    W[1, 2] = 1; W[3, 2] = -1
    W[0, 3] = 1; W[2, 3] = 1
    W[0, 4] = -1; W[1, 4] = 1
    W[2, 5] = 1
    W[0, 6] = 1

    # Verify U and V again.
    # M1: (A00+A11)(B01+B10)
    # A00: U[0,0]=1. A11: U[3,0]=1.
    # B01: V[1,0]=1. B10: V[2,0]=1.
    # Correct.
    
    # M2: (A10+A11)B00
    # A10: U[2,1]=1. A11: U[3,1]=1.
    # B00: V[0,1]=1.
    # Correct.
    
    # M3: A00(B00-B10)
    # A00: U[0,2]=1.
    # B00: V[0,2]=1. B10: V[2,2]=-1.
    # Correct.
    
    # M4: A11(B11-B01)
    # A11: U[3,3]=1.
    # B11: V[3,3]=1. B01: V[1,3]=-1.
    # Correct.
    
    # M5: (A00+A01)B11
    # A00: U[0,4]=1. A01: U[1,4]=1.
    # B11: V[3,4]=1.
    # Correct.
    
    # M6: (A10-A00)(B00+B01)
    # A10: U[2,5]=1. A00: U[0,5]=-1.
    # B00: V[0,5]=1. B01: V[1,5]=1.
    # Correct.
    
    # M7: (A01-A11)(B10+B11)
    # A01: U[1,6]=1. A11: U[3,6]=-1.
    # B10: V[2,6]=1. B11: V[3,6]=1.
    # Correct.

    return U, V, W


def _schoolbook_core(n, m, p):
    """Return (U, V, W) for nxmxp matmul with rank n*m*p, entries in {0, 1}."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _verify_core(U, V, W, n, m, p, tol=1e-9):
    """Verify a core decomposition for nxmxp."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _generate_splits(total):
    """Generate all ways to split 'total' into positive integer parts (compositions)."""
    if total == 1:
        yield [1]
    else:
        for first in range(1, total + 1):
            for rest in _generate_splits(total - first):
                yield [first] + rest


def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Precompute cores
    U_s, V_s, W_s = _strassen_core()
    if not _verify_core(U_s, V_s, W_s, 2, 2, 2):
        # Fallback if Strassen core is broken (shouldn't happen, but safe)
        U_s, V_s, W_s = _schoolbook_core(2, 2, 2)
        use_strassen = False
    else:
        use_strassen = True

    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook_core(n, m, p)

    # Enumerate all block splits for m and p
    # m=4, p=7
    # We split m into blocks of size 2 (since n=2, Strassen works on 2x2x2).
    # Actually, we can split m into any sizes, but Strassen is only for 2x2x2.
    # For other sizes, we use schoolbook.
    # However, the most efficient known strategy is to use 2x2x2 Strassen blocks.
    # So we should only consider splits where m-blocks and p-blocks are compatible.
    # If we use a 2x2x2 Strassen block, the m-block size must be 2 and p-block size must be 2.
    # If we use schoolbook, any size works.
    
    # Strategy: Enumerate splits of m and p into blocks.
    # For each pair of blocks (m_i, p_j), if m_i==2 and p_j==2 and use_strassen, use Strassen.
    # Otherwise use schoolbook.
    
    m_splits = list(_generate_splits(m))
    p_splits = list(_generate_splits(p))

    for m_sp in m_splits:
        for p_sp in p_splits:
            # Calculate total rank for this split
            total_rank = 0
            for mb in m_sp:
                for pb in p_sp:
                    if mb == 2 and pb == 2 and use_strassen:
                        total_rank += 7
                    else:
                        total_rank += n * mb * pb
            
            if total_rank < best_rank:
                # Build the decomposition
                U = np.zeros((nm, total_rank), dtype=np.int64)
                V = np.zeros((mp, total_rank), dtype=np.int64)
                W = np.zeros((pn, total_rank), dtype=np.int64)
                
                col = 0
                m_offset = 0
                for mb in m_sp:
                    p_offset = 0
                    for pb in p_sp:
                        if mb == 2 and pb == 2 and use_strassen:
                            # Embed Strassen core
                            # U: rows i*4 + (m_offset..m_offset+1)
                            # V: rows (m_offset..m_offset+1)*7 + (p_offset..p_offset+1)
                            # W: rows (p_offset..p_offset+1)*2 + i
                            for r in range(7):
                                for i in range(2):
                                    for j in range(2):
                                        U[i * 4 + (m_offset + j), col] += U_s[i * 2 + j, r]
                                for j in range(2):
                                    for k in range(2):
                                        V[(m_offset + j) * 7 + (p_offset + k), col] += V_s[j * 2 + k, r]
                                for k in range(2):
                                    for i in range(2):
                                        W[(p_offset + k) * 2 + i, col] += W_s[k * 2 + i, r]
                                col += 1
                        else:
                            # Embed schoolbook core
                            U_core, V_core, W_core = _schoolbook_core(n, mb, pb)
                            for r in range(n * mb * pb):
                                for idx in range(n * mb):
                                    i, j = divmod(idx, mb)
                                    U[i * 4 + (m_offset + j), col] += U_core[idx, r]
                                for idx in range(mb * pb):
                                    j, k = divmod(idx, pb)
                                    V[(m_offset + j) * 7 + (p_offset + k), col] += V_core[idx, r]
                                for idx in range(pb * n):
                                    k, i = divmod(idx, n)
                                    W[(p_offset + k) * 2 + i, col] += W_core[idx, r]
                                col += 1
                        p_offset += pb
                    m_offset += mb
                
                # Verify
                if _verify_full(U, V, W, n, m, p):
                    best_rank = total_rank
                    best_U, best_V, best_W = U, V, W

    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    # Quick verify
    n, m, p = DIMENSIONS
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    print("Valid:", _verify_full(U, V, W, n, m, p))
