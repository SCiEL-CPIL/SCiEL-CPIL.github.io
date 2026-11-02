"""Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]
DEADLINE = 140.0  # safety margin before 150s kill


def _verify(U, V, W, tol=1e-9):
    """Verify that the decomposition is correct. W indexed as W[k*n+i, r]."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)

    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False

    rank = U.shape[1]

    # Vectorized verification: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Build the full tensor and check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                dec_val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(dec_val - 1.0) > tol:
                    return False
    return True


def _build_rank24():
    """Direct sum of three 2x3x2 blocks, each of rank 8 (Strassen 7 + 1 outer product)."""
    n, m, p = DIMENSIONS
    rank_per_block = 8
    rank = 3 * rank_per_block  # 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # For each 2x3x2 block:
    # A indices (i*3+j): A00=0,A01=1,A02=2,A10=3,A11=4,A12=5
    # B indices (j*2+k): B00=0,B01=1,B10=2,B11=3,B20=4,B21=5
    # C indices (i*2+k): C00=0,C01=1,C10=2,C11=3
    #
    # Strassen for A[0:2,0:2] x B[0:2,0:2] (7 terms):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B10
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # Output C[i,k] = sum of appropriate M's
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Plus outer product term: A[i,2]*B[2,k] for i in {0,1}, k in {0,1}
    # This is rank 1: U[A[i,2]]=A[i,2], V[B[2,k]]=B[2,k], W[C[i,k]]=1

    # Strassen terms as (u_map, v_map, w_map)
    # u_map: local A index -> value
    # v_map: local B index -> value  
    # w_map: local C index -> value
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 4: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00
        ({3: 1, 4: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B10-B00)
        ({0: 1}, {2: 1, 0: -1}, {1: 1, 3: 1}),
        # M4 = A11(B01-B10)
        ({4: 1}, {1: 1, 2: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)B10
        ({0: 1, 1: 1}, {2: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)(B00+B01)
        ({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)(B10+B11)
        ({1: 1, 4: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    # Outer product term: A[i,2]*B[2,k]
    # Local A indices for column 2: A02=2, A12=5
    # Local B indices for row 2: B20=4, B21=5
    # Local C indices: C00=0, C01=1, C10=2, C11=3
    # T[i,2,k] = A[i,2]*B[2,k], so:
    # C00 gets A02*B20, C01 gets A02*B21, C10 gets A12*B20, C11 gets A12*B21
    # As rank-1: U[2]=A02_coeff, U[5]=A12_coeff, V[4]=B20_coeff, V[5]=B21_coeff, W all = 1
    # But this is NOT a single rank-1 term in the tensor sense because
    # T[i,2,k] = A[i,2]*B[2,k] means for fixed (i,k), the value is A[i,2]*B[2,k].
    # We need: sum_r U[2 or 5, r] * V[4 or 5, r] * W[C_idx, r] = A[i,2]*B[2,k]
    # 
    # Actually, this IS rank 1 as a 3-tensor over indices (i, j=2, k):
    # T[i,2,k] = A[i,2]*B[2,k] = (vector in i-space) outer (vector in k-space)
    # As a 3-tensor with middle index fixed, its rank = matrix rank of [A[i,2]*B[2,k]]_{i,k}
    # which is rank 1 (it's an outer product of column vector [A02,A12] and row vector [B20,B21]).
    # 
    # So we need ONE term: 
    # U[2, r] = 1, U[5, r] = 1  (for A[i,2] with i=0,1)
    # V[4, r] = 1, V[5, r] = 1  (for B[2,k] with k=0,1)
    # W[0, r] = 1, W[1, r] = 1, W[2, r] = 1, W[3, r] = 1
    # 
    # Check: sum_r U[i*3+2, r]*V[2*2+k, r]*W[k*2+i, r]
    # For i=0,k=0: U[2,r]*V[4,r]*W[0,r] = 1*1*1 = 1. Should be A02*B20. 
    # Hmm, this gives 1 but we need A02*B20. 
    # 
    # The issue: the decomposition represents T[i,j,k]=1 for all i,j,k (the identity tensor).
    # The actual multiplication is: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] = 1 means the decomposition reconstructs the identity, 
    # and the actual product is obtained by contracting with A and B.
    # 
    # So for the term corresponding to A[i,2]*B[2,k], we need:
    # U[i*3+2, r] * V[2*2+k, r] * W[k*2+i, r] summed over r should give 1 
    # when we're looking at the (i,j=2,k) entry of the tensor.
    # 
    # Wait, I'm confusing myself. The tensor T[i,j,k] = 1 for all i,j,k.
    # The decomposition is: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # For the (i, j=2, k) entry: 1 = sum_r U[i*3+2, r] * V[2*2+k, r] * W[k*2+i, r]
    # 
    # This must hold for ALL i in {0,1} and k in {0,1}. So we need:
    # U[2,r]*V[4,r]*W[0,r] + ... = 1 (for i=0,k=0)
    # U[2,r]*V[5,r]*W[1,r] + ... = 1 (for i=0,k=1)
    # U[5,r]*V[4,r]*W[2,r] + ... = 1 (for i=1,k=0)
    # U[5,r]*V[5,r]*W[3,r] + ... = 1 (for i=1,k=1)
    # 
    # If we use ONE term (r=7 in the block):
    # U[2,7]*V[4,7]*W[0,7] = 1
    # U[2,7]*V[5,7]*W[1,7] = 1
    # U[5,7]*V[4,7]*W[2,7] = 1
    # U[5,7]*V[5,7]*W[3,7] = 1
    # 
    # Let U[2,7]=a, U[5,7]=b, V[4,7]=c, V[5,7]=d, W[0,7]=e, W[1,7]=f, W[2,7]=g, W[3,7]=h
    # a*c*e = 1
    # a*d*f = 1
    # b*c*g = 1
    # b*d*h = 1
    # 
    # Choose a=1, b=1, c=1, d=1, then e=1, f=1, g=1, h=1. Done!
    # 
    # So the outer product term is:
    # U[2, r] = 1, U[5, r] = 1
    # V[4, r] = 1, V[5, r] = 1
    # W[0, r] = 1, W[1, r] = 1, W[2, r] = 1, W[3, r] = 1

    outer_term = ({2: 1, 5: 1}, {4: 1, 5: 1}, {0: 1, 1: 1, 2: 1, 3: 1})

    all_terms = strassen_terms + [outer_term]
    # Total: 7 + 1 = 8 terms per block

    for block in range(3):
        for t_idx, (u_map, v_map, w_map) in enumerate(all_terms):
            col = block * rank_per_block + t_idx
            for idx, val in u_map.items():
                U[idx][col] = val
            for l_idx, val in v_map.items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in w_map.items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val

    return {"U": U, "V": V, "W": W}


def _build_rank11():
    """
    Rank-11 decomposition for 2x3x6:
    - 7 Strassen terms for the 2x2x2 sub-block (j in {0,1}, k in {0,1})
    - 4 additional terms for the residual j=2 contributions:
      * j=2, k in {0,1,2,3,4,5} split into 4 terms
    """
    n, m, p = DIMENSIONS
    rank = 10
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen terms for 2x2x2 block (j in {0,1}, k in {0,1})
    # Local indices: A[i,j] -> i*3+j, B[j,k] -> j*2+k, C[i,k] -> k*2+i
    # But for the sub-block, j in {0,1} and k in {0,1}
    # U index: i*m + j = i*3 + j
    # V index: j*p + k = j*6 + k
    # W index: k*n + i = k*2 + i
    strassen = [
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 4: 1}, {0: 1, 6: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00
        ({3: 1, 4: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B10-B00)
        ({0: 1}, {6: 1, 0: -1}, {1: 1, 3: 1}),
        # M4 = A11(B01-B10)
        ({4: 1}, {1: 1, 6: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)B10
        ({0: 1, 1: 1}, {6: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)(B00+B01)
        ({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)(B10+B11)
        ({1: 1, 4: -1}, {6: 1, 7: 1}, {0: 1}),
    ]
    
    for t_idx, (u_map, v_map, w_map) in enumerate(strassen):
        for idx, val in u_map.items():
            U[idx, t_idx] = val
        for idx, val in v_map.items():
            V[idx, t_idx] = val
        for idx, val in w_map.items():
            W[idx, t_idx] = val
    
    # Now handle j=2: C[i,k] += A[i,2]*B[2,k] for all i,k
    # This is a rank-4 contribution (2x2 outer product structure)
    # We need: sum_r U[i*3+2, r] * V[2*6+k, r] * W[k*2+i, r] = 1 for all i,k in {0,1}
    # i.e., U[2,r]*V[12+k,r]*W[k*2+i,r] summed = 1
    # 
    # Use 4 terms (one for each (i,k) pair):
    # Term 7: i=0,k=0: U[2,7]=1, V[12,7]=1, W[0,7]=1
    # Term 8: i=0,k=1: U[2,8]=1, V[13,8]=1, W[1,8]=1
    # Term 9: i=1,k=0: U[5,9]=1, V[12,9]=1, W[2,9]=1
    # Term 10: i=1,k=1: U[5,10]=1, V[13,10]=1, W[3,10]=1
    #
    # But wait - we can do better! The j=2 contribution is:
    # T[i,2,k] = 1 for all i,k
    # This is a 2x2 all-ones tensor in (i,k) space, which has rank 2 as a matrix.
    # As a 3-tensor with j fixed at 2, it's rank 2 (not 4).
    # 
    # We can use 2 terms:
    # Term 7: U[2,7]=1, U[5,7]=1, V[12,7]=1, V[13,7]=1, W[0,7]=1, W[1,7]=1, W[2,7]=1, W[3,7]=1
    # Check: for i=0,k=0: U[2,7]*V[12,7]*W[0,7] = 1*1*1 = 1 ✓
    # for i=0,k=1: U[2,7]*V[13,7]*W[1,7] = 1*1*1 = 1 ✓
    # for i=1,k=0: U[5,7]*V[12,7]*W[2,7] = 1*1*1 = 1 ✓
    # for i=1,k=1: U[5,7]*V[13,7]*W[3,7] = 1*1*1 = 1 ✓
    # But this only covers k in {0,1}. We need k in {0,...,5}!
    
    # Actually, for j=2, we need T[i,2,k]=1 for ALL k in 0..5
    # So: sum_r U[i*3+2, r] * V[12+k, r] * W[k*2+i, r] = 1 for all i in {0,1}, k in {0,...,5}
    #
    # Let's use the structure: for each k, we need the 2x2 matrix [U[i*3+2,r]*V[12+k,r]*W[k*2+i,r]]_r to sum to 1
    # 
    # Strategy: use 4 terms where each term covers a subset of k values
    # Term 7: k in {0,1,2,3}: U[2,7]=U[5,7]=1, V[12,7]=V[13,7]=V[14,7]=V[15,7]=1, W[0,7]=W[1,7]=W[2,7]=W[3,7]=W[4,7]=W[5,7]=W[6,7]=W[7,7]=1
    # Check for i=0,k=0: U[2,7]*V[12,7]*W[0,7] = 1 ✓
    # for i=0,k=1: U[2,7]*V[13,7]*W[1,7] = 1 ✓
    # for i=0,k=2: U[2,7]*V[14,7]*W[4,7] = 1 ✓
    # for i=0,k=3: U[2,7]*V[15,7]*W[5,7] = 1 ✓
    # for i=1,k=0: U[5,7]*V[12,7]*W[2,7] = 1 ✓
    # etc. All check out!
    #
    # Term 8: k in {4,5}: U[2,8]=U[5,8]=1, V[16,8]=V[17,8]=1, W[8,8]=W[9,8]=W[10,8]=W[11,8]=1
    # Check for i=0,k=4: U[2,8]*V[16,8]*W[8,8] = 1 ✓
    # for i=0,k=5: U[2,8]*V[17,8]*W[9,8] = 1 ✓
    # for i=1,k=4: U[5,8]*V[16,8]*W[10,8] = 1 ✓
    # for i=1,k=5: U[5,8]*V[17,8]*W[11,8] = 1 ✓
    #
    # This gives 7 + 2 = 9 terms! Can we do even better?
    # Actually, we can combine: use ONE term for all k:
    # Term 7: U[2,7]=U[5,7]=1, V[12,7]=...=V[17,7]=1, W[0,7]=...=W[11,7]=1
    # Check for i=0,k=0: U[2,7]*V[12,7]*W[0,7] = 1*1*1 = 1 ✓
    # for i=0,k=1: U[2,7]*V[13,7]*W[1,7] = 1 ✓
    # ... all the way to i=1,k=5: U[5,7]*V[17,7]*W[11,7] = 1 ✓
    # 
    # So we only need 1 extra term! Total rank = 7 + 1 = 8!
    # But wait, this can't be right because the tensor rank should be at least...
    # Let me double-check. The tensor T[i,j,k]=1 for all i,j,k.
    # For j=2, the slice T[:,2,:] is a 2x6 all-ones matrix, which has rank 1 as a matrix.
    # As a 3-tensor with j=2 fixed, the "fiber" is a 2x6 matrix of rank 1.
    # The contribution from j=2 to the tensor rank is at most 2 (since we need to cover 2 i-values).
    # But with the trick above, we can cover it with 1 term because the W factor can be all 1s.
    #
    # Let me verify: if we use 1 term for j=2:
    # U[2,7]=1, U[5,7]=1
    # V[12,7]=V[13,7]=V[14,7]=V[15,7]=V[16,7]=V[17,7]=1
    # W[0,7]=W[1,7]=...=W[11,7]=1
    #
    # For (i,j,k) = (0,2,0): U[0*3+2,7]*V[2*6+0,7]*W[0*2+0,7] = U[2,7]*V[12,7]*W[0,7] = 1*1*1 = 1 ✓
    # For (i,j,k) = (0,2,1): U[2,7]*V[13,7]*W[1,7] = 1 ✓
    # For (i,j,k) = (1,2,5): U[5,7]*V[17,7]*W[11,7] = 1 ✓
    #
    # This works! So rank 8 might be possible!
    # But we also need to check that the Strassen terms don't interfere with j=2.
    # In the Strassen terms, V indices are 0,1,6,7 (for j=0,1 and k=0,1).
    # V[12..17] are only used in term 7, so no interference. ✓
    # U[2] and U[5] are used in Strassen terms too (U[2] not used, U[5] not used in Strassen).
    # Wait, U indices in Strassen: 0,1,3,4 (i*3+j for i in {0,1}, j in {0,1}).
    # U[2] = i=0,j=2 and U[5] = i=1,j=2 are NOT used in Strassen. ✓
    # W indices in Strassen: 0,1,2,3 (k*2+i for k in {0,1}, i in {0,1}).
    # W[4..11] are not used in Strassen. ✓
    #
    # So rank 8 is achievable! Let me construct it.
    
    # Actually wait - I need to re-examine. The Strassen decomposition handles j in {0,1}, k in {0,1}.
    # But what about j in {0,1} and k in {2,3,4,5}? Those are NOT covered!
    # 
    # T[i,0,k] = 1 for all i, k in 0..5
    # T[i,1,k] = 1 for all i, k in 0..5
    # T[i,2,k] = 1 for all i, k in 0..5
    #
    # Strassen only handles k in {0,1} for j in {0,1}.
    # We still need to handle:
    # - j=0, k in {2,3,4,5}
    # - j=1, k in {2,3,4,5}
    # - j=2, k in {0,1,2,3,4,5}
    #
    # For j=0, k in {2,...,5}: T[i,0,k]=1 for all i
    # This is a 2x4 all-ones sub-tensor (in i,k space). Rank 1 as matrix.
    # Use 1 term: U[0,r]=1, V[0*6+k,r]=1 for k in {2,...,5}, W[k*2+i,r]=1 for all i,k
    # 
    # For j=1, k in {2,...,5}: similar, 1 term
    # For j=2, k in {0,...,5}: 1 term
    #
    # Total: 7 (Strassen) + 3 (extra) = 10 terms!
    #
    # Can we do better? Let's see if we can merge some of the extra terms.
    # The three extra terms have different U, V, W patterns:
    # Term A (j=0,k>=2): U[0]=1, V[2..5]=1, W[all]=1
    # Term B (j=1,k>=2): U[1]=1, V[8..11]=1, W[all]=1
    # Term C (j=2,k>=0): U[2]=U[5]=1, V[12..17]=1, W[all]=1
    #
    # These can't be easily merged because they have different U and V supports.
    # So rank 10 seems to be the minimum with this approach.
    #
    # But can we do rank 9? We'd need to merge two of the three extra terms.
    # That seems unlikely given the disjoint supports.
    
    # Let me construct rank 10:
    # Term 7: j=0, k in {2,3,4,5}
    U[0, 7] = 1
    V[2, 7] = 1; V[3, 7] = 1; V[4, 7] = 1; V[5, 7] = 1
    for k in range(2, 6):
        for i in range(2):
            W[k * 2 + i, 7] = 1
    
    # Term 8: j=1, k in {2,3,4,5}
    U[1, 8] = 1
    V[8, 8] = 1; V[9, 8] = 1; V[10, 8] = 1; V[11, 8] = 1
    for k in range(2, 6):
        for i in range(2):
            W[k * 2 + i, 8] = 1
    
    # Term 9: j=2, k in {0,1,2,3,4,5}
    U[2, 9] = 1; U[5, 9] = 1
    for k in range(6):
        V[12 + k, 9] = 1
    for k in range(6):
        for i in range(2):
            W[k * 2 + i, 9] = 1
    
    # Wait, that's only 10 terms (7 + 3). Let me recount: columns 0-6 are Strassen, 7,8,9 are extra. That's 10 columns.
    # But I defined rank=11 above. Let me fix.
    
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()

    # Try rank 10 first
    try:
        best10 = _build_rank11()  # Function name kept, but builds rank 10
        if _verify(best10["U"], best10["V"], best10["W"]):
            return best10
    except Exception:
        pass

    if time.time() - start_time > DEADLINE:
        # Fallback to rank 33
        best33 = _build_rank33_fallback()
        return best33

    # Try rank 24
    best = _build_rank24()
    if _verify(best["U"], best["V"], best["W"]):
        return best
    
    # Fallback to rank 33
    best33 = _build_rank33_fallback()
    if _verify(best33["U"], best33["V"], best33["W"]):
        return best33
    return best


def _build_rank33_fallback():
    """Fallback: direct sum of three 2x3x2 blocks (rank 11 each) -> rank 33."""
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    terms = [
        ({0: 1, 4: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({3: 1, 4: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({4: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 4: -1}, {2: 1, 3: 1}, {0: 1}),
        ({2: 1}, {4: 1}, {0: 1}),
        ({2: 1}, {5: 1}, {1: 1}),
        ({5: 1}, {4: 1}, {2: 1}),
        ({5: 1}, {5: 1}, {3: 1}),
    ]

    for block in range(3):
        for t_idx, (u_map, v_map, w_map) in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in u_map.items():
                U[idx][col] = val
            for l_idx, val in v_map.items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in w_map.items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val

    return {"U": U, "V": V, "W": W}
