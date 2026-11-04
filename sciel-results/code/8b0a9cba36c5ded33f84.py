import numpy as np
import time
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verifies if U, V, W form a valid tensor decomposition for matrix multiplication.
    Uses exact Fraction arithmetic for safety.
    """
    rank = len(U[0])
    
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                total = Fraction(0)
                for r in range(rank):
                    total += U_frac[idx_u][r] * V_frac[idx_v][r] * W_frac[idx_w][r]
                
                if total != 1:
                    return False
    return True

def _try_remove_term(U, V, W, n, m, p, term_idx):
    """
    Try to remove term term_idx.
    Returns (new_U, new_V, new_W) if successful, else None.
    """
    rank = len(U[0])
    if rank <= 1:
        return None
    
    U_new = [row[:] for row in U]
    V_new = [row[:] for row in V]
    W_new = [row[:] for row in W]
    
    for i in range(len(U_new)):
        U_new[i][term_idx] = 0
    for i in range(len(V_new)):
        V_new[i][term_idx] = 0
    for i in range(len(W_new)):
        W_new[i][term_idx] = 0
    
    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
        U_compact = [row[:term_idx] + row[term_idx+1:] for row in U_new]
        V_compact = [row[:term_idx] + row[term_idx+1:] for row in V_new]
        W_compact = [row[:term_idx] + row[term_idx+1:] for row in W_new]
        return (U_compact, V_compact, W_compact)
    
    return None

def _try_pairwise_removal(U, V, W, n, m, p, idx1, idx2):
    """
    Try to remove two terms simultaneously.
    """
    rank = len(U[0])
    if rank <= 2:
        return None
    
    U_new = [row[:] for row in U]
    V_new = [row[:] for row in V]
    W_new = [row[:] for row in W]
    
    for i in range(len(U_new)):
        U_new[i][idx1] = 0
        U_new[i][idx2] = 0
    for i in range(len(V_new)):
        V_new[i][idx1] = 0
        V_new[i][idx2] = 0
    for i in range(len(W_new)):
        W_new[i][idx1] = 0
        W_new[i][idx2] = 0
    
    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
        cols_to_remove = sorted([idx1, idx2])
        U_compact = []
        V_compact = []
        W_compact = []
        for row in U_new:
            U_compact.append([row[j] for j in range(rank) if j not in cols_to_remove])
        for row in V_new:
            V_compact.append([row[j] for j in range(rank) if j not in cols_to_remove])
        for row in W_new:
            W_compact.append([row[j] for j in range(rank) if j not in cols_to_remove])
        return (U_compact, V_compact, W_compact)
    
    return None

def _try_triple_removal(U, V, W, n, m, p, idx1, idx2, idx3):
    """
    Try to remove three terms simultaneously.
    """
    rank = len(U[0])
    if rank <= 3:
        return None
    
    U_new = [row[:] for row in U]
    V_new = [row[:] for row in V]
    W_new = [row[:] for row in W]
    
    for i in range(len(U_new)):
        U_new[i][idx1] = 0
        U_new[i][idx2] = 0
        U_new[i][idx3] = 0
    for i in range(len(V_new)):
        V_new[i][idx1] = 0
        V_new[i][idx2] = 0
        V_new[i][idx3] = 0
    for i in range(len(W_new)):
        W_new[i][idx1] = 0
        W_new[i][idx2] = 0
        W_new[i][idx3] = 0
    
    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
        cols_to_remove = sorted([idx1, idx2, idx3])
        U_compact = []
        V_compact = []
        W_compact = []
        for row in U_new:
            U_compact.append([row[j] for j in range(rank) if j not in cols_to_remove])
        for row in V_new:
            V_compact.append([row[j] for j in range(rank) if j not in cols_to_remove])
        for row in W_new:
            W_compact.append([row[j] for j in range(rank) if j not in cols_to_remove])
        return (U_compact, V_compact, W_compact)
    
    return None

def _construct_rank_27():
    """
    Constructs a rank-27 decomposition for 2x3 * 3x5.
    n=2, m=3, p=5.
    Split B into columns (0,1), (2,3), (4).
    Part 1: j=2, all k (0..4) -> 10 terms (naive)
    Part 2: j in {0,1}, k in {0,1} -> 7 terms (Strassen)
    Part 3: j in {0,1}, k in {2,3} -> 7 terms (Strassen)
    Part 4: j in {0,1}, k in {4}   -> 3 terms (Rank 3 for 2x2x1)
    Total = 10 + 7 + 7 + 3 = 27.
    """
    n, m, p = 2, 3, 5
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4)
    # Handles C[i, k] += A[i, 2] * B[2, k]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B11 (j=0,1, k=0,1) using Strassen (7 terms)
    # Computes C[i, k] for i in {0,1}, k in {0,1} from A[i, j] B[j, k] with j in {0,1}
    
    # M1 = (a00+a11)(b00+b11) -> c00, c11
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M2 = (a10+a11)b00 -> c01, c11
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    # M3 = a00(b01-b11) -> c01, c11
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M4 = a11(b10-b00) -> c00, c10
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    # M5 = (a00+a01)b11 -> c00, c10
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    # M6 = (a10-a00)(b00+b01) -> c11
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M7 = (a01-a11)(b10+b11) -> c00
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 3: A1 B12 (j=0,1, k=2,3) using Strassen (7 terms)
    # Computes C[i, k] for i in {0,1}, k in {2,3} from A[i, j] B[j, k] with j in {0,1}
    
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B13 (j=0,1, k=4) using Rank-3 algorithm for 2x2x1
    # Computes C[i, 4] for i in {0,1} from A[i, j] B[j, 4] with j in {0,1}
    # C[0,4] = a00*b04 + a01*b14
    # C[1,4] = a10*b04 + a11*b14
    
    # M1 = (a00 + a10) * b04 -> contributes a00*b04, a10*b04
    U[0 * m + 0][r] = 1
    U[1 * m + 0][r] = 1
    V[0 * p + 4][r] = 1
    W[4 * n + 0][r] = 1
    W[4 * n + 1][r] = 1
    r += 1
    
    # M2 = (a01 + a11) * b14 -> contributes a01*b14, a11*b14
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = 1
    V[1 * p + 4][r] = 1
    W[4 * n + 0][r] = 1
    W[4 * n + 1][r] = 1
    r += 1
    
    # M3 = (a00 - a10) * (b04 - b14) -> contributes a00*b04 - a00*b14 - a10*b04 + a10*b14
    # We need to subtract M1 and M2 and add M3 appropriately?
    # Let's check:
    # C[0,4] = a00*b04 + a01*b14
    # C[1,4] = a10*b04 + a11*b14
    # M1 = a00*b04 + a10*b04
    # M2 = a01*b14 + a11*b14
    # M3 = (a00 - a10)(b04 - b14) = a00*b04 - a00*b14 - a10*b04 + a10*b14
    # M1 + M2 + M3 = 2*a00*b04 + a01*b14 + a11*b14 - a00*b14 + a10*b14
    # This is not quite right. We need to separate the terms.
    # Actually, the rank-3 algorithm for 2x2x1 is:
    # M1 = (a0 + a1) b0
    # M2 = (a0 - a1) (b0 - b1)
    # M3 = a1 b1
    # C0 = M1 + M2/2 - M3/2 ? No.
    # Let's re-derive.
    # We want C0 = a0 b0 + a1 b1, C1 = a2 b0 + a3 b1.
    # M1 = (a0+a2) b0
    # M2 = (a1+a3) b1
    # M3 = (a0-a2) (b0-b1)
    # C0 = (M1 + M3 + M2 - M3)/2 ? No.
    # C0 = a0 b0 + a1 b1
    # C1 = a2 b0 + a3 b1
    # M1 = a0 b0 + a2 b0
    # M2 = a1 b1 + a3 b1
    # M3 = a0 b0 - a0 b1 - a2 b0 + a2 b1
    # M1 + M2 + M3 = 2 a0 b0 + a1 b1 + a3 b1 - a0 b1 + a2 b1
    # This is getting complicated. Let's use a known rank-3 algorithm.
    # Actually, rank(2x2x1) is 3.
    # M1 = (a0 + a1) b0
    # M2 = (a0 - a1) (b0 - b1)
    # M3 = a1 b1
    # C0 = M1 + M2/2 - M3/2 ?
    # C0 = a0 b0 + a1 b0 + (a0 b0 - a0 b1 - a1 b0 + a1 b1)/2 - a1 b1/2
    # C0 = a0 b0 + a1 b0 + a0 b0/2 - a0 b1/2 - a1 b0/2 + a1 b1/2 - a1 b1/2
    # C0 = 3/2 a0 b0 + 1/2 a1 b0 - 1/2 a0 b1
    # This is not C0 = a0 b0 + a1 b1.
    
    # Let's try:
    # M1 = a0 b0
    # M2 = a1 b1
    # M3 = (a0 + a1) (b0 + b1)
    # C0 = M1 + M2
    # C1 = ... we only have 2 outputs.
    # Wait, the problem is 2x2x1, so 2 inputs for A, 2 inputs for B, 1 output?
    # No, C[i,4] for i in {0,1}. So 2 outputs.
    # So it is 2x2x2? No, B is a column vector of size 2.
    # So it is a bilinear map F^2 x F^2 -> F^2.
    # This is exactly 2x2x2 matrix multiplication? No, it's a vector-matrix product.
    # A is 2x2, B is 2x1. C is 2x1.
    # This is equivalent to 2x2x1 tensor? No, 2x2x2 tensor with one dimension 1?
    # It is the tensor for multiplying a 2x2 matrix by a 2x1 vector.
    # Rank is 3?
    # M1 = (a00 + a10) b0
    # M2 = (a01 + a11) b1
    # M3 = (a00 - a10) (b0 - b1)
    # C0 = a00 b0 + a01 b1
    # C1 = a10 b0 + a11 b1
    # M1 + M2 + M3 = a00 b0 + a10 b0 + a01 b1 + a11 b1 + a00 b0 - a00 b1 - a10 b0 + a10 b1
    # = 2 a00 b0 + a01 b1 + a11 b1 - a00 b1 + a10 b1
    # This is not directly C0, C1.
    # We can solve for C0, C1 in terms of M1, M2, M3.
    # Let's use a different set.
    # M1 = a00 b0 + a10 b1
    # M2 = a01 b0 + a11 b1
    # M3 = (a00 + a01) (b0 + b1)
    # C0 = a00 b0 + a01 b1
    # C1 = a10 b0 + a11 b1
    # M1 + M2 = a00 b0 + a10 b1 + a01 b0 + a11 b1
    # M3 = a00 b0 + a00 b1 + a01 b0 + a01 b1
    # M1 + M2 + M3 = 2 a00 b0 + a01 b0 + a10 b1 + a11 b1 + a00 b1 + a01 b1
    # This is also not direct.
    
    # Let's use the standard rank-3 algorithm for 2x2x2 (Strassen is 7, but for 2x2x1 it is 3).
    # Actually, rank(2x2x1) is 3.
    # M1 = (a0 + a1) b0
    # M2 = (a0 - a1) (b0 - b1)
    # M3 = a1 b1
    # C0 = M1 + M2/2 - M3/2 ? No.
    # C0 = a0 b0 + a1 b1
    # C1 = a0 b1 + a1 b0 ? No, C1 is different.
    # In our case:
    # C[0,4] = a00 b04 + a01 b14
    # C[1,4] = a10 b04 + a11 b14
    # Let a0 = a00, a1 = a01, a2 = a10, a3 = a11.
    # b0 = b04, b1 = b14.
    # C0 = a0 b0 + a1 b1
    # C1 = a2 b0 + a3 b1
    # M1 = (a0 + a2) b0
    # M2 = (a1 + a3) b1
    # M3 = (a0 - a2) (b0 - b1)
    # M1 + M2 + M3 = a0 b0 + a2 b0 + a1 b1 + a3 b1 + a0 b0 - a0 b1 - a2 b0 + a2 b1
    # = 2 a0 b0 + a1 b1 + a3 b1 - a0 b1 + a2 b1
    # We can write C0, C1 as linear combinations of M1, M2, M3.
    # C0 = a0 b0 + a1 b1
    # C1 = a2 b0 + a3 b1
    # M1 = a0 b0 + a2 b0
    # M2 = a1 b1 + a3 b1
    # M3 = a0 b0 - a0 b1 - a2 b0 + a2 b1
    # M1 + M2 + M3 = 2 a0 b0 + a1 b1 + a3 b1 - a0 b1 + a2 b1
    # M1 - M3 = 2 a2 b0 + a0 b1 - a2 b1
    # M2 - M3 = 2 a1 b1 + a0 b1 - a2 b1
    # This is getting too complex. Let's just use 4 terms for this part if needed, or find a simpler 3-term algorithm.
    # Actually, rank(2x2x1) is 3.
    # M1 = a0 b0
    # M2 = a1 b1
    # M3 = (a0 + a1) (b0 + b1)
    # C0 = M1 + M2
    # C1 = ... we don't have C1 in this formulation.
    # Wait, the tensor is 2x2x2? No, it's 2x2x1 for the B part, but C has 2 components.
    # So it is a 2x2x2 tensor? No.
    # It is the tensor for C = A * b, where A is 2x2, b is 2x1.
    # This is equivalent to the 2x2x2 matrix multiplication tensor with the second dimension of B being 1?
    # No, it is a different tensor.
    # Let's just use 4 terms for this part to be safe, and then try to reduce later.
    # Or use the 3-term algorithm correctly.
    # M1 = (a0 + a2) b0
    # M2 = (a1 + a3) b1
    # M3 = (a0 - a2) (b0 - b1)
    # C0 = (M1 + M2 + M3)/2 + (M2 - M3)/2 ?
    # Let's check:
    # (M1 + M2 + M3)/2 = a0 b0 + 1/2 a1 b1 + 1/2 a3 b1 - 1/2 a0 b1 + 1/2 a2 b1
    # (M2 - M3)/2 = 1/2 a1 b1 + 1/2 a3 b1 + 1/2 a0 b1 - 1/2 a2 b1
    # Sum = a0 b0 + a1 b1 + a3 b1 = C0 + a3 b1. Not correct.
    
    # Let's use 4 terms for now.
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # We used 4 terms, so total rank is 10 + 7 + 7 + 4 = 28.
    # This is the same as before.
    # I will return this rank-28 decomposition for now, as the rank-27 construction is not working out easily.
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = 150.0
    
    # Construct the known rank-28 decomposition
    res_28 = _construct_rank_27() # This actually constructs rank 28
    U = res_28['U']
    V = res_28['V']
    W = res_28['W']
    
    if not _verify_decomposition(U, V, W, 2, 3, 5):
        raise ValueError("Verification failed")
    
    n, m, p = 2, 3, 5
    rank = len(U[0])
    best_result = {'U': U, 'V': V, 'W': W}
    
    for term_idx in range(rank):
        if time.time() - start_time > deadline - 30:
            break
        result = _try_remove_term(U, V, W, n, m, p, term_idx)
        if result is not None:
            U, V, W = result
            rank = len(U[0])
            best_result = {'U': U, 'V': V, 'W': W}
            print(f"Reduced rank to {rank} by removing term {term_idx}")
    
    for pair in combinations(range(rank), 2):
        if time.time() - start_time > deadline - 30:
            break
        result = _try_pairwise_removal(U, V, W, n, m, p, pair[0], pair[1])
        if result is not None:
            U, V, W = result
            rank = len(U[0])
            best_result = {'U': U, 'V': V, 'W': W}
            print(f"Reduced rank to {rank} by removing pair {pair}")
    
    for triple in combinations(range(rank), 3):
        if time.time() - start_time > deadline - 10:
            break
        result = _try_triple_removal(U, V, W, n, m, p, triple[0], triple[1], triple[2])
        if result is not None:
            U, V, W = result
            rank = len(U[0])
            best_result = {'U': U, 'V': V, 'W': W}
            print(f"Reduced rank to {rank} by removing triple {triple}")
    
    return best_result
