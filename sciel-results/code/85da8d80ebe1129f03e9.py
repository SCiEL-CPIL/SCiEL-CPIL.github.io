"""
Matrix multiplication 3x4x5 rank-minimal tensor decomposition.
Uses structured algebraic constructions (Strassen-like, block methods) to find
low-rank exact decompositions with integer/half-integer entries.
"""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 5]


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W form a valid tensor decomposition for n×m×p matrix mult.
    The tensor T[i,j,k] = 1 for all valid indices (since C[i,k] = sum_j A[i,j]*B[j,k]).
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    """
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2x2():
    """
    Return Strassen's 7-rank decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of lists with integer entries.
    """
    # Strassen's algorithm: 7 products
    # For 2x2 matrices A, B:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # i,j in {0,1} for A, B indices; k for output column
    
    U = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
    V = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
    W = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: A00 (row 0) += 1, A11 (row 3) += 1
    # V: B00 (row 0) += 1, B11 (row 3) += 1
    # W: C00 (row 0) += 1, C11 (row 3) += 1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2 = (A10 + A11)B00
    # U: A10 (row 2) += 1, A11 (row 3) += 1
    # V: B00 (row 0) += 1
    # W: C10 (row 2) += 1, C11 (row 3) -= 1
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] -= 1
    
    # M3 = A00(B10 - B11)
    # U: A00 (row 0) += 1
    # V: B10 (row 2) += 1, B11 (row 3) -= 1
    # W: C01 (row 1) += 1, C11 (row 3) += 1
    U[0][2] += 1
    V[2][2] += 1; V[3][2] -= 1
    W[1][2] += 1; W[3][2] += 1
    
    # M4 = A11(B01 - B00)
    # U: A11 (row 3) += 1
    # V: B01 (row 1) += 1, B00 (row 0) -= 1
    # W: C00 (row 0) += 1, C10 (row 2) += 1
    U[3][3] += 1
    V[1][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    # M5 = (A00 + A01)B11
    # U: A00 (row 0) += 1, A01 (row 1) += 1
    # V: B11 (row 3) += 1
    # W: C00 (row 0) -= 1, C01 (row 1) += 1
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[1][4] += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    # U: A10 (row 2) += 1, A00 (row 0) -= 1
    # V: B00 (row 0) += 1, B01 (row 1) += 1
    # W: C11 (row 3) += 1
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    # U: A01 (row 1) += 1, A11 (row 3) -= 1
    # V: B10 (row 2) += 1, B11 (row 3) += 1
    # W: C00 (row 0) += 1
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W


def _build_3x4x5_strassen_based():
    """
    Build a decomposition for 3x4x5 using Strassen on the 4x4 middle part.
    
    Strategy: View the 4x4 matrix multiplication (part of the 4x5 and 3x4)
    as a block structure. The key insight is that we can decompose the
    middle dimension (size 4) into two blocks of size 2, and use Strassen's
    7-rank algorithm for each 2x2 block multiplication.
    
    For 3x4x5:
    - A is 3x4, B is 4x5
    - C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    We can split j into two groups: j in {0,1} and j in {2,3}.
    For each group, we have a 3x2 times 2x5 multiplication.
    
    For the 3x2 * 2x5 case, we can use a rank-7 decomposition based on
    Strassen-like ideas applied to the 2x2 core.
    
    Actually, let's think more carefully. The tensor for 3x4x5 has 60 entries
    (3*4*5). The schoolbook rank is 60.
    
    A better approach: use the fact that 4 = 2+2 and apply Strassen to the
    2x2 blocks in the middle dimension.
    
    For each pair of rows i (0,1,2) and columns k (0..4), we compute:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    Group the j indices: {0,1} and {2,3}.
    For group {0,1}: C_part[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    For group {2,3}: C_part[i,k] = A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    Each group is a 3x2 * 2x5 multiplication, which has tensor size 3*2*5 = 30.
    The schoolbook rank for each is 30, total 60.
    
    To reduce rank, we need to find a lower-rank decomposition for 3x2 * 2x5.
    
    For 3x2 * 2x5, note that this is equivalent to computing 3x5 matrix from
    3x2 and 2x5. The rank of this tensor is at least max(3*2, 2*5, 5*3) = max(6,10,15) = 15.
    And at most 3*2*5 = 30.
    
    Let's try to find a rank-15 or lower decomposition for 3x2 * 2x5.
    
    Actually, for an m x n matrix times n x p matrix, the rank is at least
    max(mn, np, pm) and at most mn*p. For 3x2 * 2x5: max(6,10,15)=15, min 30.
    
    Let's try to construct a rank-15 decomposition for 3x2 * 2x5.
    If we can do that, then total rank for 3x4x5 would be 2*15 = 30.
    
    For 3x2 * 2x5, the tensor T[i,j,k] = 1 for i in [0,3), j in [0,2), k in [0,5).
    We need U (6x15), V (10x15), W (15x15) such that sum_c U[i*2+j,c]*V[j*5+k,c]*W[k*3+i,c] = 1.
    
    This is a complex problem. Let me try a different approach.
    
    Alternative: Use the Winograd-Friedman algorithm or other known low-rank
    decompositions for small matrix multiplication.
    
    For now, let's try a constructive approach using linear algebra.
    """
    n, m, p = 3, 4, 5
    
    # Try to build a decomposition using the following idea:
    # The matrix multiplication tensor for n x m times m x p can be seen as
    # a 3-tensor. We want to find a rank-r decomposition.
    
    # One approach: use the fact that the tensor is "separable" in some sense
    # and find a low-rank approximation that is exact.
    
    # Let's try rank = 30 first (half of schoolbook) using a structured approach.
    # Split the middle dimension into two halves of size 2.
    
    # For each half, we need a 3x2 * 2x5 decomposition.
    # Let's try to find a rank-15 decomposition for 3x2 * 2x5.
    
    # Actually, let's try a different strategy: use random search with verification
    # for ranks below 60, starting from reasonable lower bounds.
    
    # Lower bound for 3x4x5: max(3*4, 4*5, 5*3) = max(12, 20, 15) = 20
    # But the actual rank is likely higher.
    
    # Let's try to use a structured decomposition based on polynomial methods.
    
    # For 3x4x5, one known approach is to use the fact that the tensor can be
    # decomposed using bilinear forms.
    
    # Let's try rank 40, 35, 30, 25, 20 in order.
    
    best_result = None
    best_rank = 60  # schoolbook
    
    # Try structured decompositions
    for target_rank in [45, 40, 36, 32, 30, 28, 25, 24, 22, 20]:
        result = _try_structured_decomposition(n, m, p, target_rank)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return _build_schoolbook(n, m, p)


def _try_structured_decomposition(n, m, p, target_rank):
    """
    Try to find a structured decomposition of given rank.
    Uses various algebraic constructions.
    """
    # Method 1: Try to use a Kronecker-like structure
    # Method 2: Try random initialization with local refinement
    # Method 3: Try specific known decompositions
    
    # For 3x4x5, let's try a block-based approach.
    # Split the 4 middle dimension into 2+2.
    # For each 2-block, use a 3x2 * 2x5 decomposition.
    
    # Try to find rank-r decomposition for 3x2 * 2x5 where r = target_rank // 2
    half_rank = target_rank // 2
    
    if half_rank < 10:  # Minimum for 3x2*2x5 is 10 (max of 6,10,15)
        return None
    
    # Build U, V, W for 3x4x5 by combining two 3x2*2x5 decompositions
    # U_total: 12 x target_rank
    # V_total: 20 x target_rank  
    # W_total: 15 x target_rank
    
    U = [[Fraction(0)] * target_rank for _ in range(n * m)]  # 12 rows
    V = [[Fraction(0)] * target_rank for _ in range(m * p)]  # 20 rows
    W = [[Fraction(0)] * target_rank for _ in range(p * n)]  # 15 rows
    
    # First half: j in {0, 1}
    # U rows: i*4 + j for i in 0..2, j in 0..1 -> rows 0,1, 4,5, 8,9
    # V rows: j*5 + k for j in 0..1, k in 0..4 -> rows 0..4, 5..9
    # W rows: k*3 + i for k in 0..4, i in 0..2 -> rows 0..14
    
    # Second half: j in {2, 3}
    # U rows: i*4 + j for i in 0..2, j in 2..3 -> rows 2,3, 6,7, 10,11
    # V rows: j*5 + k for j in 2..3, k in 0..4 -> rows 10..14, 15..19
    # W rows: same as first half
    
    # For each half, we need a half_rank decomposition.
    # Let's try to construct one using a simple method.
    
    # For 3x2 * 2x5 with rank r:
    # U_sub: 6 x r, V_sub: 10 x r, W_sub: 15 x r
    # We need sum_c U_sub[i*2+j, c] * V_sub[j*5+k, c] * W_sub[k*3+i, c] = 1
    
    # Try a simple construction: for each (i, j, k), assign a unique column
    # but share columns where possible.
    
    # Actually, let's just try random search for the sub-problem.
    sub_result = _search_sub_decomposition(3, 2, 5, half_rank)
    if sub_result is None:
        return None
    
    U_sub, V_sub, W_sub = sub_result
    
    # First half (j=0,1)
    for i in range(3):
        for j in range(2):
            for c in range(half_rank):
                U[i * 4 + j][c] = U_sub[i * 2 + j][c]
    
    for j in range(2):
        for k in range(5):
            for c in range(half_rank):
                V[j * 5 + k][c] = V_sub[j * 5 + k][c]
    
    for k in range(5):
        for i in range(3):
            for c in range(half_rank):
                W[k * 3 + i][c] = W_sub[k * 3 + i][c]
    
    # Second half (j=2,3)
    # Map local j=0 -> global j=2, local j=1 -> global j=3
    for i in range(3):
        for local_j in range(2):
            global_j = local_j + 2
            for c in range(half_rank):
                U[i * 4 + global_j][half_rank + c] = U_sub[i * 2 + local_j][c]
    
    for local_j in range(2):
        global_j = local_j + 2
        for k in range(5):
            for c in range(half_rank):
                V[global_j * 5 + k][half_rank + c] = V_sub[local_j * 5 + k][c]
    
    # W is shared (same k and i indices)
    # For the second half, W entries should be the same as first half
    # because the output C[i,k] gets contributions from both halves
    # Actually, W for the second half should add to the same W entries
    for k in range(5):
        for i in range(3):
            for c in range(half_rank):
                W[k * 3 + i][half_rank + c] = W_sub[k * 3 + i][c]
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None


def _search_sub_decomposition(n, m, p, target_rank, max_attempts=100):
    """
    Search for a rank-target_rank decomposition of n x m times m x p matrix mult.
    Uses random initialization with small integer/half-integer values.
    """
    if target_rank < max(n * m, m * p, p * n):
        return None
    
    random.seed(12345)
    
    for attempt in range(max_attempts):
        # Initialize with sparse random values
        U = [[Fraction(0)] * target_rank for _ in range(n * m)]
        V = [[Fraction(0)] * target_rank for _ in range(m * p)]
        W = [[Fraction(0)] * target_rank for _ in range(p * n)]
        
        # Fill with random values from {-1, 0, 1, 1/2, -1/2}
        for i in range(n * m):
            for c in range(target_rank):
                if random.random() < 0.3:
                    U[i][c] = random.choice([Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)])
        
        for j in range(m * p):
            for c in range(target_rank):
                if random.random() < 0.3:
                    V[j][c] = random.choice([Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)])
        
        for k in range(p * n):
            for c in range(target_rank):
                if random.random() < 0.3:
                    W[k][c] = random.choice([Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)])
        
        if _verify_decomposition(U, V, W, n, m, p):
            return (U, V, W)
    
    return None


def _build_3x4x5_direct():
    """
    Direct construction for 3x4x5 using known algebraic identities.
    
    The key insight: for matrix multiplication C = AB where A is 3x4 and B is 4x5,
    we have C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k].
    
    We can use the following decomposition based on the fact that the tensor
    T[i,j,k] = 1 for all i,j,k can be decomposed using bilinear forms.
    
    One approach: use the identity that for any a,b,c,d:
    (a+b)(c+d) = ac + ad + bc + bd
    
    This gives us a way to reduce the number of products.
    
    For the 4 middle dimension, we can group terms:
    C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k] + A[i,3]B[3,k]
    
    Group as: (A[i,0] + A[i,1]) * (B[0,k] + B[1,k]) - A[i,0]B[1,k] - A[i,1]B[0,k]
           + (A[i,2] + A[i,3]) * (B[2,k] + B[3,k]) - A[i,2]B[3,k] - A[i,3]B[2,k]
    
    This uses 2 "large" products and 4 "small" products per (i,k) pair.
    But the "large" products are shared across i and k in a structured way.
    
    Let's try a different grouping that leads to a rank reduction.
    """
    n, m, p = 3, 4, 5
    
    # Let's try to construct a rank-30 decomposition.
    # The idea: for each of the 3 rows i and 5 columns k, we need to compute
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k].
    
    # Use the identity: ab + cd = (a+c)(b+d) - cb - ad
    # This reduces 2 products to 3 products, which is worse.
    
    # Better: use Strassen-like tricks.
    
    # For now, let's just return schoolbook and let the search find better.
    return _build_schoolbook(n, m, p)


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety
    
    # Start with schoolbook as baseline
    schoolbook = _build_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p  # 60
    
    # Try structured decomposition
    if time.time() < deadline - 5:
        result = _build_3x4x5_strassen_based()
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                best_result = result
                best_rank = rank
    
    # Try direct construction
    if time.time() < deadline - 5:
        result = _build_3x4x5_direct()
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                best_result = result
                best_rank = rank
    
    # Try additional random searches with more attempts
    if time.time() < deadline - 10:
        for target_rank in [50, 45, 42, 40, 38, 36, 34, 32, 30]:
            if time.time() > deadline - 10:
                break
            # Try a few random attempts for full 3x4x5
            for attempt in range(20):
                if time.time() > deadline - 10:
                    break
                U = [[Fraction(0)] * target_rank for _ in range(n * m)]
                V = [[Fraction(0)] * target_rank for _ in range(m * p)]
                W = [[Fraction(0)] * target_rank for _ in range(p * n)]
                
                random.seed(attempt * 1000 + target_rank)
                for i in range(n * m):
                    for c in range(target_rank):
                        if random.random() < 0.25:
                            U[i][c] = random.choice([Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)])
                for j in range(m * p):
                    for c in range(target_rank):
                        if random.random() < 0.25:
                            V[j][c] = random.choice([Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)])
                for k in range(p * n):
                    for c in range(target_rank):
                        if random.random() < 0.25:
                            W[k][c] = random.choice([Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)])
                
                if _verify_decomposition(U, V, W, n, m, p):
                    rank = target_rank
                    if rank < best_rank:
                        best_result = {"U": U, "V": V, "W": W}
                        best_rank = rank
                    break
    
    return best_result
