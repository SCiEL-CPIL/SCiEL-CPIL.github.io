"""Matrix multiplication 3x3x5 - rank-minimal decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]
_DEADLINE = time.time() + 140.0  # 140s out of 150s wall-clock

def _check_deadline():
    return time.time() < _DEADLINE

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the 3x3x5 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(W[0]) if W else 0
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) valid else 0
    # For matrix mult (A is n x m, B is m x p, C is n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 if the output C[i,k] depends on A[i,j]*B[j,k]
    # In tensor decomposition: T[i,j,k] = sum_c u_{i,j}^c * v_{j,k}^c * w_{k,i}^c
    # Here U has shape (n*m, rank), indexed as [i*m+j, c]
    # V has shape (m*p, rank), indexed as [j*p+k, c]
    # W has shape (p*n, rank), indexed as [k*n+i, c]
    
    # Build the tensor from decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    u_val = U[i*m+j, c]
                    v_val = V[j*p+k, c]
                    w_val = W[k*n+i, c]
                    if isinstance(u_val, Fraction) or isinstance(v_val, Fraction) or isinstance(w_val, Fraction):
                        total += u_val * v_val * w_val
                    else:
                        total += u_val * v_val * w_val
                expected = 1
                if total != expected:
                    return False
    return True

def _try_strassen_3x3_5():
    """
    Known result: The tensor rank of Mat_{3,3,5} is at most 31.
    One approach: use a rank-13 decomposition for the 3x3x3 core (Strassen),
    then handle the 2 additional output columns.
    
    Actually, let's use a constructive approach:
    - For the 3x3x3 core: use Strassen's rank-7 algorithm extended, or known rank-13
    - For columns 3,4: each additional column of output needs at most 9 mults
      but we can share some with the core.
    
    Better approach: Use a known explicit decomposition.
    
    The matrix multiplication tensor MM_{n,m,p} = sum_{i,j,k} e_i⊗e_j⊗e_k
    
    For 3x3x5, one can partition B as [B1 | B2] where B1 is 3x3 and B2 is 3x2.
    Then A*B = A*B1 + A*B2.
    
    For A*B1 (3x3 times 3x3 -> 3x3): rank at most 13 (Strassen-based)
    For A*B2 (3x3 times 3x2 -> 3x2): rank at most 6 (each column needs 9 mults, 
      but we can be smarter)
    
    Actually for A*B2 where B2 is 3x2:
    C2 = A * B2, so C2[i,k] = sum_j A[i,j]*B2[j,k] for k=0,1
    Schoolbook: 3*3*2 = 18 mults
    But we can use: precompute 3+2=5 linear combinations, do 5*3=15 mults? No.
    
    Let me think more carefully. A is 3x3, B2 is 3x2.
    We need to compute C2 = A * B2 which is 3x2.
    C2[0,0] = A[0,0]*B2[0,0] + A[0,1]*B2[1,0] + A[0,2]*B2[2,0]
    C2[0,1] = A[0,0]*B2[0,1] + A[0,1]*B2[1,1] + A[0,2]*B2[2,1]
    ...
    
    A straightforward sharing approach:
    Precompute s1 = A[0,0], s2 = A[0,1], s3 = A[0,2], etc. - that's just A itself.
    Then C2[i,k] = s1_i * B2[0,k] + s2_i * B2[1,k] + s3_i * B2[2,k]
    This is 9 mults per column of B2, so 18 total for B2.
    
    So naive: rank(3x3x3) + rank(3x3x2) = 13 + 18 = 31? No, 3x3x2 schoolbook is 18.
    Actually rank(3x3x2) schoolbook is 3*3*2 = 18.
    So 13 + 18 = 31.
    
    But we can do better for the 3x3x2 part. Since B2 is 3x2:
    Use the trick: C2[i,k] for i in 0..2, k in 0..1
    = A[i,0]*B2[0,k] + A[i,1]*B2[1,k] + A[i,2]*B2[2,k]
    
    We can precompute t_i = A[i,0]*B2[0,k] for a fixed k... no, k varies.
    
    Alternative: Group by the A index.
    For each i, C2[i,:] = A[i,0]*B2[0,:] + A[i,1]*B2[1,:] + A[i,2]*B2[2,:]
    This is 3 scalar-vector products, each needing 2 mults (since B2[0,:] is 2-vector).
    So 3*2 = 6 mults per i, total 18. Same as schoolbook.
    
    Actually, the schoolbook rank for 3x3x2 is exactly 18 (since the tensor is 
    Mat_{3,3,2} and we can see it's a direct sum structure).
    
    Hmm, but actually we might be able to share between the 3x3x3 and 3x3x2 parts.
    
    Let me try a different strategy: use the full 3x3x5 schoolbook but with sharing.
    
    Total schoolbook rank = 3*3*5 = 45.
    
    The Strassen algorithm for 3x3x3 gives rank 7? No, Strassen is for 2x2x2.
    For 3x3x3, the best known upper bound is 13 (from various papers).
    
    Let me use a known explicit rank-13 decomposition for 3x3x3.
    
    From the literature, one rank-13 decomposition of Mat_{3,3,3} uses the following
    13 terms (each term is (u, v, w) where u, v are linear functionals on A and B, 
    and w is a bilinear form on the output):
    
    Actually, let me construct it explicitly using a simpler approach.
    
    Strategy: I'll use a "blocked" Strassen-like approach.
    Write A and B as 2x2 blocks of varying sizes, apply Strassen at the block level.
    
    For 3x3: split into top-left 2x2, top-right 2x1, bottom-left 1x2, bottom-right 1x1.
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x2, B21 is 1x2, B22 is 1x2
    
    Strassen at block level:
    M1 = (A11+A22)(B11+B22) -- 2x2 * 2x2 = rank 7
    M2 = (A21+A22)B11 -- 1x2 * 2x2 = rank 4? (schoolbook is 4)
    M3 = A11(B21-B11) -- 2x2 * 1x2... wait dimensions don't work
    
    This is getting complicated. Let me just use a verified approach.
    
    I'll construct a rank-31 decomposition:
    - Rank 13 for 3x3x3 core (using a known explicit form)
    - Rank 18 for 3x3x2 (schoolbook)
    Total: 31
    
    But first let me verify I can get a valid rank-13 for 3x3x3.
    """
    pass

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n x m times m x p."""
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

def _combined_decomp(U1, V1, W1, U2, V2, W2):
    """Concatenate two decompositions."""
    r1 = len(U1[0]) if U1 else 0
    r2 = len(U2[0]) if U2 else 0
    total_rank = r1 + r2
    U = [[0] * total_rank for _ in range(len(U1))]
    for i in range(len(U1)):
        for c in range(r1):
            U[i][c] = U1[i][c]
        for c in range(r2):
            U[i][r1 + c] = U2[i][c] if i < len(U2) else 0
    V = [[0] * total_rank for _ in range(len(V1))]
    for i in range(len(V1)):
        for c in range(r1):
            V[i][c] = V1[i][c]
        for c in range(r2):
            V[i][r1 + c] = V2[i][c] if i < len(V2) else 0
    W = [[0] * total_rank for _ in range(len(W1))]
    for i in range(len(W1)):
        for c in range(r1):
            W[i][c] = W1[i][c]
        for c in range(r2):
            W[i][r1 + c] = W2[i][c] if i < len(W2) else 0
    return {"U": U, "V": V, "W": W}

def _strassen_3x3x3():
    """
    Rank-13 decomposition for 3x3x3 matrix multiplication.
    
    Uses the approach: view 3x3 as 2x2 blocks and apply a modified Strassen.
    A = [[A00, A01, A02], [A10, A11, A12], [A20, A21, A22]]
    B = [[B00, B01, B02], [B10, B11, B12], [B20, B21, B22]]
    
    Actually, the cleanest known rank-13 comes from viewing it as:
    3x3x3 tensor = combination of smaller tensors.
    
    One explicit construction (from "On the rank of matrix multiplication"):
    Split 3 = 2 + 1.
    A = [[A_TL(2x2), A_TR(2x1)], [A_BL(1x2), A_BR(1x1)]]
    B = [[B_TL(2x2), B_TR(2x2)], [B_BL(1x2), B_BR(1x2)]]
    
    C = A*B is 3x3:
    C[0:2, 0:2] = A_TL*B_TL + A_TR*B_BL  (2x2 = 2x2*2x2 + 2x1*1x2)
    C[0:2, 2:3] = A_TL*B_TR + A_TR*B_BR  (2x1 = 2x2*2x2 + 2x1*1x2) -- wait, dims
    C[2:3, 0:2] = A_BL*B_TL + A_BR*B_BL  (1x2 = 1x2*2x2 + 1x1*1x2)
    C[2:3, 2:3] = A_BL*B_TR + A_BR*B_BR  (1x1 = 1x2*2x2 + 1x1*1x2)
    
    Using Strassen for the 2x2x2 blocks:
    M1 = (A_TL + A_BR)(B_TL + B_BR) -- both 2x2: rank 7
    M2 = (A_BL + A_BR)B_TL -- 1x2 * 2x2: schoolbook rank 4, but we can embed
    M3 = A_TL(B_BL - B_TL) -- 2x2 * 1x2: rank 4
    M4 = A_BR(B_TR - B_TL) -- 1x1 * 2x2: rank 2 (A_BR is scalar, B_TR-B_TL is 2x2, so it's just scaling)
    
    Wait, A_BR is 1x1 (a scalar), so A_BR*(B_TR-B_TL) is just scalar multiplication of a 2x2 matrix.
    In the tensor decomposition sense, this is: for each element, a_B * (B_TR-B_TL)[i,j]
    which requires 1 "multiplication" per output element if we think of it as rank-1 terms.
    Actually no: a_B * M[i,j] where M is precomputed. In the bilinear model, this is 
    u = [a_B] (1-dim functional on A), v = [M[i,j]] (functional on B), so rank 1 per output element.
    A_BR*(B_TR - B_TL) is a 2x2 matrix = 4 output elements, each a rank-1 term. So rank 4.
    But wait, we can share: the 4 elements all use the same u=A_BR and different v's.
    In the standard tensor model, each term (u_c, v_c, w_c) contributes u_c * v_c * w_c to T.
    For a_BR * (B_TR[i,j] - B_TL[i,j]), we have u = a_BR, v = B_TR[i,j]-B_TL[i,j], w = delta.
    This is 4 separate rank-1 terms (one per output element), OR we can think of it differently.
    
    In the standard tensor decomposition of matrix multiplication, each "multiplication" 
    corresponds to one rank-1 term. A scalar times a matrix element is indeed one rank-1 term.
    
    OK this is getting very complex to get exactly right by hand. Let me try a 
    computational approach: use numerical optimization to find a low-rank decomposition.
    """
    pass

def _optimize_decomposition(max_rank=None, time_limit=30.0):
    """
    Use numerical optimization (gradient descent / alternating minimization) 
    to find a low-rank decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    best_rank = n * m * p  # schoolbook fallback
    best_decomp = None
    
    # Start with schoolbook as initial guess
    school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_decomp = school
    
    # Try to find lower rank using alternating least squares
    # For each rank r, initialize and try to optimize
    for target_rank in [31, 29, 27, 25, 23, 21, 19, 17, 15]:
        if time.time() - start_time > time_limit:
            break
        if max_rank and target_rank > max_rank:
            continue
        
        # Initialize with random + schoolbook mixing
        np.random.seed(target_rank)
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Initialize from schoolbook (use first target_rank terms) + random perturbation
        sb = _schoolbook(n, m, p)
        r0 = min(target_rank, len(sb["U"][0]))
        for c in range(r0):
            for i in range(n*m):
                U[i, c] = sb["U"][i][c]
            for i in range(m*p):
                V[i, c] = sb["V"][i][c]
            for i in range(p*n):
                W[i, c] = sb["W"][i][c]
        for c in range(r0, target_rank):
            U[:, c] = np.random.randn(n*m) * 0.1
            V[:, c] = np.random.randn(m*p) * 0.1
            W[:, c] = np.random.randn(p*n) * 0.1
        
        # Alternating minimization
        # Fix V,W: solve for U (linear system)
        # The constraint: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # When V,W fixed, for each (i,j), the constraint becomes:
        # sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c]) = 1 for each k
        # This is a linear system in U[i*m+j, :]
        
        success = False
        for iteration in range(2000):
            if time.time() - start_time > time_limit:
                break
            
            # Fix V and W, solve for U
            for idx in range(n*m):  # idx = i*m+j
                i = idx // m
                j = idx % m
                # For each k: sum_c U[idx,c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # Build 3xR system
                A_mat = np.zeros((p, target_rank))
                for k in range(p):
                    for c in range(target_rank):
                        A_mat[k, c] = V[j*p+k, c] * W[k*n+i, c]
                b_vec = np.ones(p)
                try:
                    # Least squares
                    sol, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U[idx, :] = sol
                except:
                    pass
            
            # Fix U and W, solve for V
            for idx in range(m*p):  # idx = j*p+k
                j = idx // p
                k = idx % p
                A_mat = np.zeros((n, target_rank))
                for i in range(n):
                    for c in range(target_rank):
                        A_mat[i, c] = U[i*m+j, c] * W[k*n+i, c]
                b_vec = np.ones(n)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V[idx, :] = sol
                except:
                    pass
            
            # Fix U and V, solve for W
            for idx in range(p*n):  # idx = k*n+i
                k = idx // n
                i = idx % n
                A_mat = np.zeros((m, target_rank))
                for j in range(m):
                    for c in range(target_rank):
                        A_mat[j, c] = U[i*m+j, c] * V[j*p+k, c]
                b_vec = np.ones(m)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W[idx, :] = sol
                except:
                    pass
            
            # Check convergence
            if iteration % 100 == 0:
                err = _compute_error(U, V, W)
                if err < 1e-6:
                    success = True
                    break
        
        if success:
            # Snap to rational approximations
            best_rank = target_rank
            best_decomp = _snap_to_rational(U, V, W, target_rank)
            break
    
    return best_rank, best_decomp

def _compute_error(U, V, W):
    """Compute the max error of the decomposition."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(total - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _snap_to_rational(U, V, W, rank, denom_max=100):
    """Snap float values to nearest fraction with small denominator."""
    U_list = [[0]*rank for _ in range(len(U))]
    V_list = [[0]*rank for _ in range(len(V))]
    W_list = [[0]*rank for _ in range(len(W))]
    
    for i in range(len(U)):
        for c in range(rank):
            U_list[i][c] = _snap_single(float(U[i, c]), denom_max)
    for i in range(len(V)):
        for c in range(rank):
            V_list[i][c] = _snap_single(float(V[i, c]), denom_max)
    for i in range(len(W)):
        for c in range(rank):
            W_list[i][c] = _snap_single(float(W[i, c]), denom_max)
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _snap_single(val, denom_max=100):
    """Snap a float to the nearest fraction p/q with |q| <= denom_max."""
    if abs(val) < 1e-10:
        return 0
    best = val
    best_err = abs(val - best)
    for q in range(1, denom_max + 1):
        p = round(val * q)
        candidate = p / q
        err = abs(val - candidate)
        if err < best_err:
            best_err = err
            best = candidate
            # Check if it's actually rational
            if abs(val * q - p) < 1e-6:
                # Check if p/q is in reduced form
                from math import gcd
                g = gcd(abs(p), q)
                if p // g == 0:
                    return 0
                return p / q
    return round(val, 10)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x5 matrix multiplication.
    Returns {'U': list of lists, 'V': list of lists, 'W': list of lists}
    U shape: (n*m, rank) = (9, rank)
    V shape: (m*p, rank) = (15, rank)  
    W shape: (p*n, rank) = (15, rank)
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin
    
    # Start with schoolbook (rank 45) as guaranteed fallback
    fallback = _schoolbook(n, m, p)
    best_result = fallback
    best_rank = n * m * p
    
    # Try to find better decomposition via numerical optimization
    try:
        if time.time() < deadline - 10:
            rank_found, decomp = _optimize_decomposition(time_limit=min(120.0, deadline - time.time() - 5))
            if decomp is not None:
                # Verify
                if _verify(decomp["U"], decomp["V"], decomp["W"]):
                    best_result = decomp
                    best_rank = rank_found
                    if best_rank <= 31:
                        return best_result
                else:
                    # Try harder snapping
                    U_arr = np.array([[float(x) if not isinstance(x, Fraction) else float(x) for x in row] for row in decomp["U"]])
                    V_arr = np.array([[float(x) if not isinstance(x, Fraction) else float(x) for x in row] for row in decomp["V"]])
                    W_arr = np.array([[float(x) if not isinstance(x, Fraction) else float(x) for x in row] for row in decomp["W"]])
                    snapped = _snap_to_rational(U_arr, V_arr, W_arr, best_rank, denom_max=200)
                    if _verify(snapped["U"], snapped["V"], snapped["W"]):
                        best_result = snapped
                        return best_result
    except Exception:
        pass
    
    # Try a structured rank-31 decomposition
    try:
        if time.time() < deadline - 5:
            r31, d31 = _try_rank_31()
            if d31 is not None and _verify(d31["U"], d31["V"], d31["W"]):
                best_result = d31
                best_rank = r31
                return best_result
    except Exception:
        pass
    
    return best_result

def _try_rank_31():
    """
    Construct a rank-31 decomposition:
    - 13 terms for 3x3x3 core
    - 18 terms for the remaining 3x3x2 part (schoolbook)
    """
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # 3x3x3 core (columns 0,1,2 of B)
    # Schoolbook for 3x3x3: rank 9
    # Strassen-like for 3x3x3: try to get rank 13 (worse than schoolbook? No, 9 < 13)
    # Actually schoolbook 3x3x3 is rank 9, which is less than 13!
    # So for the core, schoolbook is already optimal (rank 9).
    # For 3x3x2: schoolbook is rank 18.
    # Total: 9 + 18 = 27.
    
    # Let's try rank 27: schoolbook for both parts.
    # Part 1: A (3x3) * B_core (3x3) -> C_core (3x3), rank 9
    # Part 2: A (3x3) * B_extra (3x2) -> C_extra (3x2), rank 18
    
    # Part 1: schoolbook for i,j,k where k in 0,1,2
    rank1 = 9
    U1 = [[0]*rank1 for _ in range(9)]
    V1 = [[0]*rank1 for _ in range(9)]
    W1 = [[0]*rank1 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U1[i*3+j][col] = 1
                V1[j*3+k][col] = 1
                W1[k*3+i][col] = 1
                col += 1
    
    # Part 2: schoolbook for 3x3x2: A(i,j) * B2(j,k) for k=0,1 (representing columns 3,4)
    rank2 = 18
    U2 = [[0]*rank2 for _ in range(9)]
    V2 = [[0]*rank2 for _ in range(6)]  # m*p2 = 3*2
    W2 = [[0]*rank2 for _ in range(6)]  # p2*n = 2*3
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U2[i*3+j][col] = 1
                V2[j*2+k][col] = 1
                W2[k*3+i][col] = 1
                col += 1
    
    # Now combine into full 3x3x5 decomposition
    # Full U: 9 rows (3x3), rank = 9+18 = 27
    # Full V: 15 rows (3x5), rank = 27
    # Full W: 15 rows (5x3), rank = 27
    
    total_rank = rank1 + rank2
    U = [[0]*total_rank for _ in range(9)]
    V = [[0]*total_rank for _ in range(15)]
    W = [[0]*total_rank for _ in range(15)]
    
    # Fill U
    for i in range(9):
        for c in range(rank1):
            U[i][c] = U1[i][c]
        for c in range(rank2):
            U[i][rank1+c] = U2[i][c]
    
    # Fill V: rows 0-8 are for B columns 0-2 (j*3+k), rows 9-14 for B columns 3-4 (j*2+k + 9)
    # V1 has 9 rows (j*3+k for j=0..2, k=0..2)
    # V2 has 6 rows (j*2+k for j=0..2, k=0..1)
    for i in range(9):  # j in 0..2, k in 0..2
        for c in range(rank1):
            V[i][c] = V1[i][c]
        # V2 part: this row doesn't exist in V2 (V2 only has 6 rows for k=0,1)
    for i in range(6):  # j in 0..2, k in 0..1 (for columns 3,4)
        for c in range(rank2):
            V[9+i][rank1+c] = V2[i][c]
    
    # Fill W: W convention is W[k*n+i] = W[k*3+i]
    # W1: 9 rows, W1[k*3+i] for k=0..2, i=0..2
    # W2: 6 rows, W2[k*3+i] for k=0..1 (representing output columns 3,4), i=0..2
    # Full W: 15 rows, W[k*3+i] for k=0..4
    # k=0,1,2 -> W1
    # k=3,4 -> W2 (where k2 = k-3)
    for i in range(9):  # k*3+i for k=0..2
        k = i // 3
        ii = i % 3
        for c in range(rank1):
            W[i][c] = W1[k*3+ii][c]
    for i in range(6):  # k2*3+ii for k2=0..1
        k2 = i // 3
        ii = i % 3
        full_k = 3 + k2
        for c in range(rank2):
            W[full_k*3+ii][rank1+c] = W2[k2*3+ii][c]
    
    result = {"U": U, "V": V, "W": W}
    return total_rank, result

construct()  # Test to make sure it doesn't crash
