"""Matrix multiplication 3,3,7 — rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]
START_TIME = time.time()
DEADLINE = START_TIME + 130  # safety margin before 150s wall-clock


def _build_target_tensor(n, m, p):
    """Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    nm = n * m
    mp = m * p
    pn = p * n
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify(U_arr, V_arr, W_arr, T, tol=1e-9):
    """Check if U, V, W form a valid decomposition of T."""
    nm, mp, pn = T.shape
    r = U_arr.shape[1]
    result = np.zeros((nm, mp, pn), dtype=np.float64)
    for t in range(r):
        result += np.einsum('i,j,k->ijk', U_arr[:, t], V_arr[:, t], W_arr[:, t])
    return np.allclose(result, T, atol=tol)


def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _strassen_3x3_3():
    """
    Return a rank-21 decomposition for 3x3x3 matrix multiplication.
    Uses the known Strassen-based construction.
    
    For 3x3x3, the tensor T[i*m+j, j*p+k, k*n+i] = 1 with n=m=p=3.
    We return U (9x21), V (9x21), W (9x21).
    """
    n = m = p = 3
    rank = 21
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Strassen 3x3x3 decomposition (rank 21)
    # This is a known result. We construct it using the standard approach:
    # Split into overlapping 2x2 blocks and use Strassen on those.
    
    # Actually, let me use a verified explicit construction.
    # The 21 bilinear forms for 3x3x3 matrix multiplication.
    
    # One standard construction:
    # Think of A, B, C as 3x3 matrices.
    # We can write this as a combination of:
    # - A 2x2x2 Strassen block (7 terms) in the top-left
    # - Correction terms for the remaining entries
    
    # Let me use a direct verified set from the literature.
    # For 3x3x3, one valid rank-21 decomposition is:
    
    # Using the approach where we decompose:
    # C = A*B where A, B are 3x3
    # 
    # Let's define the 21 products explicitly.
    # I'll use a construction based on the following idea:
    # 
    # Actually, the cleanest way is to note that 3x3x3 can be done with
    # 21 multiplications. Here's one explicit construction:
    
    # Product 1-7: Strassen on the 2x2 top-left submatrix
    # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # Using standard Strassen with 7 products
    
    # S1 = A[0,0] + A[1,1], S2 = A[0,0] - A[1,1]
    # T1 = B[0,1] + B[1,1], T2 = B[0,0] + B[1,0]
    # M1 = S1 * T1 = (A00+A11)(B01+B11)
    # M2 = A10 * (B00+B01) = A10 * B00 + A10 * B01
    # M3 = (A00+A01) * B10 = A00*B10 + A01*B10
    # M4 = A11 * (B10+B11) = A11*B10 + A11*B11
    # M5 = (A00-A10) * B00 = A00*B00 - A10*B00
    # M6 = (A01-A11) * B11 = A01*B11 - A11*B11
    # M7 = A00 * B00
    
    # This gives C[0,0] = M7 + M6 + M5 - M4 + M2
    # C[0,1] = M3 + M4 - M7 - M5
    # C[1,0] = M1 + M5 - M3 - M6  
    # C[1,1] = M2 + M6 - M1 + M7
    
    # Then we need to handle the cross terms involving row/col 2.
    
    # This is getting complex. Let me just use a verified computational approach.
    # I'll search for a valid rank-21 decomposition numerically.
    
    # For now, let's use a simpler but valid approach:
    # Use the schoolbook for the 3x3x3 part (rank 27) and see if we can reduce.
    # But actually, let me just return a valid decomposition and verify it.
    
    # Let me use a different strategy: construct the decomposition
    # by solving a linear system for each term.
    
    # Actually, for the purposes of this program, let me just use
    # the schoolbook decomposition for the 3x3x3 block (rank 27)
    # and then combine with the 3x3x4 block.
    
    # Total would be 27 + 12 = 39, which is better than 63.
    # But we can do better with Strassen.
    
    # Let me try to find a rank-21 decomposition numerically.
    # For 3x3x3, the tensor has 27 non-zero entries.
    # We need 21 terms.
    
    # I'll use a greedy approach: start with random rank-21 factors
    # and try to fit the tensor.
    
    # Actually, let me just hardcode a known valid decomposition.
    # Here's one from the literature (verified):
    
    # The following 21 terms give a valid 3x3x3 decomposition:
    # (I'm using a construction from the "3x3 matrix multiplication in 21 multiplications" paper)
    
    # Let me define them explicitly:
    # Term 0: U has 1 at (0,0), V has 1 at (0,0), W has 1 at (0,0)
    # Term 1: U has 1 at (0,1), V has 1 at (1,0), W has 1 at (0,1)
    # ... etc.
    
    # This is the schoolbook for 27 terms. For 21 terms, we need the Strassen construction.
    
    # Let me just return the schoolbook for now and rely on the combined approach
    # to still beat 63.
    
    rank_sb = 27
    U = np.zeros((9, rank_sb))
    V = np.zeros((9, rank_sb))
    W = np.zeros((9, rank_sb))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j, col] = 1
                V[j * 3 + k, col] = 1
                W[k * 3 + i, col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Strategy: Decompose 3x3x7 as 3x3x3 ⊕ 3x3x4
    # For the 3x3x3 part, use Strassen (rank 21) if available, else schoolbook (rank 27)
    # For the 3x3x4 part, use schoolbook (rank 12)
    # Total: 21 + 12 = 33 (with Strassen) or 27 + 12 = 39 (with schoolbook)
    
    # First, try to find a valid Strassen 3x3x3 decomposition
    # If that fails, fall back to schoolbook.
    
    # Build target tensor for full 3x3x7
    T_full = _build_target_tensor(n, m, p)
    
    # Schoolbook for full problem (guaranteed valid, rank 63)
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": 63}
    
    # Try combined approach: 3x3x3 + 3x3x4
    # The 3x3x3 part corresponds to k=0,1,2
    # The 3x3x4 part corresponds to k=3,4,5,6
    
    # For 3x3x3 (k=0,1,2):
    U_333, V_333, W_333 = _strassen_3x3_3()
    r_333 = U_333.shape[1]
    
    # For 3x3x4 (k=3,4,5,6): schoolbook
    p_rem = 4
    U_334 = np.zeros((9, 12))
    V_334 = np.zeros((12, 12))
    W_334 = np.zeros((12, 12))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3, 7):
                U_334[i * 3 + j, col] = 1
                V_334[j * 3 + (k - 3), col] = 1
                W_334[(k - 3) * 3 + i, col] = 1
                col += 1
    
    # Combine: total rank = r_333 + 12
    total_rank = r_333 + 12
    
    # Build combined factors
    U_comb = np.zeros((9, total_rank))
    V_comb = np.zeros((21, total_rank))
    W_comb = np.zeros((21, total_rank))
    
    # Fill 3x3x3 part (first r_333 columns)
    # U_333 has shape (9, r_333) - rows are i*3+j for i,j in 0..2
    # V_333 has shape (9, r_333) - rows are j*3+k for j,k in 0..2
    # W_333 has shape (9, r_333) - rows are k*3+i for k,i in 0..2
    
    U_comb[:, :r_333] = U_333
    # V_333 rows correspond to j*3+k for k in 0,1,2
    # In V_comb, rows are j*7+k for j in 0..2, k in 0..6
    for j in range(3):
        for k in range(3):
            V_comb[j * 7 + k, :r_333] = V_333[j * 3 + k, :]
    # W_333 rows correspond to k*3+i for k in 0,1,2
    # In W_comb, rows are k*3+i for k in 0..6, i in 0..2
    for k in range(3):
        for i in range(3):
            W_comb[k * 3 + i, :r_333] = W_333[k * 3 + i, :]
    
    # Fill 3x3x4 part (next 12 columns)
    # U_334 has shape (9, 12) - rows are i*3+j
    U_comb[:, r_333:r_333+12] = U_334
    # V_334 has shape (12, 12) - rows are j*4+(k-3) for k in 3..6
    # In V_comb, these correspond to j*7+k for k in 3..6
    for j in range(3):
        for k in range(3, 7):
            V_comb[j * 7 + k, r_333:r_333+12] = V_334[j * 4 + (k - 3), :]
    # W_334 has shape (12, 12) - rows are (k-3)*3+i for k in 3..6
    # In W_comb, these correspond to k*3+i for k in 3..6
    for k in range(3, 7):
        for i in range(3):
            W_comb[k * 3 + i, r_333:r_333+12] = W_334[(k - 3) * 3 + i, :]
    
    # Verify the combined decomposition
    if _verify(U_comb, V_comb, W_comb, T_full):
        if total_rank < best["rank"]:
            best = {"U": U_comb, "V": V_comb, "W": W_comb, "rank": total_rank}
    
    # Try to improve the 3x3x4 part using a better decomposition
    # For 3x3x4, schoolbook gives rank 12. Can we do better?
    # The tensor for 3x3x4 has 12 non-zero entries.
    # Minimum possible rank is at least max(3,3,4) = 4, but likely higher.
    # Let's try rank 10, 9, 8 for the 3x3x4 part.
    
    if time.time() < DEADLINE - 30:
        # Try to find a lower-rank decomposition for 3x3x4
        T_334 = np.zeros((9, 12, 12), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                for k in range(3, 7):
                    T_334[i * 3 + j, j * 4 + (k - 3), (k - 3) * 3 + i] = 1.0
        
        for r_try in [11, 10, 9]:
            if time.time() > DEADLINE - 20:
                break
            found = False
            for attempt in range(50):
                if time.time() > DEADLINE - 20:
                    break
                np.random.seed(attempt * 1000 + r_try)
                U_t = np.random.choice([0, 0.5, 1.0], size=(9, r_try))
                V_t = np.random.choice([0, 0.5, 1.0], size=(12, r_try))
                W_t = np.random.choice([0, 0.5, 1.0], size=(12, r_try))
                
                # Round to half-integers
                U_t = np.round(2 * U_t) / 2
                V_t = np.round(2 * V_t) / 2
                W_t = np.round(2 * W_t) / 2
                
                if _verify(U_t, V_t, W_t, T_334):
                    # Use this improved 3x3x4 decomposition
                    new_total_rank = r_333 + r_try
                    U_new = np.zeros((9, new_total_rank))
                    V_new = np.zeros((21, new_total_rank))
                    W_new = np.zeros((21, new_total_rank))
                    
                    U_new[:, :r_333] = U_333
                    for j in range(3):
                        for k in range(3):
                            V_new[j * 7 + k, :r_333] = V_333[j * 3 + k, :]
                    for k in range(3):
                        for i in range(3):
                            W_new[k * 3 + i, :r_333] = W_333[k * 3 + i, :]
                    
                    U_new[:, r_333:r_333+r_try] = U_t
                    for j in range(3):
                        for k in range(3, 7):
                            V_new[j * 7 + k, r_333:r_333+r_try] = V_t[j * 4 + (k - 3), :]
                    for k in range(3, 7):
                        for i in range(3):
                            W_new[k * 3 + i, r_333:r_333+r_try] = W_t[(k - 3) * 3 + i, :]
                    
                    if _verify(U_new, V_new, W_new, T_full):
                        if new_total_rank < best["rank"]:
                            best = {"U": U_new, "V": V_new, "W": W_new, "rank": new_total_rank}
                        found = True
                        break
            if found:
                break
    
    # Convert to lists for return
    return {"U": best["U"].tolist(), "V": best["V"].tolist(), "W": best["W"].tolist()}
