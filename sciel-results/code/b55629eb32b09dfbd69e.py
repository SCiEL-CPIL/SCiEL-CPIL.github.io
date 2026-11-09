import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _validate(U, V, W):
    """Check if U, V, W form a valid exact decomposition."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    recon = np.einsum('ac,bc,dc->abd', U_arr, V_arr, W_arr)
    
    if not np.allclose(recon, T, atol=1e-9):
        return False
    
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                v = float(val)
                if abs(v * 2 - round(v * 2)) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _construct_winograd_like():
    """
    Construct a rank-21 decomposition based on the structure:
    The tensor for 2x3 * 3x7 has entries T[i, j, k] = 1.
    We can view the 'k' dimension (size 7) as independent from 'i' (size 2) and 'j' (size 3) in terms of the output index,
    but they are coupled in the sum.
    
    Actually, a standard block decomposition:
    If we can decompose 2x3 * 3x1 with rank R, we might be able to extend it.
    But 7 is prime, so we can't just split k.
    
    Let's try a specific construction that achieves rank 21.
    Rank 21 is m * p. This suggests we can use the 'j' and 'k' indices to define the rank, and 'i' is handled by linear combinations?
    Or rank = n * m * (something)?
    
    Let's look at the rank-21 hypothesis.
    U: (6, 21), V: (21, 21), W: (14, 21).
    
    Consider the identity:
    (A B)_{i, k} = sum_j A_{i, j} B_{j, k}
    
    If we fix k, we are computing a linear map from A (2x3) to C (2x1).
    This is equivalent to multiplying a 2x3 matrix by a 3x1 vector.
    The rank of 2x3 * 3x1 is 5 (since 2*3*1 - 2 - 3 + 1 = 5? No, rank is usually n*m for small ones, but 2x3*3x1 is rank 5? 
    Actually, 2x2*2x2 is rank 7. 2x3*3x2 is rank 13? No, 2x3*3x2 is rank 11? 
    Known ranks:
    2x2*2x2: 7
    2x3*3x2: 11
    2x3*3x1: 5? 
    Let's check 2x3 * 3x1.
    C_0 = A_00 B_0 + A_01 B_1 + A_02 B_2
    C_1 = A_10 B_0 + A_11 B_1 + A_12 B_2
    This is just 2 dot products of length 3.
    Rank of dot product of length 3 is 3? No, that's for 1x3 * 3x1.
    For 2x3 * 3x1, it's two independent dot products.
    Can we share intermediates?
    x0 = A_00 B_0
    x1 = A_01 B_1
    x2 = A_02 B_2
    y0 = A_10 B_0
    y1 = A_11 B_1
    y2 = A_12 B_2
    C_0 = x0+x1+x2
    C_1 = y0+y1+y2
    This is rank 6.
    Can we do better?
    (A_00+A_10) B_0, (A_01+A_11) B_1, (A_02+A_12) B_2, A_00 B_0, A_10 B_0...
    Let's use the known result: Rank of 2x3 * 3x1 is 5.
    Construction for 2x3 * 3x1 (rank 5):
    Let A be 2x3, B be 3x1.
    1. (A00+A01) B0
    2. A00 B0
    3. A01 B0
    4. A02 B2
    5. (A10+A11+A12) B1 ?? No.
    
    Actually, let's just try to find a rank 21 solution for 2x3*3x7 directly using the "separable" structure if possible.
    If the rank is 21, it equals m*p.
    This implies U has full row rank? No, U is 6x21.
    V is 21x21.
    W is 14x21.
    
    If we assume the decomposition separates over k?
    T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    If we set r = (j, k')? No, r is the rank index.
    
    Let's try a constructive search for rank 21 with a specific structure.
    Structure:
    U[i, j, (j', k')] ?
    
    Alternative: Use the "block" idea.
    2x3 * 3x7 = 2x3 * 3x(3+4).
    If we had a rank-11 decomposition for 2x3 * 3x3 and rank-? for 2x3 * 3x4, we could combine?
    No, the 'j' index is summed over, so we can't just split k.
    
    However, we can use the linearity in k.
    Let's try to find a rank 21 solution by random search with a strong bias towards structured solutions.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    # 1. Try to find a rank-21 solution via optimized random search
    target_rank = 21
    U = np.zeros((n * m, target_rank), dtype=float)
    V = np.zeros((m * p, target_rank), dtype=float)
    W = np.zeros((p * n, target_rank), dtype=float)
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    best_err = np.inf
    best_solution = None
    
    # Heuristic initialization: 
    # Often, good decompositions have many zeros.
    # Let's try a sparse random initialization.
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Random sparse initialization
        for c in range(target_rank):
            for a in range(n * m):
                if random.random() < 0.5:
                    U[a, c] = random.choice([0.5, 1, -0.5, -1])
                else:
                    U[a, c] = 0
            for b in range(m * p):
                if random.random() < 0.3:
                    V[b, c] = random.choice([0.5, 1, -0.5, -1])
                else:
                    V[b, c] = 0
            for d in range(p * n):
                if random.random() < 0.5:
                    W[d, c] = random.choice([0.5, 1, -0.5, -1])
                else:
                    W[d, c] = 0
        
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        err = np.sum((recon - T) ** 2)
        
        if err < best_err:
            best_err = err
            best_solution = (U.copy(), V.copy(), W.copy())
        
        # Local search with acceptance probability (Simulated Annealing style, but simple)
        temp = 1.0
        for step in range(2000):
            if time.time() > deadline:
                break
                
            # Pick random entry
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                idx = random.randint(0, n * m - 1)
                col = random.randint(0, target_rank - 1)
                old_val = U[idx, col]
                new_val = random.choice([0, 0.5, 1, -0.5, -1])
                if new_val == old_val: continue
                U[idx, col] = new_val
            elif mat_idx == 1:
                idx = random.randint(0, m * p - 1)
                col = random.randint(0, target_rank - 1)
                old_val = V[idx, col]
                new_val = random.choice([0, 0.5, 1, -0.5, -1])
                if new_val == old_val: continue
                V[idx, col] = new_val
            else:
                idx = random.randint(0, p * n - 1)
                col = random.randint(0, target_rank - 1)
                old_val = W[idx, col]
                new_val = random.choice([0, 0.5, 1, -0.5, -1])
                if new_val == old_val: continue
                W[idx, col] = new_val
                
            recon = np.einsum('ac,bc,dc->abd', U, V, W)
            new_err = np.sum((recon - T) ** 2)
            
            delta_err = new_err - err
            if delta_err < 0 or random.random() < np.exp(-delta_err / temp):
                err = new_err
                if err < best_err:
                    best_err = err
                    best_solution = (U.copy(), V.copy(), W.copy())
                    if err < 1e-9:
                        # Found a solution!
                        U_frac = [[Fraction(U[a, c]).limit_denominator(2) for c in range(target_rank)] for a in range(n * m)]
                        V_frac = [[Fraction(V[b, c]).limit_denominator(2) for c in range(target_rank)] for b in range(m * p)]
                        W_frac = [[Fraction(W[d, c]).limit_denominator(2) for c in range(target_rank)] for d in range(p * n)]
                        if _validate(U_frac, V_frac, W_frac):
                            return {"U": U_frac, "V": V_frac, "W": W_frac}
            else:
                # Revert
                if mat_idx == 0:
                    U[idx, col] = old_val
                elif mat_idx == 1:
                    V[idx, col] = old_val
                else:
                    W[idx, col] = old_val
            
            temp *= 0.995
    
    # 2. If rank 21 failed, try rank 22? Or just return schoolbook?
    # Given the time constraint, returning schoolbook is safer than a broken rank 22.
    # But let's try one more thing: a structured rank 21 attempt.
    
    # Structure:
    # U[i, j, r]
    # V[j, k, r]
    # W[k, i, r]
    
    # Try to construct a solution where r corresponds to (j, k) pairs? No, that's rank 21.
    # If r = j*p + k, then V[j, k, r] is delta(j, r//p) delta(k, r%p).
    # Then sum_r U[i, j, r] V[j, k, r] W[k, i, r] = U[i, j, j*p+k] * 1 * W[k, i, j*p+k]
    # We need this to be 1 for all i, j, k.
    # So U[i, j, j*p+k] * W[k, i, j*p+k] = 1.
    # This means U[i, j, j*p+k] and W[k, i, j*p+k] must be non-zero and multiply to 1.
    # We can set U[i, j, j*p+k] = 1 and W[k, i, j*p+k] = 1.
    # Let's check if this works.
    # U is (6, 21). U[i, j, col] where col = j*p+k.
    # For fixed i, j, there are 7 non-zero entries in U[i, j, :] at indices j*7+0 ... j*7+6.
    # V is (21, 21). V[j, k, col] where col = j*p+k.
    # V[j, k, j*p+k] = 1. All other entries 0.
    # W is (14, 21). W[k, i, col] where col = j*p+k.
    # W[k, i, j*p+k] = 1.
    
    # Let's verify this construction.
    U_test = np.zeros((6, 21))
    V_test = np.zeros((21, 21))
    W_test = np.zeros((14, 21))
    
    for i in range(2):
        for j in range(3):
            for k in range(7):
                col = j * 7 + k
                U_test[i * 3 + j, col] = 1
                V_test[j * 7 + k, col] = 1
                W_test[k * 2 + i, col] = 1
                
    recon_test = np.einsum('ac,bc,dc->abd', U_test, V_test, W_test)
    if np.allclose(recon_test, T, atol=1e-9):
        # This is a valid rank 21 solution!
        U_frac = [[int(U_test[a, c]) for c in range(21)] for a in range(6)]
        V_frac = [[int(V_test[b, c]) for c in range(21)] for b in range(21)]
        W_frac = [[int(W_test[d, c]) for c in range(21)] for d in range(14)]
        if _validate(U_frac, V_frac, W_frac):
            return {"U": U_frac, "V": V_frac, "W": W_frac}

    # If all else fails
    return _schoolbook()
