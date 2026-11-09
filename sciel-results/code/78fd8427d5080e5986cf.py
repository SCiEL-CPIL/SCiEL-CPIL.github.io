"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]
SEED_TIME_BUDGET = 120.0  # seconds

def _verify_exact(U, V, W, n, m, p):
    """Check if decomposition is exact using rational arithmetic."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0, 1)
                for c in range(rank):
                    u_val = U[i*m+j][c] if isinstance(U, list) else U[i*m+j, c]
                    v_val = V[j*p+k][c] if isinstance(V, list) else V[j*p+k, c]
                    w_val = W[k*n+i][c] if isinstance(W, list) else W[k*n+i, c]
                    # Convert to Fraction
                    if isinstance(u_val, Fraction):
                        uf = u_val
                    else:
                        uf = Fraction(u_val).limit_denominator(10000)
                    if isinstance(v_val, Fraction):
                        vf = v_val
                    else:
                        vf = Fraction(v_val).limit_denominator(10000)
                    if isinstance(w_val, Fraction):
                        wf = w_val
                    else:
                        wf = Fraction(w_val).limit_denominator(10000)
                    s += uf * vf * wf
                if s != Fraction(1, 1):
                    return False
    return True

def _verify_float(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is exact within tolerance."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _compute_error_fast(U, V, W, n, m, p):
    """Compute total error using numpy for speed."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    total_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m+j, :]
                v_val = V_arr[j*p+k, :]
                w_val = W_arr[k*n+i, :]
                s = np.dot(u_val, v_val * w_val)
                total_error += abs(s - 1.0)
    return total_error

def _structured_search(rank, n, m, p, seed=42, max_iters=5000, deadline=None):
    """
    Search for low-rank decomposition using structured random rank-1 terms.
    Uses numpy for fast error evaluation.
    """
    random.seed(seed)
    
    # Coefficient range for structured search
    coeff_range = [-2, -1, 1, 2]
    
    best_error = float('inf')
    best_solution = None
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate random sparse structure for each rank-1 term
        U_terms = []
        V_terms = []
        W_terms = []
        
        for c in range(rank):
            # Random sparse vectors with small integer coefficients
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            # Randomly select a few non-zero entries
            num_nonzero_u = random.randint(1, min(3, n * m))
            num_nonzero_v = random.randint(1, min(3, m * p))
            num_nonzero_w = random.randint(1, min(3, p * n))
            
            for _ in range(num_nonzero_u):
                idx = random.randint(0, n * m - 1)
                u_vec[idx] = random.choice(coeff_range)
            
            for _ in range(num_nonzero_v):
                idx = random.randint(0, m * p - 1)
                v_vec[idx] = random.choice(coeff_range)
            
            for _ in range(num_nonzero_w):
                idx = random.randint(0, p * n - 1)
                w_vec[idx] = random.choice(coeff_range)
            
            U_terms.append(u_vec)
            V_terms.append(v_vec)
            W_terms.append(w_vec)
        
        # Build factor matrices as numpy arrays
        U = np.array(U_terms).T  # shape (n*m, rank)
        V = np.array(V_terms).T  # shape (m*p, rank)
        W = np.array(W_terms).T  # shape (p*n, rank)
        
        # Check if this decomposition is exact
        error = _compute_error_fast(U, V, W, n, m, p)
        
        if error < 1e-10:
            # Convert back to lists for exact verification
            U_list = [[int(round(U[i, c])) for c in range(rank)] for i in range(n * m)]
            V_list = [[int(round(V[i, c])) for c in range(rank)] for i in range(m * p)]
            W_list = [[int(round(W[i, c])) for c in range(rank)] for i in range(p * n)]
            return U_list, V_list, W_list
        
        if error < best_error:
            best_error = error
            best_solution = (U.copy(), V.copy(), W.copy())
    
    return best_solution

def _polish_solution(U, V, W, n, m, p, seed=42, max_iters=2000, deadline=None):
    """
    Try to polish a near-solution by adjusting individual entries.
    Uses numpy for fast error computation.
    """
    random.seed(seed)
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    def compute_error():
        total_error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    s = np.dot(u_val, v_val * w_val)
                    total_error += abs(s - 1.0)
        return total_error
    
    best_error = compute_error()
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if best_error < 1e-10:
            break
        
        # Randomly select a matrix and entry to modify
        mat_choice = random.choice(['U', 'V', 'W'])
        
        if mat_choice == 'U':
            row = random.randint(0, n*m-1)
            col = random.randint(0, rank-1)
            old_val = U[row, col]
            U[row, col] = old_val + random.choice([-1, 1])
            new_error = compute_error()
            if new_error < best_error:
                best_error = new_error
            else:
                U[row, col] = old_val
        elif mat_choice == 'V':
            row = random.randint(0, m*p-1)
            col = random.randint(0, rank-1)
            old_val = V[row, col]
            V[row, col] = old_val + random.choice([-1, 1])
            new_error = compute_error()
            if new_error < best_error:
                best_error = new_error
            else:
                V[row, col] = old_val
        else:
            row = random.randint(0, p*n-1)
            col = random.randint(0, rank-1)
            old_val = W[row, col]
            W[row, col] = old_val + random.choice([-1, 1])
            new_error = compute_error()
            if new_error < best_error:
                best_error = new_error
            else:
                W[row, col] = old_val
    
    if best_error < 1e-10:
        U_list = [[int(round(U[i, c])) for c in range(rank)] for i in range(n * m)]
        V_list = [[int(round(V[i, c])) for c in range(rank)] for i in range(m * p)]
        W_list = [[int(round(W[i, c])) for c in range(rank)] for i in range(p * n)]
        return U_list, V_list, W_list
    return None

def _combine_decompositions(U1, V1, W1, U2, V2, W2, shift_k):
    """
    Combine two decompositions.
    Decomposition 1 is for (n, m, p1).
    Decomposition 2 is for (n, m, p2).
    shift_k is the offset for the k-index of the second decomposition.
    """
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    total_rank = rank1 + rank2
    
    n = len(U1) // 3 # Assuming n=3
    m = len(V1) // 5 # Assuming m=3, p=5? No, V is (m*p, rank).
    # We need dimensions to be passed or inferred.
    # U is (n*m, rank). V is (m*p, rank). W is (p*n, rank).
    
    # Infer n, m, p1, p2 from shapes
    # This is tricky without explicit dims.
    # But in our case, we know the dims.
    # U1: 9 x r1. V1: 15 x r1. W1: 15 x r1. (3x3x5? No, 3x3x2 -> V is 3x2=6? No, V is (m*p). m=3, p=2 -> 6.
    # Wait, the task is 3x3x5.
    # If we split 3x3x5 into 3x3x2 and 3x3x3.
    # Decomp 1 (3x3x2): U (9x18), V (6x18), W (6x18)? No.
    # V is (m*p). For 3x3x2, m=3, p=2 -> 6.
    # W is (p*n). For 3x3x2, p=2, n=3 -> 6.
    # Decomp 2 (3x3x3): U (9x23), V (9x23), W (9x23).
    
    # We need to pad V and W to match the full 3x3x5 dimensions.
    # Full V is (3*5=15, rank).
    # Full W is (5*3=15, rank).
    
    # V1 (6, r1) -> Pad to (15, r1). Rows 0-5 correspond to k=0,1.
    # V2 (9, r2) -> Pad to (15, r2). Rows 6-14 correspond to k=2,3,4.
    
    # W1 (6, r1) -> Pad to (15, r1). Rows 0-5 correspond to k=0,1 (index k*n+i).
    # W2 (9, r2) -> Pad to (15, r2). Rows 6-14 correspond to k=2,3,4.
    
    # U1 (9, r1) -> Pad to (9, total_rank).
    # U2 (9, r2) -> Pad to (9, total_rank).
    
    # This function is too specific. I will do the combination in construct().
    pass

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + SEED_TIME_BUDGET - 10.0
    
    # Fallback: Schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Strategy: Split 3x3x5 into 3x3x2 and 3x3x3.
    # Rank(3x3x2) = 18 (Schoolbook for two 3x3x1).
    # Rank(3x3x3) = 23 (Strassen-like).
    # Total Rank = 41.
    
    # 1. Construct Rank-18 decomposition for 3x3x2
    # Indices: i in 0..2, j in 0..2, k in 0..1
    U_12 = [[0]*18 for _ in range(9)]
    V_12 = [[0]*18 for _ in range(6)] # 3*2
    W_12 = [[0]*18 for _ in range(6)] # 2*3
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U_12[i*3+j][col] = 1
                V_12[j*2+k][col] = 1
                W_12[k*3+i][col] = 1
                col += 1
                
    # 2. Search for Rank-23 decomposition for 3x3x3
    # We will try to find a rank-23 decomposition for 3x3x3.
    U_33 = None
    V_33 = None
    W_33 = None
    
    # Try structured search for 3x3x3 rank 23
    # To speed up, we can try a few seeds.
    # If we fail, we fall back to schoolbook 27 for 3x3x3, total 18+27=45.
    
    found_33 = False
    for seed in range(50):
        if time.time() > deadline:
            break
        # Search for rank 23
        result = _structured_search(23, 3, 3, 3, seed=seed, max_iters=1000, deadline=deadline)
        if result is not None:
            U_33, V_33, W_33 = result
            # Verify
            if _verify_exact(U_33, V_33, W_33, 3, 3, 3):
                found_33 = True
                break
    
    if found_33:
        # Combine 3x3x2 (rank 18) and 3x3x3 (rank 23)
        total_rank = 18 + 23
        
        U_full = [[0]*total_rank for _ in range(9)]
        V_full = [[0]*total_rank for _ in range(15)]
        W_full = [[0]*total_rank for _ in range(15)]
        
        # Copy 3x3x2 part (cols 0..17)
        for i in range(9):
            for c in range(18):
                U_full[i][c] = U_12[i][c]
        
        for i in range(6):
            for c in range(18):
                V_full[i][c] = V_12[i][c]
        
        for i in range(6):
            for c in range(18):
                W_full[i][c] = W_12[i][c]
                
        # Copy 3x3x3 part (cols 18..40)
        # V_33 is 9x23 (indices j*3+k, j in 0..2, k in 0..2)
        # Full V is 15x41 (indices j*5+k, j in 0..2, k in 0..4)
        # The 3x3x3 part corresponds to k=2,3,4.
        # So V_33[j*3+k] maps to V_full[j*5 + (k+2)].
        
        for i in range(9):
            for c in range(23):
                U_full[i][18+c] = U_33[i][c]
        
        for j in range(3):
            for k in range(3):
                v_idx_33 = j*3 + k
                v_idx_full = j*5 + (k+2)
                for c in range(23):
                    V_full[v_idx_full][18+c] = V_33[v_idx_33][c]
                    
        # W_33 is 9x23 (indices k*3+i, k in 0..2, i in 0..2)
        # Full W is 15x41 (indices k*3+i, k in 0..4, i in 0..2)
        # The 3x3x3 part corresponds to k=2,3,4.
        # So W_33[k*3+i] maps to W_full[(k+2)*3 + i].
        
        for k in range(3):
            for i in range(3):
                w_idx_33 = k*3 + i
                w_idx_full = (k+2)*3 + i
                for c in range(23):
                    W_full[w_idx_full][18+c] = W_33[w_idx_33][c]
                    
        # Verify the combined decomposition
        if _verify_exact(U_full, V_full, W_full, 3, 3, 5):
            best_result = {"U": U_full, "V": V_full, "W": W_full}
            best_rank = total_rank
            return best_result
            
    # If 3x3x3 rank 23 not found, try rank 24, 25...
    # Or just return schoolbook.
    # Given time constraints, returning schoolbook is safe.
    
    return best_result
# EVOLVE-BLOCK-END
