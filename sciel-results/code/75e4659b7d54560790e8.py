import time
import random
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Schoolbook decomposition (Rank 12) as fallback
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][col] = 1
                V_s[j * p + k][col] = 1
                W_s[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_s, V_s, W_s
    
    def is_valid(U, V, W, r, n, m, p, tol=1e-9):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Main term must equal 1
                    s_main = 0.0
                    for c in range(r):
                        s_main += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s_main - 1.0) > tol:
                        return False
                    # Cross terms must be 0
                    for i2 in range(n):
                        if i2 == i:
                            continue
                        s_cross = 0.0
                        for c in range(r):
                            s_cross += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i2][c]
                        if abs(s_cross) > tol:
                            return False
        return True
    
    def round_to_half(x):
        return round(x * 2) / 2
    
    def solve_for_U(V, W, n, m, p, r):
        num_eqs = n * m * p
        num_unk = n * m * r
        A = np.zeros((num_eqs, num_unk))
        b = np.ones(num_eqs)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        unk_idx = (i * m + j) * r + c
                        A[eq_idx, unk_idx] = V[j * p + k][c] * W[k * n + i][c]
                    eq_idx += 1
        U_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        U = [[0] * r for _ in range(n * m)]
        for i in range(n):
            for j in range(m):
                for c in range(r):
                    U[i * m + j][c] = U_flat[(i * m + j) * r + c]
        return U
    
    def solve_for_V(U, W, n, m, p, r):
        num_eqs = n * m * p
        num_unk = m * p * r
        A = np.zeros((num_eqs, num_unk))
        b = np.ones(num_eqs)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        unk_idx = (j * p + k) * r + c
                        A[eq_idx, unk_idx] = U[i * m + j][c] * W[k * n + i][c]
                    eq_idx += 1
        V_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        V = [[0] * r for _ in range(m * p)]
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    V[j * p + k][c] = V_flat[(j * p + k) * r + c]
        return V
    
    def solve_for_W(U, V, n, m, p, r):
        num_eqs = n * m * p
        num_unk = p * n * r
        A = np.zeros((num_eqs, num_unk))
        b = np.ones(num_eqs)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        unk_idx = (k * n + i) * r + c
                        A[eq_idx, unk_idx] = U[i * m + j][c] * V[j * p + k][c]
                    eq_idx += 1
        W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        W = [[0] * r for _ in range(p * n)]
        for k in range(p):
            for i in range(n):
                for c in range(r):
                    W[k * n + i][c] = W_flat[(k * n + i) * r + c]
        return W
    
    def try_rank(target_rank, max_attempts=100):
        """Try to find a valid decomposition at the given rank."""
        nonlocal best_rank, best_U, best_V, best_W
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return False
            
            # Strategy 1: Pattern-based initialization with ±1/0 entries
            if attempt < 40:
                # Generate random ±1/0 patterns
                U = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(p * n)]
                
                # Ensure each row has at least one non-zero
                for row in U:
                    if all(x == 0 for x in row):
                        row[random.randint(0, target_rank-1)] = 1
                for row in V:
                    if all(x == 0 for x in row):
                        row[random.randint(0, target_rank-1)] = 1
                for row in W:
                    if all(x == 0 for x in row):
                        row[random.randint(0, target_rank-1)] = 1
                
                # ALS refinement
                for it in range(30):
                    if time.time() > deadline:
                        return False
                    U = solve_for_U(V, W, n, m, p, target_rank)
                    V = solve_for_V(U, W, n, m, p, target_rank)
                    W = solve_for_W(U, V, n, m, p, target_rank)
                
                # Round to half-integers
                U_r = [[round_to_half(x) for x in row] for row in U]
                V_r = [[round_to_half(x) for x in row] for row in V]
                W_r = [[round_to_half(x) for x in row] for row in W]
                
                if is_valid(U_r, V_r, W_r, target_rank, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    return True
                
                # Also try rounding to integers
                U_i = [[round(x) for x in row] for row in U]
                V_i = [[round(x) for x in row] for row in V]
                W_i = [[round(x) for x in row] for row in W]
                if is_valid(U_i, V_i, W_i, target_rank, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_i, V_i, W_i
                    return True
                    
            elif attempt < 70:
                # Strategy 2: Random continuous initialization
                U = [[random.uniform(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.uniform(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.uniform(-2, 2) for _ in range(target_rank)] for _ in range(p * n)]
                
                for it in range(50):
                    if time.time() > deadline:
                        return False
                    U = solve_for_U(V, W, n, m, p, target_rank)
                    V = solve_for_V(U, W, n, m, p, target_rank)
                    W = solve_for_W(U, V, n, m, p, target_rank)
                
                U_r = [[round_to_half(x) for x in row] for row in U]
                V_r = [[round_to_half(x) for x in row] for row in V]
                W_r = [[round_to_half(x) for x in row] for row in W]
                
                if is_valid(U_r, V_r, W_r, target_rank, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    return True
                
                U_i = [[round(x) for x in row] for row in U]
                V_i = [[round(x) for x in row] for row in V]
                W_i = [[round(x) for x in row] for row in W]
                if is_valid(U_i, V_i, W_i, target_rank, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_i, V_i, W_i
                    return True
                    
            else:
                # Strategy 3: Sparse pattern with more structure
                # Try patterns where each column has limited support
                U = [[0] * target_rank for _ in range(n * m)]
                V = [[0] * target_rank for _ in range(m * p)]
                W = [[0] * target_rank for _ in range(p * n)]
                
                for c in range(target_rank):
                    # Choose which entries are active in this column
                    u_active = random.sample(range(n * m), random.randint(2, 4))
                    v_active = random.sample(range(m * p), random.randint(2, 4))
                    w_active = random.sample(range(p * n), random.randint(2, 4))
                    
                    for idx in u_active:
                        U[idx][c] = random.choice([1, -1])
                    for idx in v_active:
                        V[idx][c] = random.choice([1, -1])
                    for idx in w_active:
                        W[idx][c] = random.choice([1, -1])
                
                for it in range(40):
                    if time.time() > deadline:
                        return False
                    U = solve_for_U(V, W, n, m, p, target_rank)
                    V = solve_for_V(U, W, n, m, p, target_rank)
                    W = solve_for_W(U, V, n, m, p, target_rank)
                
                U_r = [[round_to_half(x) for x in row] for row in U]
                V_r = [[round_to_half(x) for x in row] for row in V]
                W_r = [[round_to_half(x) for x in row] for row in W]
                
                if is_valid(U_r, V_r, W_r, target_rank, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    return True
                
                U_i = [[round(x) for x in row] for row in U]
                V_i = [[round(x) for x in row] for row in V]
                W_i = [[round(x) for x in row] for row in W]
                if is_valid(U_i, V_i, W_i, target_rank, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_i, V_i, W_i
                    return True
        
        return False
    
    # Try ranks in increasing order
    for target_rank in [9, 10, 11]:
        if time.time() > deadline:
            break
        if try_rank(target_rank):
            break
    
    # Final validity check
    if not is_valid(best_U, best_V, best_W, best_rank, n, m, p):
        # Fall back to schoolbook
        best_rank = rank_school
        best_U, best_V, best_W = U_s, V_s, W_s
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
