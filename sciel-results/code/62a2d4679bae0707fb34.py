import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    
    start_time = time.time()
    deadline = start_time + 110
    
    def make_schoolbook(rank):
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
    
    best_result = make_schoolbook(rank_schoolbook)
    best_rank = rank_schoolbook

    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact using numpy for speed."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_reshaped[i, j, :], np.multiply(V_reshaped[j, k, :], W_reshaped[k, i, :]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def check_half_integer(mat):
        """Check if all entries are integers or half-integers."""
        arr = np.array(mat, dtype=float)
        doubled = arr * 2
        rounded = np.round(doubled)
        if np.any(np.abs(doubled - rounded) > 1e-6):
            return False
        return True

    def try_block_structured(r, attempt_num):
        """Try block-structured sparse U and V, solve for W, check if W is half-integer."""
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        
        # Divide rank into m blocks
        block_size = r // m
        remainder = r % m
        
        # Assign blocks to j
        # j=0 gets block_size + (1 if remainder > 0 else 0)
        # j=1 gets block_size + (1 if remainder > 1 else 0)
        # j=2 gets block_size
        
        block_sizes = [block_size] * m
        for idx in range(remainder):
            block_sizes[idx] += 1
            
        start_idx = 0
        for j in range(m):
            end_idx = start_idx + block_sizes[j]
            # For each i in block j, pick random cols in [start_idx, end_idx)
            for i in range(n):
                row = i * m + j
                if block_sizes[j] > 0:
                    # Pick 1-2 non-zeros
                    nnz = min(2, block_sizes[j])
                    cols = random.sample(range(start_idx, end_idx), nnz)
                    for c in cols:
                        U[row, c] = random.choice([-1, 1])
                        
            # For each k in block j
            for k in range(p):
                row = j * p + k
                if block_sizes[j] > 0:
                    nnz = min(2, block_sizes[j])
                    cols = random.sample(range(start_idx, end_idx), nnz)
                    for c in cols:
                        V[row, c] = random.choice([-1, 1])
                        
            start_idx = end_idx
        
        # Build linear system for W
        num_eqs = n * m * p
        num_unk = p * n * r
        
        A = np.zeros((num_eqs, num_unk))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(r):
                        col = (k * n + i) * r + r_idx
                        A[eq_idx, col] = U[i * m + j, r_idx] * V[j * p + k, r_idx]
                    eq_idx += 1
        
        try:
            w_vec, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Check residual
            if len(residuals) > 0 and residuals[0] > 1e-10:
                return None
            
            w_vec = w_vec.reshape(p * n, r)
            
            # Try to round to half-integers
            w_rounded = np.round(w_vec * 2) / 2
            
            # Verify rounded version
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r_idx in range(r):
                            val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * w_rounded[k * n + i, r_idx]
                        if abs(val - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                U_list = [[float(U[i, r_idx]) for r_idx in range(r)] for i in range(n * m)]
                V_list = [[float(V[i, r_idx]) for r_idx in range(r)] for i in range(m * p)]
                W_list = [[float(w_rounded[i, r_idx]) for r_idx in range(r)] for i in range(p * n)]
                return {"U": U_list, "V": V_list, "W": W_list}
        except:
            pass
        
        return None

    # Try ranks from 44 down to 30
    for target_rank in range(rank_schoolbook - 1, 29, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Try multiple random attempts
        max_attempts = 500
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            result = try_block_structured(target_rank, attempt)
            
            if result is not None:
                best_result = result
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return best_result
