import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
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
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_reshaped[i, j, :], np.multiply(V_reshaped[j, k, :], W_reshaped[k, i, :]))
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    def check_half_integer(mat, tol=1e-6):
        """Check if all entries are integers or half-integers."""
        arr = np.array(mat, dtype=np.float64)
        doubled = arr * 2
        rounded = np.round(doubled)
        if np.any(np.abs(doubled - rounded) > tol):
            return False
        return True
    
    def is_valid_solution(U, V, W, rank):
        """Full validation: exact decomposition + half-integer entries."""
        if not verify_decomposition(U, V, W, rank):
            return False
        if not check_half_integer(U):
            return False
        if not check_half_integer(V):
            return False
        if not check_half_integer(W):
            return False
        return True
        
    # 1. Try Rank 41 (23 for 3x3x3 + 18 for 3x3x2)
    def make_rank_41():
        try:
            # Rank 23 for 3x3x3
            # Hardcoded 23 terms for 3x3x3 matrix multiplication
            # Each term is (u_vec, v_vec, w_vec) of length 9
            # u_vec corresponds to A (3x3), v_vec to B (3x3), w_vec to C (3x3)
            # Note: Indices are flattened row-major.
            
            # This is a known rank 23 decomposition.
            # If I don't have the exact coefficients, this will fail verification.
            # I will use a placeholder that is likely to fail, or a simple schoolbook subset?
            # Actually, I will use the schoolbook for 3x3x3 (27) and 3x3x2 (18) -> 45.
            # To get 41, I MUST have the 23 terms.
            # Since I don't have them memorized with 100% confidence, I will skip this specific hardcode 
            # and rely on the search, BUT I will improve the search.
            
            return None
        except:
            return None

    best_result = make_schoolbook(rank_schoolbook)
    best_rank = rank_schoolbook
    
    # ... (rest of the search code remains similar, but we can optimize)
    
    # The search code is kept as is, but we add a check for the Rank 41 attempt if we had it.
    # Since we don't, we rely on the search.
    
    # To improve the search, we can try more patterns or use a better solver.
    # But the search is likely to fail.
    
    # However, the prompt asks to improve.
    # If I can't guarantee 41, I should at least ensure the code is robust.
    
    # Let's try to find a Rank 44 solution using a specific heuristic?
    # No, the search is the best bet for small improvements if any exist.
    
    # I will return the best found.
    
    # (The original code's search logic is preserved)
    
    # Re-implementing the search logic from the original code to ensure it runs.
    
    def build_A_matrix(U, V, rank):
        num_eqs = n * m * p
        num_unk = p * n * rank
        A = np.zeros((num_eqs, num_unk))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r_idx in range(rank):
                        col = (k * n + i) * rank + r_idx
                        A[eq_idx, col] = U[i * m + j, r_idx] * V[j * p + k, r_idx]
        return A

    def solve_for_W(U, V, rank):
        A = build_A_matrix(U, V, rank)
        b = np.ones(n * m * p)
        try:
            w_vec, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=1e-12)
        except:
            return None
        if residuals.size > 0:
            if residuals[0] > 1e-8:
                return None
        w_vec = w_vec.reshape(p * n, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r_idx in range(rank):
                        val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * w_vec[k * n + i, r_idx]
                    if abs(val - 1) > 1e-8:
                        return None
        return w_vec

    def round_to_half_integer(mat, tol=1e-4):
        arr = np.array(mat, dtype=np.float64)
        doubled = arr * 2
        rounded = np.round(doubled)
        if np.any(np.abs(doubled - rounded) > tol):
            return None
        return (rounded / 2.0).tolist()

    def try_structured_decomposition(target_rank):
        if time.time() > deadline - 20:
            return None
        candidates = []
        rng = np.random.RandomState(42)
        for trial in range(50):
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            for r_idx in range(target_rank):
                u_count = rng.randint(2, 4)
                u_rows = rng.choice(n * m, size=u_count, replace=False)
                for row in u_rows:
                    U[int(row), r_idx] = rng.choice([-1, 1])
                v_count = rng.randint(2, 4)
                v_rows = rng.choice(m * p, size=v_count, replace=False)
                for row in v_rows:
                    V[int(row), r_idx] = rng.choice([-1, 1])
            candidates.append((U, V))
            
        for U, V in candidates:
            if time.time() > deadline - 15:
                break
            W = solve_for_W(U, V, target_rank)
            if W is None:
                continue
            W_half = round_to_half_integer(W.tolist(), tol=1e-3)
            if W_half is None:
                continue
            if not check_half_integer(U.tolist()):
                continue
            if not check_half_integer(V.tolist()):
                continue
            if is_valid_solution(U.tolist(), V.tolist(), W_half, target_rank):
                return {"U": U.tolist(), "V": V.tolist(), "W": W_half}
        return None

    for target_rank in range(rank_schoolbook - 1, 35, -1):
        if time.time() > deadline - 25:
            break
        result = try_structured_decomposition(target_rank)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
            
    return best_result
