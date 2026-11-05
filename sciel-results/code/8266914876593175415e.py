import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s hard limit (accounting for overhead)

    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    num_eqs = n * m * p  # 84
    min_rank = max(dim_U, dim_V, dim_W)  # 28

    # Fallback: Schoolbook (Rank 84)
    def get_schoolbook():
        rank = num_eqs
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fast integer verification
    # U, V are integer arrays. W_int is integer array where W = W_int / 2.
    # Condition: sum_c U[i,j,c] * V[j,k,c] * W_int[k,i,c] == 2
    def verify_integer(U_int, V_int, W_int, r):
        # U_int: (n*m, r) int
        # V_int: (m*p, r) int
        # W_int: (p*n, r) int
        for i in range(n):
            for k in range(p):
                # Extract relevant rows
                # U rows: i*m to i*m+m-1
                U_sub = U_int[i * m:(i + 1) * m, :]  # (m, r)
                # V cols: k to k+m-1 ? No, V is (m, p).
                # V[j*p+k]. So for fixed k, we need columns k, k+m, ...? No.
                # V index is j*p+k.
                # So V_sub[j, :] = V[j*p+k, :]
                V_sub = np.array([V_int[j * p + k, :] for j in range(m)])  # (m, r)
                # W row: k*n+i
                W_row = W_int[k * n + i, :]  # (r,)
                
                # Compute product: (U_sub * V_sub) @ W_row
                # (U_sub * V_sub) is (m, r)
                # Dot product with W_row (r,) gives (m,)
                prod = (U_sub * V_sub) @ W_row
                if not np.all(prod == 2):
                    return False
        return True

    # Search for better rank
    # Strategy: Random search over small integer entries for U and V.
    # Solve for W_int using integer linear algebra or least squares + rounding.
    
    # We want to find r such that there exist U, V in Z^{dim_U x r} x Z^{dim_V x r}
    # and W_int in Z^{dim_W x r} such that the identity holds.
    
    # Since the search space is huge, we use a heuristic:
    # Fix U and V randomly. For each (i, k), solve M_{ik} w = 2*1_m.
    # M_{ik}[j, c] = U[i*m+j, c] * V[j*p+k, c]
    # If a solution exists, check if it is integer.
    
    # To increase the chance of integer solutions, we can bias U and V towards 0, 1, -1.
    
    best_result = None
    best_rank = 84

    # We iterate from min_rank up to 83.
    # However, finding rank < 84 is very hard.
    # Most "improvements" in these tasks come from structural knowledge or very lucky random hits.
    # Given the dimensions 3x4x7, the rank is likely 84 or slightly below if there are zeros.
    # Actually, standard matrix multiplication rank is np.
    # If we allow half-integers in W, it doesn't change the rank over reals, but restricts the solution space to a lattice.
    # If the real rank is R, the lattice rank might be higher or equal.
    # Schoolbook is 84.
    
    # Let's try to find a decomposition of rank 83, 82, etc.
    # We will limit the number of attempts per rank.
    
    rng = np.random.default_rng(42)
    
    # Try ranks in descending order from 83 down to min_rank?
    # No, better to try a few low ranks first in case there's a miracle, then higher ones.
    # Actually, low ranks are extremely unlikely.
    # Let's try ranks 80-83 with more iterations, and 70-79 with fewer.
    
    # To be efficient, we'll prioritize ranks close to 84 because deviations are small.
    
    rank_order = list(range(83, min_rank - 1, -1))
    # Actually, let's just loop 83 down to min_rank.
    # But if we don't find 83, we return 84.
    
    # Optimized search:
    # 1. Initialize U, V with random {-1, 0, 1}
    # 2. For each (i, k), solve M w = 2
    # 3. Check if w is integer
    
    for r in range(83, min_rank - 1, -1):
        if time.time() > deadline - 5:
            break
            
        # Number of attempts: more for ranks closer to 84
        if r > 75:
            num_tries = 500
        else:
            num_tries = 100
            
        for attempt in range(num_tries):
            if time.time() > deadline - 2:
                break
                
            # Random sparse/dense initialization
            # Use {-1, 0, 1}
            U_int = rng.choice([-1, 0, 1], size=(dim_U, r))
            V_int = rng.choice([-1, 0, 1], size=(dim_V, r))
            
            W_int = np.zeros((dim_W, r), dtype=int)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build M: m x r
                    M = np.zeros((m, r), dtype=int)
                    for j in range(m):
                        M[j, :] = U_int[i * m + j, :] * V_int[j * p + k, :]
                    
                    # Solve M w = 2
                    # Use least squares
                    try:
                        w_float, residuals, rank_sol, s = np.linalg.lstsq(M, np.full(m, 2.0), rcond=None)
                    except Exception:
                        valid = False
                        break
                    
                    if not valid:
                        break
                    
                    # Check if close to integer
                    w_int_candidate = np.round(w_float)
                    if np.max(np.abs(w_float - w_int_candidate)) > 1e-9:
                        valid = False
                        break
                    
                    # Verify exactly
                    if not np.allclose(M @ w_int_candidate, np.full(m, 2.0), atol=1e-9):
                        valid = False
                        break
                        
                    W_int[k * n + i, :] = w_int_candidate
            
            if not valid:
                continue
            
            # Final full verification
            if verify_integer(U_int, V_int, W_int, r):
                best_rank = r
                best_result = {
                    "U": U_int.tolist(),
                    "V": V_int.tolist(),
                    "W": (W_int / 2.0).tolist()
                }
                # If we find a low rank, we can stop early?
                # The task asks for rank-minimal. We should search down to min_rank.
                # But if we find 83, we should continue to see if 82 exists.
                # However, given time constraints, if we find a valid rank < 84,
                # it's already an improvement.
                # We will continue searching for lower ranks if time permits.
                # But for now, let's keep the best found.
                # Actually, the loop goes from 83 down. So if we find 83, we continue to 82.
                # If we find 82, we update best_result.
                pass

    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
