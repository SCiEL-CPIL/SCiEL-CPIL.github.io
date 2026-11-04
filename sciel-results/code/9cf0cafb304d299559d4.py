"""Program for matrix multiplication tensor decomposition 3x4x7."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135.0  # Conservative: evaluator kills at 150s

    n, m, p = 3, 4, 7
    nmp = n * m * p  # 84 (schoolbook rank)

    def make_schoolbook():
        rank = nmp
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify_fast2(U, V, W):
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.all(T == 1)

    best_rank = nmp
    best_U, best_V, best_W = make_schoolbook()

    random.seed(12345)
    np.random.seed(12345)

    assert verify_fast2(best_U, best_V, best_W), "Schoolbook should verify"

    def find_W_given_UV(U, V, target_rank):
        """Solve for W given U and V. Uses least squares with integer rounding and local search."""
        W = np.zeros((p * n, target_rank), dtype=np.int64)
        for k in range(p):
            for i in range(n):
                # Build system: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all j
                A = np.zeros((m, target_rank), dtype=np.float64)
                for j in range(m):
                    A[j, :] = U[i * m + j, :].astype(float) * V[j * p + k, :].astype(float)
                
                # Try to find integer solution to A @ w = ones(m)
                # First check if exact solution exists via lstsq
                try:
                    sol, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    return None
                
                if len(residuals) > 0 and residuals[0] > 1e-6:
                    # Check if the system is even consistent
                    # Try a different approach: check rank
                    if rank_a < m:
                        # Underdetermined or inconsistent
                        pass
                    else:
                        return None
                
                sol_int = np.round(sol).astype(np.int64)
                
                # Check if rounded solution works
                if np.allclose(A @ sol_int, np.ones(m), atol=1e-8):
                    W[k * n + i, :] = sol_int
                    continue
                
                # Local search: try perturbations
                found = False
                # Adaptive search radius based on target_rank
                search_radius = 3 if target_rank <= 40 else 2
                # Use iterative deepening for larger ranks
                for radius in [1, 2, search_radius]:
                    if time.time() > deadline - 5:
                        return None
                    # For small ranks, do full grid search
                    if target_rank <= 4:
                        for combo in product(range(-radius, radius+1), repeat=target_rank):
                            test = sol_int + np.array(combo, dtype=np.int64)
                            if np.allclose(A @ test, np.ones(m), atol=1e-8):
                                W[k * n + i, :] = test
                                found = True
                                break
                        if found:
                            break
                    else:
                        # For larger ranks, do coordinate descent style search
                        test = sol_int.copy()
                        improved = True
                        max_iter = 50
                        for _ in range(max_iter):
                            if not improved:
                                break
                            improved = False
                            for c in range(target_rank):
                                for delta in range(-radius, radius+1):
                                    if delta == 0:
                                        continue
                                    test[c] += delta
                                    if np.allclose(A @ test, np.ones(m), atol=1e-8):
                                        found = True
                                        improved = True
                                        break
                                    test[c] -= delta
                                if found:
                                    break
                            if found:
                                break
                    if found:
                        break
                
                if not found:
                    # Last resort: try solving as integer program with small coefficients
                    # Check if there's a solution with entries in {-1, 0, 1}
                    if target_rank <= 8:
                        from itertools import product as iproduct
                        vals = [-1, 0, 1]
                        for combo in iproduct(vals, repeat=target_rank):
                            test = np.array(combo, dtype=np.int64)
                            if np.allclose(A @ test, np.ones(m), atol=1e-8):
                                W[k * n + i, :] = test
                                found = True
                                break
                        if found:
                            continue
                
                if not found:
                    return None
        return W

    def structured_init_UV(target_rank, seed_val):
        """Create structured initializations for U and V."""
        rng = np.random.RandomState(seed_val)
        U = np.zeros((n * m, target_rank), dtype=np.int64)
        V = np.zeros((m * p, target_rank), dtype=np.int64)
        
        # Strategy 1: Random with wider range
        U[:, :] = rng.randint(-2, 3, size=(n * m, target_rank))
        V[:, :] = rng.randint(-2, 3, size=(m * p, target_rank))
        
        # Ensure some structure: make sure each row has at least one nonzero
        for row in range(n * m):
            if np.all(U[row, :] == 0):
                U[row, 0] = 1
        for row in range(m * p):
            if np.all(V[row, :] == 0):
                V[row, 0] = 1
        
        return U, V

    # Phase 1: Try structured initializations for ranks near lower bound
    # Theoretical lower bound: ceil(n*m*p / (n+m+p)) but practical bounds are higher
    # For 3x4x7, known best is around 50-60 range
    lower_bound = 35  # Practical starting point
    
    for target_rank in range(lower_bound, nmp):
        if time.time() > deadline - 30:
            break
        
        # Number of attempts based on remaining time
        time_per_attempt = 0.5  # Estimate
        remaining = deadline - 30 - time.time()
        max_attempts = max(10, int(remaining / time_per_attempt))
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 25:
                break
            
            U, V = structured_init_UV(target_rank, 12345 + attempt * 1000)
            W = find_W_given_UV(U, V, target_rank)
            if W is not None:
                if verify_fast2(U, V, W):
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    break
        
        if best_rank < nmp:
            # Found improvement, try to go lower
            lower_bound = max(30, best_rank - 5)
            continue
    
    # Phase 2: If we found something, try to improve further with more diverse searches
    if best_rank < nmp and time.time() < deadline - 20:
        for target_rank in range(max(30, best_rank - 10), best_rank):
            if time.time() > deadline - 15:
                break
            for attempt in range(50):
                if time.time() > deadline - 12:
                    break
                rng = np.random.RandomState(99999 + attempt * 777)
                U = rng.randint(-3, 4, size=(n * m, target_rank))
                V = rng.randint(-3, 4, size=(m * p, target_rank))
                W = find_W_given_UV(U, V, target_rank)
                if W is not None:
                    if verify_fast2(U, V, W):
                        best_rank = target_rank
                        best_U, best_V, best_W = U, V, W
                        break
            if best_rank < nmp:
                break
    
    # Phase 3: Final polish - try to verify and clean up the best solution
    if best_rank < nmp:
        # Verify one more time
        if not verify_fast2(best_U, best_V, best_W):
            # Fallback to schoolbook if something went wrong
            best_U, best_V, best_W = make_schoolbook()
            best_rank = nmp
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
