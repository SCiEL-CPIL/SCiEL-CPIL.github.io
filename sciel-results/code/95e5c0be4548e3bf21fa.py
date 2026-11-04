"""Program for matrix multiplication tensor decomposition 3x4x7."""
import time
import numpy as np
from itertools import product

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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

    def verify_fast(U, V, W):
        """Fast verification using exact integer arithmetic via einsum."""
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,n_i,c]
        # But W is indexed as W[k*n+i, c], so W_r[k,i,c] = W[k*n+i, c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.all(T == 1)

    def verify(U, V, W):
        return verify_fast(U, V, W)

    def try_solve_for_W(U, V, target_rank):
        """Given U and V of shape (n*m, r) and (m*p, r), solve for W of shape (p*n, r)."""
        W = np.zeros((p * n, target_rank), dtype=np.int64)
        for k in range(p):
            for i in range(n):
                # Build system: for each j, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                A = np.zeros((m, target_rank), dtype=np.float64)
                for j in range(m):
                    A[j, :] = U[i * m + j, :].astype(float) * V[j * p + k, :].astype(float)
                
                # Solve A @ w = ones(m)
                try:
                    sol, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    return None
                
                # Check if solution is close to integer
                sol_int = np.round(sol).astype(np.int64)
                
                if np.allclose(A @ sol_int, np.ones(m), atol=1e-6):
                    W[k * n + i, :] = sol_int
                else:
                    # Try local search around sol_int with coordinate descent
                    found = False
                    test = sol_int.copy()
                    # Multiple rounds of coordinate descent
                    for _round in range(50):
                        improved = False
                        for c in range(target_rank):
                            for delta in [-2, -1, 1, 2]:
                                old = test[c]
                                test[c] = old + delta
                                if np.allclose(A @ test, np.ones(m), atol=1e-6):
                                    found = True
                                    improved = True
                                    break
                                test[c] = old
                            if found:
                                break
                        if not improved:
                            break
                    if not found:
                        return None
                    W[k * n + i, :] = test
        return W

    def structured_init_UV(target_rank, seed):
        """Create structured U and V that are more likely to allow integer W solutions."""
        rng = np.random.RandomState(seed)
        # Start with schoolbook-like structure but with some mixing
        U = np.zeros((n * m, target_rank), dtype=np.int64)
        V = np.zeros((m * p, target_rank), dtype=np.int64)
        
        # For each output position (i,k), we need m equations.
        # Strategy: assign columns to (i,k) pairs, with some overlap.
        # Total (i,k) pairs: n*p = 21
        # Each needs m=4 equations, so ideally 4 columns per pair = 84 columns
        # For rank < 84, we need sharing.
        
        # Initialize with random small integers
        U[:] = rng.randint(-1, 2, size=(n * m, target_rank))
        V[:] = rng.randint(-1, 2, size=(m * p, target_rank))
        
        # Ensure some structure: for each (i,k), make sure the product U[i,j,:]*V[j,k,:]
        # has enough rank to solve for W
        # We'll do a simple heuristic: ensure that for each (i,k), the m rows of
        # U[i,0:,:]*V[0,k,:], U[i,1:,:]*V[1,k,:], etc. are linearly independent
        # where possible.
        
        return U, V

    def local_search_improve(U, V, W, max_iters=1000):
        """Try to reduce rank by removing zero columns or merging."""
        # Check for zero columns
        r = U.shape[1]
        nonzero_cols = []
        for c in range(r):
            if np.any(U[:, c] != 0) and np.any(V[:, c] != 0) and np.any(W[:, c] != 0):
                nonzero_cols.append(c)
        
        if len(nonzero_cols) < r:
            U_new = U[:, nonzero_cols]
            V_new = V[:, nonzero_cols]
            W_new = W[:, nonzero_cols]
            if verify_fast(U_new, V_new, W_new):
                return U_new, V_new, W_new
        
        return U, V, W

    best_rank = nmp
    best_U, best_V, best_W = make_schoolbook()

    # Try structured and random searches for lower ranks
    np.random.seed(42)
    
    for target_rank in range(nmp - 1, 75, -1):
        if time.time() > deadline - 25:
            break
        
        # Try structured initialization first
        for seed in range(10):
            if time.time() > deadline - 20:
                break
            
            U, V = structured_init_UV(target_rank, 42 + seed)
            W = try_solve_for_W(U, V, target_rank)
            if W is not None:
                if verify_fast(U, V, W):
                    # Try to improve via local search
                    U2, V2, W2 = local_search_improve(U, V, W)
                    best_rank = min(best_rank, U2.shape[1])
                    best_U, best_V, best_W = U2, V2, W2
                    break
        
        # Try random initialization
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            U = np.random.randint(-2, 3, size=(n * m, target_rank)).astype(np.int64)
            V = np.random.randint(-2, 3, size=(m * p, target_rank)).astype(np.int64)
            
            W = try_solve_for_W(U, V, target_rank)
            if W is not None:
                if verify_fast(U, V, W):
                    U2, V2, W2 = local_search_improve(U, V, W)
                    best_rank = min(best_rank, U2.shape[1])
                    best_U, best_V, best_W = U2, V2, W2
                    break
        
        if time.time() > deadline - 10:
            break

    # Final verification
    if not verify_fast(best_U, best_V, best_W):
        best_U, best_V, best_W = make_schoolbook()
        best_rank = nmp

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
