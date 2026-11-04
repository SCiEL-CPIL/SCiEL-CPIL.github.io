import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    max_time = 145  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    target_rank = n * m * p  # 24
    
    # Build the target tensor T[i, j, k]
    # T[i, j, k] = 1 for all valid i, j, k
    T = np.ones((n, m, p))
    
    def compute_tensor(U, V, W, R):
        """Compute the approximated tensor from factors."""
        U3 = np.array(U).reshape(n, m, R)
        V3 = np.array(V).reshape(m, p, R)
        W3 = np.array(W).reshape(p, n, R)
        # T_approx[i, j, k] = sum_c U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
        return np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
    def verify_exact(U, V, W, R):
        """Check if the decomposition is exact."""
        T_approx = compute_tensor(U, V, W, R)
        return np.allclose(T_approx, T, atol=1e-9)

    def to_half_int_list(arr):
        """Convert a numpy array to a list of lists of floats, rounded to nearest 0.5."""
        rounded = np.round(arr * 2) / 2.0
        return rounded.tolist()

    def check_and_round(U, V, W, R):
        """Check if the decomposition is exact and round to half-integers."""
        T_approx = compute_tensor(U, V, W, R)
        if np.allclose(T_approx, T, atol=1e-6):
            # Round to nearest 0.5
            U_rounded = np.round(U * 2) / 2.0
            V_rounded = np.round(V * 2) / 2.0
            W_rounded = np.round(W * 2) / 2.0
            
            T_check = compute_tensor(U_rounded, V_rounded, W_rounded, R)
            if np.allclose(T_check, T, atol=1e-9):
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist(),
                    "rank": R
                }
        return None

    def try_rank_als(R, num_restarts=50):
        """
        Try to find a decomposition of rank R using Alternating Least Squares.
        """
        
        def solve_U(U, V, W):
            U_new = np.zeros((n, m, R))
            U3 = U.reshape(n, m, R)
            V3 = V.reshape(m, p, R)
            W3 = W.reshape(p, n, R)
            for i in range(n):
                for j in range(m):
                    # A is (p, R), A[:, c] = V3[j, :, c] * W3[:, i, c]
                    A = V3[j] * W3[:, i]
                    b = T[i, j, :]
                    # Solve A x = b
                    # A is p x R. If R > p, we have underdetermined system.
                    # np.linalg.lstsq gives min norm solution.
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i, j, :] = x
            return U_new.reshape(n * m, R)
        
        def solve_V(U, V, W):
            V_new = np.zeros((m, p, R))
            U3 = U.reshape(n, m, R)
            V3 = V.reshape(m, p, R)
            W3 = W.reshape(p, n, R)
            for j in range(m):
                for k in range(p):
                    # B is (n, R), B[:, c] = U3[:, j, c] * W3[k, :, c]
                    B = U3[:, j] * W3[k]
                    b = T[:, j, k]
                    x, _, _, _ = np.linalg.lstsq(B, b, rcond=None)
                    V_new[j, k, :] = x
            return V_new.reshape(m * p, R)
        
        def solve_W(U, V, W):
            W_new = np.zeros((p, n, R))
            U3 = U.reshape(n, m, R)
            V3 = V.reshape(m, p, R)
            W3 = W.reshape(p, n, R)
            for k in range(p):
                for i in range(n):
                    # C is (m, R), C[:, c] = U3[i, :, c] * V3[:, k, c]
                    C = U3[i] * V3[:, k]
                    b = T[i, :, k]
                    x, _, _, _ = np.linalg.lstsq(C, b, rcond=None)
                    W_new[k, i, :] = x
            return W_new.reshape(p * n, R)
        
        # Try multiple restarts with different initializations
        for r in range(num_restarts):
            if time.time() - start_time > max_time - 5:
                return None
            
            # Different initialization strategies
            if r % 3 == 0:
                # Random {-1, 0, 1}
                np.random.seed(r)
                U = np.random.choice([-1, 0, 1], size=(n * m, R))
                V = np.random.choice([-1, 0, 1], size=(m * p, R))
                W = np.random.choice([-1, 0, 1], size=(p * n, R))
            elif r % 3 == 1:
                # Random {-1, 1}
                np.random.seed(r + 1000)
                U = np.random.choice([-1, 1], size=(n * m, R))
                V = np.random.choice([-1, 1], size=(m * p, R))
                W = np.random.choice([-1, 1], size=(p * n, R))
            else:
                # Random normal
                np.random.seed(r + 2000)
                U = np.random.normal(size=(n * m, R))
                V = np.random.normal(size=(m * p, R))
                W = np.random.normal(size=(p * n, R))
            
            # Run ALS
            for it in range(30):
                U = solve_U(U, V, W)
                V = solve_V(U, V, W)
                W = solve_W(U, V, W)
                
                # Check convergence / exactness
                result = check_and_round(U, V, W, R)
                if result:
                    return result
            
        return None

    def try_rank_sparse_search(R, time_budget=10):
        """
        Try to find a decomposition of rank R using sparse support patterns.
        This is a heuristic search.
        """
        start_search = time.time()
        
        # We know the rank is likely around 18-20.
        # For rank R, we can try to fix the support of U, V, W to be sparse.
        # Specifically, we can try to find a decomposition where each entry is in {-1, 0, 1} or similar.
        
        # A more robust approach for small R is to use a local search on the values.
        # But let's stick to the ALS family as per the directive, maybe with a "polish" step.
        
        # The directive says: "Treat these as hypotheses to test. Do NOT switch to a different design family."
        # So I will stick to ALS but improve the "polish" or "exact solve" step.
        
        # One improvement: after ALS converges, try to solve the linear system exactly using fractions
        # if the values are close to rationals.
        
        # Let's try a few more restarts with specific initializations that might help.
        # For example, starting from the schoolbook solution and removing columns.
        
        # Schoolbook has rank 24.
        # If we want rank R < 24, we can try to merge some columns.
        
        # This is getting complicated. Let's just run ALS with more restarts and tighter checking.
        
        for r in range(100):
            if time.time() - start_search > time_budget or time.time() - start_time > max_time - 5:
                break
            
            np.random.seed(42 + r)
            U = np.random.normal(size=(n * m, R))
            V = np.random.normal(size=(m * p, R))
            W = np.random.normal(size=(p * n, R))
            
            # Scale to avoid overflow/underflow
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for it in range(50):
                U = solve_U(U, V, W)
                V = solve_V(U, V, W)
                W = solve_W(U, V, W)
                
                result = check_and_round(U, V, W, R)
                if result:
                    return result
                    
        return None

    # Try lower ranks
    # The known rank for 2x3x4 is 20? Or 19?
    # Strassen-like algorithms suggest lower ranks.
    # Let's try from 19 upwards? No, we want minimal rank, so try low to high?
    # Or high to low to find a valid one quickly?
    # The task says "Find a rank-minimal exact tensor decomposition".
    # So we should try the lowest possible rank first.
    # But if we don't know the lower bound, we might waste time.
    # Let's try 19, 20, 21, 22, 23.
    
    # Actually, the current program tries 18 to 23.
    # Let's keep that range but improve the solver.
    
    for R in range(18, 24):
        if time.time() - start_time > max_time - 10:
            break
        # Use more restarts for lower ranks
        num_restarts = 100 if R <= 19 else 50
        res = try_rank_als(R, num_restarts)
        if res:
            return res
        
        # Try the sparse search variant (which is just more ALS restarts)
        res = try_rank_sparse_search(R, time_budget=5)
        if res:
            return res

    # Fallback: Schoolbook decomposition
    rank = target_rank
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
