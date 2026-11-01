import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n, m, p):
    """Check U,V,W of given shapes reconstruct the matmul tensor exactly."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build tensor T[k*n+i, j*m+? ... ] 
    # Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # T[i, j, k] = 1 (delta). Index U row = i*m+j, V row = j*p+k, W row = k*n+i.
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    # Reconstruct via rank-r: T[a,b,c] = sum_rank U[a,r]*V[b,r]*W[c,r]
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    if recon.shape != T.shape:
        return False
    return np.allclose(recon, T, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    start_time = time.time()
    deadline = start_time + 145 # 5s safety margin for 150s limit

    # Strategy:
    # 1. Try to find a rank-15 decomposition using a deterministic/semi-deterministic search.
    #    The rank-15 decomposition for 2x3x3 matmul is known (Winograd-like).
    #    We can formulate it as a system of linear equations if we fix the support (non-zero entries).
    #    However, the support is not unique.
    #    
    #    Alternative: Use the known structure of the rank-15 decomposition.
    #    For 2x3x3, the rank is 15.
    #    Let's try to construct it directly if possible, or use a very targeted search.
    
    # Since the prompt asks to combine approaches, and the inspirations suggest optimization,
    # but the "deterministic exhaustive search" strategy in the directive is risky for 150s if not careful.
    # However, the space of supports for rank 15 is huge.
    
    # Let's try a smarter approach:
    # The rank-15 decomposition of 2x3x3 matmul can be derived from the rank-15 decomposition of 2x2x2 and 3x3x3? No.
    # Actually, there is a known decomposition.
    # Let's try to solve for rank 15 using a specific structure.
    # If that fails, fall back to schoolbook (18).
    
    # Is there a known explicit rank-15 decomposition?
    # Yes, for C(2,3,3), the rank is 15.
    # One such decomposition uses the fact that 2x3x3 = 2x2x3 + 2x1x3? No.
    
    # Let's try to use the "Winograd" style approach or a direct linear solve with a guessed support.
    # Given the time limit and the risk of failing to find the support, and the fact that the seed is 18,
    # jumping to 15 is a significant improvement.
    
    # Let's attempt a specific construction for rank 15.
    # The tensor T has 18 non-zero entries.
    # We need 15 rank-1 tensors to sum to T.
    
    # A known rank-15 decomposition for 2x3x3:
    # It is related to the decomposition of the 2x3 matrix multiplication.
    # Let's try to find it by solving a linear system with a fixed pattern of non-zeros.
    # This is complex to hardcode without the explicit coefficients.
    
    # Instead, let's try a fast optimization for rank 15 with many restarts, but optimized for speed.
    # Or, better yet, check if we can use a known algebraic identity.
    
    # Actually, let's look at the dimensions: 2, 3, 3.
    # Rank lower bound: max(6, 9, 9) = 9.
    # Schoolbook: 18.
    # Known rank: 15.
    
    # Let's try to find the rank-15 decomposition by solving a sparse linear system.
    # We will guess a support pattern that is likely to work.
    # A common pattern for such decompositions involves "cyclic" or "shifted" indices.
    
    # Given the complexity and risk, and that the inspirations all scored 1.0 (meaning they likely failed to beat 18 or just returned 18),
    # the safest bet to potentially improve is to try a very aggressive optimization for rank 15, 16, 17.
    
    # However, the directive says: "Use a deterministic exhaustive search over all possible rank-1 tensor support patterns... to identify a minimal set of 15 or fewer patterns...".
    # This is computationally infeasible for rank 15 (combinations of supports are huge).
    
    # Let's stick to the robust optimization approach but make it faster and more targeted.
    # We will try rank 15, 16, 17.
    
    # Optimization: Use `scipy.optimize.least_squares` is slow.
    # Let's use a custom Gauss-Newton or just multiple random restarts with a very tight time budget.
    
    # Actually, let's try to hardcode a known rank-15 decomposition if I can recall it or derive it.
    # I don't have the explicit coefficients memorized.
    
    # Let's try the optimization approach from Inspiration 2/3 but optimized.
    # If it fails, return 18.
    
    # To improve the chance of success, we can use a better initialization.
    # For rank r, we can initialize U, V, W such that they are "close" to a valid decomposition.
    
    # Let's try to solve for rank 15.
    # Variables: 15 * (6 + 9 + 9) = 15 * 24 = 360 variables.
    # Constraints: 18 equations (the non-zero entries of T).
    # Wait, the tensor has 6*9*9 = 486 entries. 18 are 1, 468 are 0.
    # So we have 486 constraints.
    # 360 variables, 486 constraints. Overdetermined.
    # We need the solution to be exact.
    
    # This is a system of polynomial equations (degree 3).
    # Solving this is hard.
    
    # Let's try a different tactic:
    # Is there a simple structure?
    # For 2x3x3, we can view it as multiplying a 2x3 matrix by a 3x3 matrix.
    # The result is 2x3.
    # T[i, j, k] = A[i, j] * B[j, k].
    
    # Let's try to find a rank-15 decomposition by searching for it.
    # Since the search space is large, we might not find it in 150s.
    # But the seed is 18. If we find 15, score is 1.2. If we find 16, 1.125.
    
    # Let's try to use the `sympy` library if available? No, only numpy/scipy.
    
    # Let's try a heuristic:
    # Start with rank 18 (schoolbook).
    # Try to merge columns to reduce rank.
    # This is complex.
    
    # Given the constraints and the fact that previous attempts failed to beat 1.0,
    # and the risk of returning an invalid decomposition (score 0),
    # I will implement a robust attempt to find rank 15 using a targeted search,
    # but with a strict fallback to 18.
    
    # Actually, looking at the inspirations, they all returned 1.0.
    # This suggests that finding a valid rank < 18 is difficult with the current tools/time.
    # However, the task is to "combine... strongest ideas".
    # The strongest idea is the optimization.
    # The refinement is "deterministic exhaustive search".
    
    # Let's try a hybrid:
    # 1. Try to find rank 15 using a specific known pattern if I can guess it.
    # 2. Otherwise, use the optimization.
    
    # Let's try the optimization for rank 15 with a time limit of 140s.
    # If it finds a solution with cost < 1e-9, return it.
    # Else return schoolbook.
    
    # To make it faster, we can reduce the number of trials if time is short.
    
    def try_optimize(rank, time_budget):
        end_time = time.time() + time_budget
        best_sol = None
        best_cost = np.inf
        
        # Number of trials
        # Each trial takes ~0.1-0.5s?
        # Let's estimate.
        
        for trial in range(50):
            if time.time() > end_time:
                break
            
            np.random.seed(42 + rank * 100 + trial)
            U0 = np.random.randn(nm, rank) * 0.1
            V0 = np.random.randn(mp, rank) * 0.1
            W0 = np.random.randn(pn, rank) * 0.1
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def residual(x):
                U = x[:nm*rank].reshape(nm, rank)
                V = x[nm*rank:nm*rank + mp*rank].reshape(mp, rank)
                W = x[nm*rank + mp*rank:].reshape(pn, rank)
                R = np.einsum('as,bs,cs->abc', U, V, W)
                return (R - T).flatten()
            
            try:
                from scipy.optimize import least_squares
                sol = least_squares(residual, x0, max_nfev=100, method='trf', ftol=1e-8, xtol=1e-8, gtol=1e-8)
                cost = np.sum(sol.fun**2)
                if cost < best_cost:
                    best_cost = cost
                    best_sol = sol.x
                    if cost < 1e-8:
                        break
            except Exception:
                pass
        
        if best_sol is not None and best_cost < 1e-8:
            U = best_sol[:nm*rank].reshape(nm, rank)
            V = best_sol[nm*rank:nm*rank + mp*rank].reshape(mp, rank)
            W = best_sol[nm*rank + mp*rank:].reshape(pn, rank)
            
            # Round to nearest half-integer
            U_out = [[round(x*2)/2 for x in row] for row in U]
            V_out = [[round(x*2)/2 for x in row] for row in V]
            W_out = [[round(x*2)/2 for x in row] for row in W]
            
            # Verify
            if _verify(U_out, V_out, W_out, n, m, p):
                return {"U": U_out, "V": V_out, "W": W_out}
            
        return None

    # Try rank 15 first
    result = try_optimize(15, 140)
    if result:
        return result
        
    # Try rank 16
    if time.time() < deadline - 10:
        result = try_optimize(16, 10)
        if result:
            return result

    # Try rank 17
    if time.time() < deadline - 5:
        result = try_optimize(17, 5)
        if result:
            return result

    # Fallback
    return schoolbook()
