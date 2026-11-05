"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def make_schoolbook():
        """Return the trivial (schoolbook) decomposition with rank = n*m*p = 54."""
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
    
    # Strategy: Use Strassen's rank-7 algorithm for 3x3 * 3x3, then extend to 3x3 * 3x6
    # by treating the 3x6 matrix as two 3x3 matrices (or using polynomial extension).
    # 
    # Key insight: A 3x6 matrix B can be viewed as [B1 | B2] where B1, B2 are 3x3.
    # C = A * [B1 | B2] = [A*B1 | A*B2]
    # 
    # If we can compute A*B1 and A*B2 using the same rank-7 decomposition structure,
    # we might share some of the intermediate products.
    #
    # Actually, a better approach: use the fact that 3x6 = 3x(3+3), and use
    # Strassen's algorithm which has 7 multiplications for 3x3*3x3.
    # For 3x3*3x6, we need to compute two 3x3 products, but we can potentially
    # share work.
    #
    # Let's try a direct approach: use known rank-7 Strassen for 3x3*3x3,
    # then replicate/adapt for the 6-column case.
    #
    # Strassen's 7-multiplication algorithm for 3x3 * 3x3:
    # Let A = [[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]]
    #     B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
    #
    # The 7 products are:
    # M1 = (a11+a22+a33) * (b11+b22+b33)
    # M2 = (a12-a22+a33) * (b12-b22+b33)  -- wait, this isn't standard
    #
    # Actually, let me use the well-known rank-7 decomposition for 3x3*3x3.
    # The standard Strassen-like algorithm for 3x3 uses 7 scalar multiplications.
    #
    # Here's a known rank-7 decomposition:
    # M1 = (a11+a22) * (b11+b22)
    # M2 = (a11+a22) * (b12+b21)  -- no, this isn't right either
    #
    # Let me use a different approach. I'll use the fact that we can pad 3x3 to 4x4
    # and use Strassen's 4x4 algorithm, but that gives rank 14 for 4x4*4x4.
    #
    # Better: directly search for a low-rank decomposition using optimization.
    # We know the minimum rank for 3x3*3x3 is 7 (Strassen), so for 3x3*3x6,
    # a reasonable target is around 14-20.
    
    def try_rank_optimization(rank, max_iters=50000):
        """Try to find a valid decomposition of given rank using optimization."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(42)
        
        # Initialize with random values
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize to avoid scale issues
        for r in range(rank):
            U[:, r] /= (np.linalg.norm(U[:, r]) + 1e-10)
            V[:, r] /= (np.linalg.norm(V[:, r]) + 1e-10)
            W[:, r] /= (np.linalg.norm(W[:, r]) + 1e-10)
        
        def compute_residual(U, V, W):
            """Compute the residual for all (i,j,k) triples."""
            residual = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        residual[i, j, k] = val
            return residual
        
        target = np.ones((n, m, p))
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Random perturbation with adaptive learning rate
            lr = 0.05 if it < max_iters // 2 else 0.01
            U_new = U + lr * np.random.randn(n * m, rank)
            V_new = V + lr * np.random.randn(m * p, rank)
            W_new = W + lr * np.random.randn(p * n, rank)
            
            res = compute_residual(U_new, V_new, W_new)
            err = np.sum((res - target) ** 2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
                U, V, W = U_new.copy(), V_new.copy(), W_new.copy()
                if err < 1e-8:
                    break
            else:
                # Occasionally reset to best
                if it % 1000 == 0 and err > best_err * 2:
                    U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
        
        if best_err < 1e-6:
            # Try to round to integers or half-integers
            for factor in [2, 4, 8]:
                U_round = np.round(best_U * factor) / factor
                V_round = np.round(best_V * factor) / factor
                W_round = np.round(best_W * factor) / factor
                
                res = compute_residual(U_round, V_round, W_round)
                if np.allclose(res, target, atol=1e-6):
                    return U_round.tolist(), V_round.tolist(), W_round.tolist()
            
            # Try integer rounding
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            res = compute_residual(U_int, V_int, W_int)
            if np.allclose(res, target, atol=1e-6):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        return None
    
    # Try ranks in increasing order, starting from a reasonable lower bound
    # Lower bound: max(n*p, m*p, n*m) = max(18, 18, 9) = 18
    # But we know 3x3*3x3 needs rank 7, so 3x3*3x6 might need around 14
    for rank in range(14, 40):
        if time.time() > deadline - 20:
            break
        result = try_rank_optimization(rank, max_iters=20000)
        if result is not None:
            U_list, V_list, W_list = result
            # Final verification
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return schoolbook if we couldn't find better
    return make_schoolbook()
# EVOLVE-BLOCK-END
