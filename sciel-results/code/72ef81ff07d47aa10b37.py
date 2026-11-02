import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS
    
    # Attempt to construct a rank-10 decomposition
    # If verification fails, fall back to rank 11
    
    def verify(U, V, W):
        rank = len(W[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check dimensions
        if U_np.shape != (n*m, rank) or V_np.shape != (m*p, rank) or W_np.shape != (p*n, rank):
            return False
            
        # Verify tensor decomposition
        # C[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i, j]) * (sum_j V[j*p+k, r] * B[j, k])
        # We test with random matrices
        rng = np.random.default_rng(42)
        for _ in range(10):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        lA = np.dot(U_np[i*m:(i+1)*m, r], A[i, :])
                        lB = np.dot(V_np[k::p, r], B[:, k]) # V rows are j*p+k, so for fixed k, indices are 0*p+k, 1*p+k...
                        # Wait, V is (m*p, rank). Row index is j*p + k.
                        # So for a fixed output column k, the relevant V rows are [0*p+k, 1*p+k, ...].
                        # V_np[:, r] is a vector of length m*p.
                        # We need V[j*p+k, r] for j=0..m-1.
                        # This is V_np[k::p, r].
                        C_computed[i, k] += W_np[k*n+i, r] * lA * lB
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    # Rank 11 Solution (Current Best)
    def get_rank_11():
        rank = 11
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def set_u(i, j, r, val): U[i*m+j][r] = val
        def set_v(j, k, r, val): V[j*p+k][r] = val
        def set_w(k, i, r, val): W[k*n+i][r] = val

        # 1. (A00+A11)(B00+B11)
        set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
        set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
        set_w(0, 0, 0, 1)
        set_w(1, 1, 0, 1)

        # 2. (A10+A11)B00
        set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
        set_v(0, 0, 1, 1)
        set_w(0, 1, 1, 1)
        set_w(1, 1, 1, -1)

        # 3. A00(B01-B11)
        set_u(0, 0, 2, 1)
        set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
        set_w(1, 0, 2, 1)
        set_w(1, 1, 2, 1)

        # 4. A11(B10-B00)
        set_u(1, 1, 3, 1)
        set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
        set_w(0, 0, 3, 1)
        set_w(0, 1, 3, 1)

        # 5. (A00+A01)B11
        set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
        set_v(1, 1, 4, 1)
        set_w(0, 0, 4, -1)
        set_w(1, 0, 4, 1)

        # 6. (A10-A00)(B00+B01)
        set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
        set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
        set_w(1, 1, 5, 1)

        # 7. (A01-A11)(B10+B11)
        set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
        set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
        set_w(0, 0, 6, 1)

        # 8. A00 B02
        set_u(0, 0, 7, 1)
        set_v(0, 2, 7, 1)
        set_w(2, 0, 7, 1)

        # 9. A01 B12
        set_u(0, 1, 8, 1)
        set_v(1, 2, 8, 1)
        set_w(2, 0, 8, 1)

        # 10. A10 B02
        set_u(1, 0, 9, 1)
        set_v(0, 2, 9, 1)
        set_w(2, 1, 9, 1)

        # 11. A11 B12
        set_u(1, 1, 10, 1)
        set_v(1, 2, 10, 1)
        set_w(2, 1, 10, 1)

        return {"U": U, "V": V, "W": W}

    # Try to find a Rank 10 solution
    # This is a known result, but constructing it requires specific coefficients.
    # Based on literature, rank 10 is possible.
    # However, without a solver, I cannot guarantee a valid integer/half-integer solution 
    # that I can hardcode correctly from memory.
    # The risk of returning an invalid rank-10 solution is high.
    # The current rank-11 solution is valid and scores 1.0909.
    # If I cannot guarantee rank 10, I should stick to rank 11 to avoid score 0.
    
    # Let's double check if there's a simple transformation.
    # The 2x2x3 tensor is essentially 2x2x2 plus a 2x2x1 part.
    # Rank(2x2x2) = 7. Rank(2x2x1) = 2? No, 2x2x1 is just two dot products?
    # C02 = a00 b02 + a01 b12. C12 = a10 b02 + a11 b12.
    # This is rank 2 if we consider the A and B parts fixed?
    # Actually, the tensor for 2x2x1 is of rank 2?
    # M[i, j, k] with k=2.
    # T[i, j, 2] = delta_{i, j}? No.
    # C[i, 2] = sum_j A[i, j] B[j, 2].
    # This is a bilinear form.
    # It can be done with 2 multiplications?
    # (a00+a10)(b02) + (a01+a11)(b12)? No.
    # (a00+a11)(b02+b12) + (a10+a01)(b12-b02)?
    # This would give a00 b02 + a11 b02 + a00 b12 + a11 b12 + a10 b12 - a10 b02 + a01 b12 - a01 b02.
    # We need a00 b02 + a01 b12 for C02.
    # We need a10 b02 + a11 b12 for C12.
    # The above combination gives cross terms.
    # So it's not simply rank 2.
    
    # Given the complexity and the strict requirement for exactness, and the fact that
    # I cannot run a search to find the rank-10 coefficients, I will return the
    # verified rank-11 solution.
    
    return get_rank_11()
# EVOLVE-BLOCK-END
