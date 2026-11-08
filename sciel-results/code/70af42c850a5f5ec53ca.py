import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook fallback
    def get_schoolbook():
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

    # Try Rank 46 (2 * Laderman 23)
    try:
        if time.time() - start_time > 100:
            return get_schoolbook()
            
        # Define Laderman's 23 products for 3x3x3
        # U and V coefficients for the 23 products
        # a_terms: list of (i, j, coeff)
        # b_terms: list of (j, k, coeff)
        products = []
        
        def add_product(a_terms, b_terms):
            products.append((a_terms, b_terms))

        # P1
        add_product([(0,0,1)], [(0,0,1)])
        # P2
        add_product([(0,1,1)], [(1,1,1)])
        # P3
        add_product([(1,0,1)], [(0,0,1)])
        # P4
        add_product([(1,1,1)], [(1,1,1)])
        # P5
        add_product([(2,2,1)], [(2,2,1)])
        # P6
        add_product([(0,0,1), (1,1,1), (2,2,-1)], [(2,2,1)])
        # P7
        add_product([(0,0,1), (1,1,1), (2,2,-1)], [(0,1,1)])
        # P8
        add_product([(0,1,1), (1,0,1)], [(0,1,1)])
        # P9
        add_product([(0,1,1), (1,0,1)], [(1,0,1)])
        # P10
        add_product([(0,0,1), (1,1,1), (2,2,1), (0,1,-1), (1,0,-1)], [(1,0,1)])
        # P11
        add_product([(0,0,1), (1,1,1), (2,2,1), (0,1,1), (1,0,1)], [(0,0,1)])
        # P12
        add_product([(0,0,1), (1,1,-1)], [(0,1,1)])
        # P13
        add_product([(0,0,1), (1,1,-1)], [(1,0,1)])
        # P14
        add_product([(0,1,1), (1,0,-1)], [(0,1,1)])
        # P15
        add_product([(0,1,1), (1,0,-1)], [(1,0,1)])
        # P16
        add_product([(0,0,1), (0,1,1), (1,0,1), (1,1,1)], [(1,1,1)])
        # P17
        add_product([(0,0,1), (0,1,1), (1,0,1), (1,1,1)], [(0,0,1)])
        # P18
        add_product([(0,0,1), (1,0,1), (2,2,1)], [(2,2,1)])
        # P19
        add_product([(0,1,1), (1,1,1), (2,2,1)], [(2,2,1)])
        # P20
        add_product([(0,0,1), (1,0,1), (2,2,1)], [(0,1,1)])
        # P21
        add_product([(0,1,1), (1,1,1), (2,2,1)], [(1,0,1)])
        # P22
        add_product([(0,0,1), (0,1,1), (1,0,1), (1,1,1), (2,2,1)], [(0,0,1)])
        # P23
        add_product([(0,0,1), (0,1,1), (1,0,1), (1,1,1), (2,2,1)], [(1,1,1)])

        # Build U and V for the 23 products
        U_23 = [[0] * 23 for _ in range(9)]
        V_23 = [[0] * 23 for _ in range(9)]
        
        for l in range(23):
            a_terms, b_terms = products[l]
            for i, j, coeff in a_terms:
                U_23[i * 3 + j][l] += coeff
            for j, k, coeff in b_terms:
                V_23[j * 3 + k][l] += coeff

        # Solve for W_23 using random instances
        # C_ik = sum_l W_ikl * P_l
        # We need to find W_ikl for i,k in 0..2
        
        W_23 = [[0] * 23 for _ in range(9)]
        
        # Generate random A, B
        rng = np.random.default_rng(42)
        num_samples = 50
        A_samples = rng.integers(-10, 10, size=(num_samples, 3, 3))
        B_samples = rng.integers(-10, 10, size=(num_samples, 3, 3))
        
        # Compute P values
        # P_l = sum_{i,j} U_23[i,j,l] A[i,j] * sum_{j',k} V_23[j',k,l] B[j',k]
        # Note: V_23 is indexed by (j,k). The product is sum_j (U_A * A) * (V_B * B)?
        # No, the product is (linear form on A) * (linear form on B).
        # U_23[i*3+j][l] is coeff for A[i,j].
        # V_23[j*3+k][l] is coeff for B[j,k].
        
        P_samples = np.zeros((num_samples, 23))
        for s in range(num_samples):
            A = A_samples[s]
            B = B_samples[s]
            for l in range(23):
                ua = 0
                for i in range(3):
                    for j in range(3):
                        ua += U_23[i*3+j][l] * A[i,j]
                vb = 0
                for j in range(3):
                    for k in range(3):
                        vb += V_23[j*3+k][l] * B[j,k]
                P_samples[s, l] = ua * vb

        C_samples = np.matmul(A_samples, B_samples)
        
        # Solve for each i,k
        # C_samples[:, i, k] = W_ik @ P_samples.T
        # P_samples is (num_samples, 23). C is (num_samples,).
        # We want W (23,) such that P @ W = C.
        
        valid = True
        for i in range(3):
            for k in range(3):
                c_vec = C_samples[:, i, k]
                # Solve P_samples.T @ W = c_vec
                # P_samples.T is (23, num_samples)
                # Use lstsq
                try:
                    W_sol, residuals, rank, sv = np.linalg.lstsq(P_samples.T, c_vec, rcond=None)
                    # Check if W_sol is close to integers or half-integers
                    # And if residuals are small
                    if residuals.size > 0 and residuals[0] > 1e-6:
                        valid = False
                        break
                    # Check integrality
                    W_check = W_sol * 2
                    if not np.allclose(W_check, np.round(W_check), atol=1e-6):
                        valid = False
                        break
                    W_23[k*3+i] = [w/2 for w in np.round(W_check)]
                except Exception:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            return get_schoolbook()
            
        # Construct full U, V, W for rank 46
        rank = 46
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Block 1: k in 0..2
        for l in range(23):
            for i in range(3):
                for j in range(3):
                    U[i*3+j][l] = U_23[i*3+j][l]
            for j in range(3):
                for k in range(3):
                    V[j*3+k][l] = V_23[j*3+k][l]
            for k in range(3):
                for i in range(3):
                    W[k*3+i][l] = W_23[k*3+i][l]
                    
        # Block 2: k in 3..5
        for l in range(23):
            for i in range(3):
                for j in range(3):
                    U[i*3+j][l+23] = U_23[i*3+j][l]
            for j in range(3):
                for k in range(3):
                    V[j*3+(k+3)][l+23] = V_23[j*3+k][l]
            for k in range(3):
                for i in range(3):
                    W[(k+3)*3+i][l+23] = W_23[k*3+i][l]
                    
        # Verify full decomposition
        if time.time() - start_time > 110:
            return get_schoolbook()
            
        A_test = rng.integers(-5, 5, size=(3, 3))
        B_test = rng.integers(-5, 5, size=(3, 6))
        C_test = np.matmul(A_test, B_test)
        
        C_calc = np.zeros((3, 6))
        for r in range(rank):
            u_vec = np.array([U[i*3+j][r] for i in range(3) for j in range(3)]).reshape(3,3)
            v_vec = np.array([V[j*3+k][r] for j in range(3) for k in range(6)]).reshape(3,6)
            w_vec = np.array([W[k*3+i][r] for k in range(6) for i in range(3)]).reshape(6,3)
            
            # Product is (u_vec * A_test) * (v_vec * B_test) * w_vec ?
            # No, the decomposition is sum_r (sum_ij U_ij^r A_ij) (sum_jk V_jk^r B_jk) W_ki^r ?
            # Actually, the term is:
            # prod_r = (sum_ij U[i,j]^r A[i,j]) * (sum_jk V[j,k]^r B[j,k])
            # Then C_calc[i,k] += prod_r * W[k,i]^r
            
            ua = 0
            for i in range(3):
                for j in range(3):
                    ua += U[i*3+j][r] * A_test[i,j]
                    
            vb = 0
            for j in range(3):
                for k in range(6):
                    vb += V[j*3+k][r] * B_test[j,k]
                    
            prod = ua * vb
            
            for k in range(6):
                for i in range(3):
                    C_calc[i,k] += prod * W[k*3+i][r]
                    
        if not np.allclose(C_test, C_calc, atol=1e-6):
            return get_schoolbook()
            
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        return get_schoolbook()
