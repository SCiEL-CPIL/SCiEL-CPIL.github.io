import numpy as np
import time

def construct():
    n, m, p = 2, 3, 4
    deadline = time.time() + 100

    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_rank_22():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        # A indices: 00:0, 01:1, 10:3, 11:4
        # B indices: 00:0, 01:1, 10:4, 11:5
        # C indices (W): 00:0, 01:2, 10:1, 11:3
        
        # M1: (A00+A11)(B00+B11) -> C00+C11
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        # M2: (A10+A11)B00 -> C10-C11
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        # M3: A00(B01-B11) -> C01+C11
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        # M4: A11(B10-B00) -> C00+C10
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        # M5: (A00+A01)B11 -> -C00+C01
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        # M6: (A10-A00)(B00+B01) -> C11
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        # M7: (A01-A11)(B10+B11) -> C00
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        # B indices: 02:2, 03:3, 12:6, 13:7
        # C indices (W): 02:4, 03:6, 12:5, 13:7
        
        # M1': (A00+A11)(B02+B13) -> C02+C13
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        # M2': (A10+A11)B02 -> C12-C13
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        # M3': A00(B03-B13) -> C03+C13
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        # M4': A11(B12-B02) -> C02+C12
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        # M5': (A00+A01)B13 -> -C02+C03
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        # M6': (A10-A00)(B02+B03) -> C13
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        # M7': (A01-A11)(B12+B13) -> C02
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        # A indices: 02:2, 12:5
        # B indices: 20:8, 21:9, 22:10, 23:11
        # C indices: 00:0, 01:2, 02:4, 03:6 for i=0
        #            10:1, 11:3, 12:5, 13:7 for i=1
        
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W

    # First, ensure we have a valid baseline
    U_base, V_base, W_base = build_rank_22()
    if verify(U_base, V_base, W_base, 22):
        best_result = {"U": U_base, "V": V_base, "W": W_base}
    else:
        # Fallback: simple schoolbook rank 24
        U = [[0]*24 for _ in range(n*m)]
        V = [[0]*24 for _ in range(m*p)]
        W = [[0]*24 for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t = i*m*p + j*p + k
                    U[i*m+j][t] = 1
                    V[j*p+k][t] = 1
                    W[k*n+i][t] = 1
        best_result = {"U": U, "V": V, "W": W}
        if not verify(U, V, W, 24):
            raise RuntimeError("Baseline verification failed")

    def try_als(rank, max_iter=500, tol=1e-10, seed=42):
        """Alternating least squares to find rank decomposition."""
        rng = np.random.RandomState(seed)
        
        # Initialize with random sparse-ish vectors
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = np.ones((p * n, rank)) # Start with 1s for W to help convergence
        
        best_err = float('inf')
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Solve for W: fix U, V
            # For each (k, i), we want sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] = 1 for all j
            # Let x_t = W[k*n+i, t]. Then sum_t x_t (U[i*m+j, t] V[j*p+k, t]) = 1
            # This is a linear system: A x = b, where A[j, t] = U[i*m+j, t] V[j*p+k, t], b[j] = 1
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        W_new[w, :] = rng.randn(rank)
            
            # Solve for U: fix V, W
            # For each (i, j), sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] = 1 for all k
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        U_new[u, :] = rng.randn(rank)
            
            # Solve for V: fix U, W
            # For each (j, k), sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] = 1 for all i
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        V_new[v, :] = rng.randn(rank)
            
            # Check convergence / error
            U, V, W = U_new, V_new, V_new
            
            # Calculate error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for t in range(rank):
                            s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                        err += (s - 1.0)**2
            err = np.sqrt(err)
            
            if err < best_err:
                best_err = err
                if err < 1e-8:
                    # Verify exactly
                    U_list = [[round(float(x), 10) for x in row] for row in U]
                    V_list = [[round(float(x), 10) for x in row] for row in V]
                    W_list = [[round(float(x), 10) for x in row] for row in W]
                    if verify(U_list, V_list, W_list, rank):
                        return U_list, V_list, W_list

        return None

    # Try to find lower rank
    for rank in range(20, 22):
        if time.time() > deadline - 10:
            break
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            result = try_als(rank, max_iter=300, tol=1e-12, seed=seed)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}

    return best_result
