import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12

    start_time = time.time()
    deadline = start_time + 110

    def build_rank44():
        rank = 44
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]

        col = 0
        def add_strassen(X_u, Y_v, Z_w, col):
            U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1
            W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
            col += 1
            U[X_u[0]][col] += 1
            V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
            W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            U[X_u[3]][col] += 1
            V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
            W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
            col += 1
            U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
            V[Y_v[3]][col] += 1
            W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
            col += 1
            U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
            V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
            W[Z_w[3]][col] += 1
            col += 1
            U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
            V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1
            col += 1
            return col

        col = add_strassen([0,1,4,5], [0,1,4,5], [0,3,1,4], col)
        col = add_strassen([2,3,6,7], [8,9,12,13], [0,3,1,4], col)
        col = add_strassen([0,1,4,5], [2,3,6,7], [6,9,7,10], col)
        col = add_strassen([2,3,6,7], [10,11,14,15], [6,9,7,10], col)

        for j in range(4):
            U[8+j][col] += 1; V[j*4+0][col] += 1; W[2][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+1][col] += 1; W[5][col] += 1
            col += 1
        for j in range(4):
            U[8+j][col] += 1; V[j*4+2][col] += 1; W[8][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+3][col] += 1; W[11][col] += 1
            col += 1

        return U, V, W

    def verify(U, V, W, n, m, p):
        U_int = np.round(np.array(U) * 2).astype(np.int64)
        V_int = np.round(np.array(V) * 2).astype(np.int64)
        W_int = np.round(np.array(W) * 2).astype(np.int64)

        T_expected = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i*m+j, j*p+k, k*n+i] = 8

        T_actual = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
        return np.array_equal(T_actual, T_expected)

    def fast_verify(U, V, W):
        """Faster verification using float arithmetic with tolerance."""
        U_f = np.array(U, dtype=np.float64)
        V_f = np.array(V, dtype=np.float64)
        W_f = np.array(W, dtype=np.float64)

        T_expected = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i*m+j, j*p+k, k*n+i] = 1.0

        T_actual = np.einsum('ir,jr,kr->ijk', U_f, V_f, W_f)
        return np.allclose(T_actual, T_expected, atol=1e-9)

    def verify_fast_int(U_arr, V_arr, W_arr):
        """Fast exact verification using precomputed expected tensor and int64 einsum."""
        U_int = np.round(U_arr * 2).astype(np.int64)
        V_int = np.round(V_arr * 2).astype(np.int64)
        W_int = np.round(W_arr * 2).astype(np.int64)
        T_actual = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
        # Check against expected: T[i*m+j, j*p+k, k*n+i] = 8 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T_actual[i*m+j, j*p+k, k*n+i] != 8:
                        return False
        # Check all other entries are 0
        T_check = T_actual.copy()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_check[i*m+j, j*p+k, k*n+i] = 0
        return np.all(T_check == 0)

    # Build the rank-44 decomposition
    U, V, W = build_rank44()
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # First verify the base decomposition is correct
    if not verify_fast_int(U_np, V_np, W_np):
        rank = 48
        U = [[0]*rank for _ in range(dim_u)]
        V = [[0]*rank for _ in range(dim_v)]
        W = [[0]*rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to reduce rank by checking linear dependence and removing redundant columns
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    current_rank = 44

    # Precompute expected tensor for fast checking
    T_expected = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 8

    def verify_against_expected(U_arr, V_arr, W_arr):
        """Verify decomposition against precomputed expected tensor."""
        U_int = np.round(U_arr * 2).astype(np.int64)
        V_int = np.round(V_arr * 2).astype(np.int64)
        W_int = np.round(W_arr * 2).astype(np.int64)
        T_actual = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
        return np.array_equal(T_actual, T_expected)

    # Phase 1: Try removing single columns (fast pass)
    for attempt in range(60):
        if time.time() > deadline:
            break
        if current_rank <= 1:
            break

        found = False
        for r_idx in range(current_rank):
            if time.time() > deadline:
                break
            
            keep_indices = [i for i in range(current_rank) if i != r_idx]
            
            U_test = current_U[:, keep_indices]
            V_test = current_V[:, keep_indices]
            W_test = current_W[:, keep_indices]
            
            if verify_against_expected(U_test, V_test, W_test):
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 1
                found = True
                break
        
        if not found:
            break

    # Phase 2: Try removing pairs of columns (slower but more thorough)
    pair_deadline = min(deadline, time.time() + 40)
    if current_rank > 1:
        found_pair = False
        for r1 in range(current_rank):
            if time.time() > pair_deadline:
                break
            for r2 in range(r1+1, current_rank):
                if time.time() > pair_deadline:
                    break
                
                keep_indices = [i for i in range(current_rank) if i != r1 and i != r2]
                
                U_test = current_U[:, keep_indices]
                V_test = current_V[:, keep_indices]
                W_test = current_W[:, keep_indices]
                
                if verify_against_expected(U_test, V_test, W_test):
                    current_U = U_test
                    current_V = V_test
                    current_W = W_test
                    current_rank -= 2
                    found_pair = True
                    break
            if found_pair:
                break
        if found_pair:
            # After removing a pair, try single removals again
            for attempt in range(30):
                if time.time() > deadline:
                    break
                if current_rank <= 1:
                    break
                found = False
                for r_idx in range(current_rank):
                    if time.time() > deadline:
                        break
                    keep_indices = [i for i in range(current_rank) if i != r_idx]
                    U_test = current_U[:, keep_indices]
                    V_test = current_V[:, keep_indices]
                    W_test = current_W[:, keep_indices]
                    if verify_against_expected(U_test, V_test, W_test):
                        current_U = U_test
                        current_V = V_test
                        current_W = W_test
                        current_rank -= 1
                        found = True
                        break
                if not found:
                    break

    # Phase 3: Try local perturbation - adjust columns slightly to enable removal
    # This is a heuristic: try adding small integer adjustments to see if we can
    # create a dependency that allows removal
    if current_rank > 1 and time.time() < deadline - 20:
        # Try swapping column orderings or combining
        best_rank = current_rank
        best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
        
        # Try all permutations of a subset of columns to see if different ordering helps
        # (This is limited - just try a few random subsets)
        import random
        random.seed(42)
        
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            # Pick a random column to try removing
            r_idx = random.randint(0, current_rank - 1)
            keep_indices = [i for i in range(current_rank) if i != r_idx]
            
            U_test = current_U[:, keep_indices]
            V_test = current_V[:, keep_indices]
            W_test = current_W[:, keep_indices]
            
            if verify_against_expected(U_test, V_test, W_test):
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 1
                best_rank = current_rank
                best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
            else:
                # Try perturbing one of the kept columns by ±1 in a random position
                # to see if it enables the removal
                pert_col = random.randint(0, len(keep_indices) - 1)
                pert_row = random.randint(0, dim_u - 1)
                pert_val = random.choice([-1, 1])
                
                U_pert = U_test.copy()
                U_pert[pert_row, pert_col] += pert_val * 0.5  # half-integer perturbation
                
                if verify_against_expected(U_pert, V_test, W_test):
                    current_U = U_pert
                    current_V = V_test.copy()
                    current_W = W_test.copy()
                    current_rank -= 1
                    best_rank = current_rank
                    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()

    # Final verification
    if current_rank < 44:
        if verify_against_expected(current_U, current_V, current_W):
            return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}

    return {"U": U, "V": V, "W": W}
