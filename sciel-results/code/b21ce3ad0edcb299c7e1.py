import numpy as np
import time
import itertools
from scipy.optimize import minimize, least_squares

DIMENSIONS = [2, 3, 4]
SEED_DEADLINE = 120.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=4

    def time_left():
        return SEED_DEADLINE - 5.0 - (time.time() - start_time)

    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for l in range(p):
                T[i * m + j, j * p + l, l * n + i] = 1

    def verify(U, V, W, tol=1e-9):
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        r = U.shape[1]
        recon = np.zeros(T.shape, dtype=np.float64)
        for c in range(r):
            recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        if not np.allclose(recon, T, atol=tol, rtol=0):
            return False
        for M in (U, V, W):
            for v in M.flat:
                if abs(v - round(v)) > 1e-6 and abs(2*v - round(2*v)) > 1e-6:
                    return False
        return True

    def make_base_solution():
        rank = 22
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)

        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx, col] += val
            for idx, val in v_indices:
                V[idx, col] += val
            for idx, val in w_indices:
                W[idx, col] += val

        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer: A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W

    U0, V0, W0 = make_base_solution()
    assert verify(U0, V0, W0), "Base solution must be valid"

    best_U, best_V, best_W = U0, V0, W0
    best_rank = 22

    # Try to find rank 21 by perturbing and checking
    # Strategy: for each pair of columns, try to merge them by finding
    # a linear combination that preserves the tensor

    # Flatten the tensor to a vector for linear algebra
    T_flat = T.flatten()

    def col_contribution(U, V, W, c):
        """Get the flat contribution of column c."""
        return np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c]).flatten()

    # Try removing each column and see if we can fix it
    for target in range(22):
        if time_left() < 2.0:
            break
        U_test = np.delete(U0, target, axis=1)
        V_test = np.delete(V0, target, axis=1)
        W_test = np.delete(W0, target, axis=1)
        
        # Compute the residual
        recon = np.zeros(T.shape, dtype=np.float64)
        for c in range(21):
            recon += np.einsum('i,j,k->ijk', U_test[:, c], V_test[:, c], W_test[:, c])
        residual = (T - recon).flatten()
        
        if np.allclose(residual, 0, atol=1e-9):
            best_U, best_V, best_W = U_test, V_test, W_test
            best_rank = 21
            break

    # If rank 21 not found by simple deletion, try a local search
    # Perturb one column slightly and see if another becomes redundant
    if best_rank == 22 and time_left() > 10.0:
        np.random.seed(42)
        for attempt in range(50):
            if time_left() < 3.0:
                break
            # Pick a random column to perturb
            col = np.random.randint(0, 22)
            U_test = U0.copy()
            V_test = V0.copy()
            W_test = W0.copy()
            
            # Add small half-integer perturbation
            perturb = np.random.choice([-0.5, 0.5], size=3)
            # Apply to a random subset of entries
            for mat, dim in [(U_test, 6), (V_test, 12), (W_test, 8)]:
                idx = np.random.randint(0, dim)
                mat[idx, col] += perturb[0]
            
            if not verify(U_test, V_test, W_test, tol=1e-6):
                continue
            
            # Check if any column is now redundant
            for c in range(22):
                U_r = np.delete(U_test, c, axis=1)
                V_r = np.delete(V_test, c, axis=1)
                W_r = np.delete(W_test, c, axis=1)
                if verify(U_r, V_r, W_r, tol=1e-6):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = 21
                    break
            if best_rank == 21:
                break

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
