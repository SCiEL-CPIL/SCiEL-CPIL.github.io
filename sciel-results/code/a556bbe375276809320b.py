import time
import numpy as np

DIMENSIONS = [3, 3, 5]
TIME_LIMIT = 140

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    A_dim, B_dim, C_dim = n * m, m * p, p * n

    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support.append((a, b, c))

    def elapsed():
        return time.time() - t0

    def check_deadline():
        return elapsed() > TIME_LIMIT - 5

    def verify(U, V, W, tol=1e-6):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        T_target = np.zeros((A_dim, B_dim, C_dim))
        for a, b, c in support:
            T_target[a, b, c] = 1.0
        return np.allclose(T_recon, T_target, atol=tol)

    def round_to_half(X):
        return np.round(X * 2.0) / 2.0

    # Schoolbook
    U_school = np.zeros((A_dim, 45))
    V_school = np.zeros((B_dim, 45))
    W_school = np.zeros((C_dim, 45))
    for col, (a, b, c) in enumerate(support):
        U_school[a, col] = 1
        V_school[b, col] = 1
        W_school[c, col] = 1

    best_result = (U_school, V_school, W_school)
    best_rank = 45

    # Try to find lower rank using ALS with full constraints
    # Use a subset of non-support entries to keep it tractable
    support_set = set(support)
    all_entries = [(a, b, c) for a in range(A_dim) for b in range(B_dim) for c in range(C_dim)]
    
    # Use all support + a random subset of non-support
    non_support = [(a, b, c) for a, b, c in all_entries if (a, b, c) not in support_set]
    
    def solve_als_subset(U, V, W, r, max_iter=150, num_non_support=200, seed=0):
        np.random.seed(seed)
        # Select subset of non-support entries
        if len(non_support) > num_non_support:
            idx = np.random.choice(len(non_support), num_non_support, replace=False)
            non_support_subset = [non_support[i] for i in idx]
        else:
            non_support_subset = non_support
        
        eqs = support + non_support_subset
        targets = [1.0 if (a,b,c) in support_set else 0.0 for a,b,c in eqs]
        
        from collections import defaultdict
        eqs_a = defaultdict(list)
        eqs_b = defaultdict(list)
        eqs_c = defaultdict(list)
        for (a, b, c), t in zip(eqs, targets):
            eqs_a[a].append((b, c, t))
            eqs_b[b].append((a, c, t))
            eqs_c[c].append((a, b, t))
        
        U = U.astype(float).copy()
        V = V.astype(float).copy()
        W = W.astype(float).copy()
        
        for it in range(max_iter):
            if check_deadline():
                break
            U_new = np.zeros_like(U)
            for a in range(A_dim):
                if a in eqs_a:
                    M = np.array([V[b] * W[c] for b, c, _ in eqs_a[a]])
                    rhs = np.array([t for _, _, t in eqs_a[a]])
                    sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    U_new[a] = sol
            U = U_new
            V_new = np.zeros_like(V)
            for b in range(B_dim):
                if b in eqs_b:
                    M = np.array([U[a] * W[c] for a, c, _ in eqs_b[b]])
                    rhs = np.array([t for _, _, t in eqs_b[b]])
                    sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    V_new[b] = sol
            V = V_new
            W_new = np.zeros_like(W)
            for c in range(C_dim):
                if c in eqs_c:
                    M = np.array([U[a] * V[b] for a, b, _ in eqs_c[c]])
                    rhs = np.array([t for _, _, t in eqs_c[c]])
                    sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    W_new[c] = sol
            W = W_new
            if it > 0:
                diff = np.linalg.norm(U - U_prev) + np.linalg.norm(V - V_prev) + np.linalg.norm(W - W_prev)
                if diff < 1e-10:
                    break
            U_prev, V_prev, W_prev = U.copy(), V.copy(), W.copy()
        return U, V, W

    for trial_rank in range(15, 46):
        if check_deadline():
            break
        found = False
        for trial in range(8):
            if check_deadline():
                break
            np.random.seed(trial_rank * 100 + trial)
            U_init = np.random.randn(A_dim, trial_rank) * 0.5
            V_init = np.random.randn(B_dim, trial_rank) * 0.5
            W_init = np.random.randn(C_dim, trial_rank) * 0.5
            U_sol, V_sol, W_sol = solve_als_subset(U_init, V_init, W_init, trial_rank, max_iter=100, num_non_support=150, seed=trial_rank*1000+trial)
            if verify(U_sol, V_sol, W_sol, tol=1e-5):
                U_round = round_to_half(U_sol)
                V_round = round_to_half(V_sol)
                W_round = round_to_half(W_sol)
                if verify(U_round, V_round, W_round, tol=1e-5):
                    best_rank = trial_rank
                    best_result = (U_round, V_round, W_round)
                else:
                    best_rank = trial_rank
                    best_result = (U_sol, V_sol, W_sol)
                found = True
                break
        if found:
            break

    U_out, V_out, W_out = best_result
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
