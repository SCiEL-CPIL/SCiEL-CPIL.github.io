# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _check_exact(U, V, W, n, m, p):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _alternating_solve(n, m, p, target_rank, max_iters=300, seed=42, deadline=None):
    import numpy as np
    import time
    
    rng = np.random.RandomState(seed)
    nm, mp, pn = n * m, m * p, p * n
    r = target_rank
    
    U = rng.randn(nm, r)
    V = rng.randn(mp, r)
    W = rng.randn(pn, r)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Fix U, W -> solve for V
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                A_mat = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                b_vec = np.ones(n)
                V[v_idx, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Fix V, W -> solve for U
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                A_mat = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                b_vec = np.ones(p)
                U[u_idx, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Fix U, V -> solve for W
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                A_mat = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                b_vec = np.ones(m)
                W[w_idx, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    
    # Check exactness
    U_list = [[float(U[i, c]) for c in range(r)] for i in range(nm)]
    V_list = [[float(V[i, c]) for c in range(r)] for i in range(mp)]
    W_list = [[float(W[i, c]) for c in range(r)] for i in range(pn)]
    
    if _check_exact(U_list, V_list, W_list, n, m, p):
        return U_list, V_list, W_list
    return None


def _random_structured(n, m, p, target_rank, seed=42, deadline=None):
    import time
    
    rng = random.Random(seed)
    nm, mp, pn = n * m, m * p, p * n
    r = target_rank
    
    for _ in range(50):
        if deadline is not None and time.time() > deadline:
            break
        
        U = [[0] * r for _ in range(nm)]
        V = [[0] * r for _ in range(mp)]
        W = [[0] * r for _ in range(pn)]
        
        for c in range(r):
            su = rng.randint(1, 3)
            sv = rng.randint(1, 3)
            sw = rng.randint(1, 3)
            
            for idx in rng.sample(range(nm), min(su, nm)):
                U[idx][c] = rng.choice([-1, 1])
            for idx in rng.sample(range(mp), min(sv, mp)):
                V[idx][c] = rng.choice([-1, 1])
            for idx in rng.sample(range(pn), min(sw, pn)):
                W[idx][c] = rng.choice([-1, 1])
        
        if _check_exact(U, V, W, n, m, p):
            return U, V, W
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    lower_bound = max(n * m, m * p, p * n)
    
    for target_rank in range(lower_bound, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try alternating solve
        for seed in range(10):
            if time.time() > deadline:
                break
            result = _alternating_solve(n, m, p, target_rank, max_iters=100, seed=seed, deadline=deadline)
            if result is not None:
                U_list, V_list, W_list = result
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            continue
        
        # Try random structured
        for seed in range(5):
            if time.time() > deadline:
                break
            result = _random_structured(n, m, p, target_rank, seed=seed, deadline=deadline)
            if result is not None:
                U_list, V_list, W_list = result
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
