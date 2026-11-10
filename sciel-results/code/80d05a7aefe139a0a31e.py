# EVOLVE-BLOCK-START
import numpy as np
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Maximize D/M where D = H(x-y) and M = max(H(x), H(x+y), H(x+2y)).
    """
    start_time = time.time()
    deadline = start_time + 580
    
    n = 10
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X = X.flatten()
    Y = Y.flatten()
    
    idx_m1 = X - Y + 9
    idx_0 = X
    idx_1 = X + Y
    idx_2 = X + 2 * Y
    
    n_m1 = 19
    n_0 = 10
    n_1 = 19
    n_2 = 27
    
    def shannon_entropy(p):
        p = p[p > 1e-15]
        if len(p) == 0:
            return 0.0
        return -np.sum(p * np.log2(p))
    
    def compute_D_M(P_flat):
        P_flat = np.maximum(P_flat, 0)
        s = P_flat.sum()
        if s <= 1e-12:
            return 0.0, 0.0
        P_flat = P_flat / s
        
        p_m1 = np.zeros(n_m1)
        np.add.at(p_m1, idx_m1, P_flat)
        p_0 = np.zeros(n_0)
        np.add.at(p_0, idx_0, P_flat)
        p_1 = np.zeros(n_1)
        np.add.at(p_1, idx_1, P_flat)
        p_2 = np.zeros(n_2)
        np.add.at(p_2, idx_2, P_flat)
        
        D = shannon_entropy(p_m1)
        M = max(shannon_entropy(p_0), shannon_entropy(p_1), shannon_entropy(p_2))
        return D, M
    
    # Initialize
    rng = np.random.RandomState(42)
    best_P = None
    best_val = 0.0
    
    P_init = np.ones(100) / 100.0
    D, M = compute_D_M(P_init)
    val = D / M if M > 1e-12 else 0.0
    if val > best_val:
        best_val = val
        best_P = P_init.copy()
    
    for trial in range(100):
        k = rng.randint(8, 35)
        support = rng.choice(100, size=k, replace=False)
        P_try = np.zeros(100)
        P_try[support] = 1.0 / k
        D, M = compute_D_M(P_try)
        val = D / M if M > 1e-12 else 0.0
        if val > best_val:
            best_val = val
            best_P = P_try.copy()
    
    P = best_P.copy()
    P = np.maximum(P, 0)
    P = P / P.sum()
    
    # Gradient ascent with soft-max annealing
    n_iter = 30000
    lr_start = 0.03
    lr_end = 0.0005
    
    for i in range(n_iter):
        if time.time() > deadline - 60:
            break
        
        beta = 1.0 + 99.0 * (i / n_iter) ** 2
        
        P_safe = np.maximum(P, 1e-15)
        P_safe = P_safe / P_safe.sum()
        
        p_m1 = np.zeros(n_m1)
        np.add.at(p_m1, idx_m1, P_safe)
        p_0 = np.zeros(n_0)
        np.add.at(p_0, idx_0, P_safe)
        p_1 = np.zeros(n_1)
        np.add.at(p_1, idx_1, P_safe)
        p_2 = np.zeros(n_2)
        np.add.at(p_2, idx_2, P_safe)
        
        D = shannon_entropy(p_m1)
        H0 = shannon_entropy(p_0)
        H1 = shannon_entropy(p_1)
        H2 = shannon_entropy(p_2)
        
        e = np.array([H0, H1, H2]) * beta
        m = np.max(e)
        exp_e = np.exp(e - m)
        w = exp_e / np.sum(exp_e)
        M_soft = (m + np.log(np.sum(exp_e))) / beta
        
        if M_soft <= 1e-12:
            break
        
        # Gradient of H w.r.t. P_flat: for each cell i, dH/dP_i = -log2(p_pool) - 1/ln(2)
        # where p_pool is the pooled probability for the pool containing i
        dD_dP = np.zeros(100)
        dD_dP[idx_m1] = -np.log2(np.maximum(p_m1[idx_m1], 1e-15)) - 1.0 / np.log(2.0)
        
        dM_dP = np.zeros(100)
        dM_dP[idx_0] += w[0] * (-np.log2(np.maximum(p_0[idx_0], 1e-15)) - 1.0 / np.log(2.0))
        dM_dP[idx_1] += w[1] * (-np.log2(np.maximum(p_1[idx_1], 1e-15)) - 1.0 / np.log(2.0))
        dM_dP[idx_2] += w[2] * (-np.log2(np.maximum(p_2[idx_2], 1e-15)) - 1.0 / np.log(2.0))
        
        dObj_dP = (dD_dP * M_soft - D * dM_dP) / (M_soft ** 2)
        
        lr = lr_start * (lr_end / lr_start) ** (i / n_iter)
        P_new = P + lr * dObj_dP
        P_new = np.maximum(P_new, 0)
        s = P_new.sum()
        if s > 1e-12:
            P_new = P_new / s
        P = P_new
    
    # Final polish
    for i in range(10000):
        if time.time() > deadline - 30:
            break
        
        P_safe = np.maximum(P, 1e-15)
        P_safe = P_safe / P_safe.sum()
        
        p_m1 = np.zeros(n_m1)
        np.add.at(p_m1, idx_m1, P_safe)
        p_0 = np.zeros(n_0)
        np.add.at(p_0, idx_0, P_safe)
        p_1 = np.zeros(n_1)
        np.add.at(p_1, idx_1, P_safe)
        p_2 = np.zeros(n_2)
        np.add.at(p_2, idx_2, P_safe)
        
        D = shannon_entropy(p_m1)
        H0 = shannon_entropy(p_0)
        H1 = shannon_entropy(p_1)
        H2 = shannon_entropy(p_2)
        M = max(H0, H1, H2)
        
        if M <= 1e-12:
            break
        
        dD_dP = np.zeros(100)
        dD_dP[idx_m1] = -np.log2(np.maximum(p_m1[idx_m1], 1e-15)) - 1.0 / np.log(2.0)
        
        dM_dP = np.zeros(100)
        if H0 >= H1 and H0 >= H2:
            dM_dP[idx_0] += -np.log2(np.maximum(p_0[idx_0], 1e-15)) - 1.0 / np.log(2.0)
        if H1 >= H0 and H1 >= H2:
            dM_dP[idx_1] += -np.log2(np.maximum(p_1[idx_1], 1e-15)) - 1.0 / np.log(2.0)
        if H2 >= H0 and H2 >= H1:
            dM_dP[idx_2] += -np.log2(np.maximum(p_2[idx_2], 1e-15)) - 1.0 / np.log(2.0)
        
        dObj_dP = (dD_dP * M - D * dM_dP) / (M ** 2)
        
        lr = 0.001 * (1.0 - i / 10000.0) + 0.0001
        P_new = P + lr * dObj_dP
        P_new = np.maximum(P_new, 0)
        s = P_new.sum()
        if s > 1e-12:
            P_new = P_new / s
        P = P_new
    
    P = np.maximum(P, 0)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones(100) / 100.0
    else:
        P = P / s
    
    return P.reshape(10, 10)

# EVOLVE-BLOCK-END
