import numpy as np
import time
import scipy.optimize as opt

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    timeout = 630
    safety_margin = 30
    deadline = start_time + timeout - safety_margin

    n = 10
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')

    D_vals = X - Y
    H0_vals = X
    H1_vals = X + Y
    H2_vals = X + 2 * Y

    D_uniq, D_inv = np.unique(D_vals, return_inverse=True)
    H0_uniq, H0_inv = np.unique(H0_vals, return_inverse=True)
    H1_uniq, H1_inv = np.unique(H1_vals, return_inverse=True)
    H2_uniq, H2_inv = np.unique(H2_vals, return_inverse=True)

    D_inv_flat = D_inv.ravel()
    H0_inv_flat = H0_inv.ravel()
    H1_inv_flat = H1_inv.ravel()
    H2_inv_flat = H2_inv.ravel()

    D_bins = len(D_uniq)
    H0_bins = len(H0_uniq)
    H1_bins = len(H1_uniq)
    H2_bins = len(H2_uniq)

    def get_entropy(P, inv_flat, n_bins):
        q = np.bincount(inv_flat, weights=P, minlength=n_bins)
        mask = q > 0
        if np.sum(mask) == 0:
            return 0.0
        qnz = q[mask]
        return -np.sum(qnz * np.log2(qnz))

    def get_entropy_grad(P, inv_flat, n_bins):
        q = np.bincount(inv_flat, weights=P, minlength=n_bins)
        mask = q > 0
        if np.sum(mask) == 0:
            return 0.0, np.zeros_like(P)
        qnz = q[mask]
        H = -np.sum(qnz * np.log2(qnz))
        grad_q = np.zeros(n_bins)
        grad_q[mask] = -np.log2(qnz) - 1.0 / np.log(2.0)
        grad_P = grad_q[inv_flat]
        return H, grad_P

    def score_P(P):
        Pf = P.ravel()
        Dv = get_entropy(Pf, D_inv_flat, D_bins)
        H0v = get_entropy(Pf, H0_inv_flat, H0_bins)
        H1v = get_entropy(Pf, H1_inv_flat, H1_bins)
        H2v = get_entropy(Pf, H2_inv_flat, H2_bins)
        M = max(H0v, H1v, H2v)
        if M < 1e-9:
            return 0.0
        return Dv / M

    best_P = np.ones((n, n)) / (n * n)
    best_score = score_P(best_P)

    def update_best(P):
        nonlocal best_P, best_score
        P = np.clip(P, 0, None)
        s = P.sum()
        if s > 1e-9:
            P = P / s
            sc = score_P(P)
            if sc > best_score:
                best_score = sc
                best_P = P.copy()

    # Precompute for combinatorial search
    all_cells = [(yy, xx) for yy in range(n) for xx in range(n)]

    def set_cardinalities(cell_list):
        D_set = set()
        H0_set = set()
        H1_set = set()
        H2_set = set()
        for (yy, xx) in cell_list:
            D_set.add(xx - yy)
            H0_set.add(xx)
            H1_set.add(xx + yy)
            H2_set.add(xx + 2 * yy)
        return len(D_set), len(H0_set), len(H1_set), len(H2_set)

    rng = np.random.default_rng(42)

    # Phase 1: Greedy combinatorial construction
    for trial in range(100):
        if time.time() > deadline:
            break
        shuffled = all_cells[:]
        rng.shuffle(shuffled)
        current = set()
        for _ in range(150):
            if time.time() > deadline:
                break
            best_cell = None
            best_ratio = -1
            for cell in shuffled:
                if cell in current:
                    continue
                new_set = current | {cell}
                cell_list = list(new_set)
                dc, h0c, h1c, h2c = set_cardinalities(cell_list)
                if dc == 0:
                    continue
                m = max(h0c, h1c, h2c)
                if m == 0:
                    continue
                ratio = np.log2(dc) / np.log2(m)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cell = cell
            if best_cell is None:
                break
            current.add(best_cell)
            cell_list = list(current)
            dc, h0c, h1c, h2c = set_cardinalities(cell_list)
            if dc == 19:
                break
        if current:
            P = np.zeros((n, n))
            for (yy, xx) in current:
                P[yy, xx] = 1.0 / len(current)
            update_best(P)

    # Phase 2: Local combinatorial improvement on best support
    if time.time() < deadline:
        support = set()
        for yy in range(n):
            for xx in range(n):
                if best_P[yy, xx] > 1e-6:
                    support.add((yy, xx))
        
        for _ in range(500):
            if time.time() > deadline:
                break
            improved = False
            for (yy, xx) in list(support):
                if time.time() > deadline:
                    break
                for (yy2, xx2) in all_cells:
                    if (yy2, xx2) in support:
                        continue
                    new_support = support - {(yy, xx)} | {(yy2, xx2)}
                    cell_list_new = list(new_support)
                    dc_new, h0c_new, h1c_new, h2c_new = set_cardinalities(cell_list_new)
                    if dc_new == 0:
                        continue
                    m_new = max(h0c_new, h1c_new, h2c_new)
                    if m_new == 0:
                        continue
                    new_ratio = np.log2(dc_new) / np.log2(m_new)
                    
                    cell_list_old = list(support)
                    dc_old, h0c_old, h1c_old, h2c_old = set_cardinalities(cell_list_old)
                    m_old = max(h0c_old, h1c_old, h2c_old)
                    if m_old == 0:
                        continue
                    old_ratio = np.log2(dc_old) / np.log2(m_old)
                    
                    if new_ratio > old_ratio + 1e-9:
                        support = new_support
                        P = np.zeros((n, n))
                        for (yy3, xx3) in support:
                            P[yy3, xx3] = 1.0 / len(support)
                        update_best(P)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    # Phase 3: Continuous optimization
    def objective_and_grad(z, beta=50.0):
        P = z
        Dv, Dg = get_entropy_grad(P, D_inv_flat, D_bins)
        H0v, H0g = get_entropy_grad(P, H0_inv_flat, H0_bins)
        H1v, H1g = get_entropy_grad(P, H1_inv_flat, H1_bins)
        H2v, H2g = get_entropy_grad(P, H2_inv_flat, H2_bins)
        
        H012 = np.array([H0v, H1v, H2v])
        grads_012 = [H0g, H1g, H2g]
        
        scaled_H = beta * H012
        max_scaled = np.max(scaled_H)
        exp_scaled = np.exp(scaled_H - max_scaled)
        sum_exp = np.sum(exp_scaled)
        
        if sum_exp < 1e-300:
            M = np.max(H012)
            if M < 1e-9:
                return -1e9, np.zeros_like(P)
            ratio = Dv / M
            idx_max = np.argmax(H012)
            M_grad = grads_012[idx_max]
            grad_ratio = (M * Dg - Dv * M_grad) / (M * M)
            return -ratio, -grad_ratio
        
        weights = exp_scaled / sum_exp
        M = (max_scaled + np.log(sum_exp)) / beta
        
        if M < 1e-9:
            return -1e9, np.zeros_like(P)
        
        M_grad = np.zeros_like(P)
        for i in range(3):
            M_grad += weights[i] * grads_012[i]
        
        ratio = Dv / M
        grad_ratio = (M * Dg - Dv * M_grad) / (M * M)
        
        return -ratio, -grad_ratio

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                    'jac': lambda x: np.ones_like(x)}]
    bounds = [(0.0, 1.0) for _ in range(n * n)]

    def try_optimize(P_flat, maxiter, beta, ftol=1e-12):
        nonlocal best_P, best_score
        try:
            def obj_wrapper(x):
                return objective_and_grad(x, beta=beta)
            result = opt.minimize(
                obj_wrapper,
                P_flat,
                jac=True,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
            )
            P_opt = np.clip(result.x, 0, None).reshape(n, n)
            update_best(P_opt)
        except Exception:
            pass

    if time.time() < deadline:
        P_flat = best_P.ravel()
        for beta in [30, 50, 100, 200, 400]:
            if time.time() > deadline:
                break
            try_optimize(P_flat, 300, beta)
            P_flat = best_P.ravel()

    # Phase 4: More random combinatorial search
    if time.time() < deadline:
        for trial in range(300):
            if time.time() > deadline:
                break
            k = rng.integers(15, 35)
            indices = rng.choice(n * n, size=k, replace=False)
            support = set()
            for idx in indices:
                yy = idx // n
                xx = idx % n
                support.add((yy, xx))
            
            # Local improvement
            for _ in range(30):
                if time.time() > deadline:
                    break
                improved = False
                for _swap in range(10):
                    if time.time() > deadline:
                        break
                    if not support:
                        break
                    to_remove = rng.choice(list(support))
                    for (yy2, xx2) in all_cells:
                        if (yy2, xx2) in support:
                            continue
                        new_support = support - {to_remove} | {(yy2, xx2)}
                        cell_list_new = list(new_support)
                        dc_new, h0c_new, h1c_new, h2c_new = set_cardinalities(cell_list_new)
                        if dc_new == 0:
                            continue
                        m_new = max(h0c_new, h1c_new, h2c_new)
                        if m_new == 0:
                            continue
                        new_ratio = np.log2(dc_new) / np.log2(m_new)
                        
                        cell_list_old = list(support)
                        dc_old, h0c_old, h1c_old, h2c_old = set_cardinalities(cell_list_old)
                        m_old = max(h0c_old, h1c_old, h2c_old)
                        if m_old == 0:
                            continue
                        old_ratio = np.log2(dc_old) / np.log2(m_old)
                        
                        if new_ratio > old_ratio + 1e-9:
                            support = new_support
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            
            if support:
                P = np.zeros((n, n))
                for (yy, xx) in support:
                    P[yy, xx] = 1.0 / len(support)
                update_best(P)

    # Phase 5: Final polish
    if time.time() < deadline:
        P_flat = best_P.ravel()
        try_optimize(P_flat, 500, 500, ftol=1e-14)
    if time.time() < deadline:
        P_flat = best_P.ravel()
        try_optimize(P_flat, 300, 1000, ftol=1e-15)

    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 0:
        best_P = best_P / s
    else:
        best_P = np.ones((n, n)) / (n * n)

    return best_P
