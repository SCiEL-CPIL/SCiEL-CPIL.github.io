# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation).
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 600  # Safety margin for 630s limit

    # Precompute diagonal indices
    # d ranges from -9 to 9
    # Map d to index 0..18
    diag_map = {}
    diag_cells = {i: [] for i in range(19)}
    for y in range(n):
        for x in range(n):
            d = x - y
            idx = d + 9
            diag_cells[idx].append((y, x))
    
    # Precompute bin indices for x, x+y, x+2y
    # x: 0..9
    # x+y: 0..18
    # x+2y: 0..27
    cell_x_bin = np.zeros((n, n), dtype=int)
    cell_xy_bin = np.zeros((n, n), dtype=int)
    cell_x2y_bin = np.zeros((n, n), dtype=int)
    
    cell_diag_idx = np.zeros((n, n), dtype=int)
    
    for y in range(n):
        for x in range(n):
            cell_x_bin[y, x] = x
            cell_xy_bin[y, x] = x + y
            cell_x2y_bin[y, x] = x + 2 * y
            cell_diag_idx[y, x] = x - y + 9

    def calc_entropies(P):
        """Calculate H_-1, H_0, H_1, H_2 for a normalized P"""
        # P is 10x10
        # H_-1: distribution of x-y
        probs_d = np.zeros(19)
        for i in range(19):
            cells = diag_cells[i]
            probs_d[i] = sum(P[y, x] for y, x in cells)
        
        H_d = 0.0
        for p in probs_d:
            if p > 0:
                H_d -= p * np.log2(p)
        
        # H_0: distribution of x
        probs_x = np.zeros(n)
        for x in range(n):
            probs_x[x] = np.sum(P[:, x])
        H_x = 0.0
        for p in probs_x:
            if p > 0:
                H_x -= p * np.log2(p)
                
        # H_1: distribution of x+y
        probs_xy = np.zeros(2 * n - 1)
        for s in range(2 * n - 1):
            # Sum P[y, x] where x+y=s
            # x+y=s => x = s-y
            # y from max(0, s-9) to min(9, s)
            y_min = max(0, s - (n - 1))
            y_max = min(n - 1, s)
            for y in range(y_min, y_max + 1):
                x = s - y
                probs_xy[s] += P[y, x]
        
        H_xy = 0.0
        for p in probs_xy:
            if p > 0:
                H_xy -= p * np.log2(p)
                
        # H_2: distribution of x+2y
        probs_x2y = np.zeros(3 * n - 1) # 0 to 27
        for val in range(3 * n - 1):
            # x + 2y = val => x = val - 2y
            # y from 0 to 9
            # x must be 0..9
            # val - 2y >= 0 => y <= val/2
            # val - 2y <= 9 => 2y >= val - 9 => y >= (val-9)/2
            y_min = max(0, int(np.ceil((val - 9) / 2.0)))
            y_max = min(n - 1, int(val / 2))
            for y in range(y_min, y_max + 1):
                x = val - 2 * y
                if 0 <= x < n:
                    probs_x2y[val] += P[y, x]
                    
        H_x2y = 0.0
        for p in probs_x2y:
            if p > 0:
                H_x2y -= p * np.log2(p)
                
        return H_d, H_x, H_xy, H_x2y

    def build_P(w):
        """Build P from diagonal weights w (length 19)"""
        P = np.zeros((n, n), dtype=np.float64)
        total_w = sum(w)
        if total_w == 0:
             return None
        for i in range(19):
            cells = diag_cells[i]
            if not cells: continue
            # Distribute weight w[i] uniformly among cells in this diagonal
            p_cell = w[i] / len(cells)
            for y, x in cells:
                P[y, x] = p_cell
        # Normalize just in case (sum w[i]/len(cells) * len(cells) = sum w[i])
        s = np.sum(P)
        if s > 0:
            P /= s
        return P

    def objective(w):
        """Minimize negative ratio"""
        P = build_P(w)
        if P is None:
            return 1e10
        H_d, H_x, H_xy, H_x2y = calc_entropies(P)
        M = max(H_x, H_xy, H_x2y)
        if M < 1e-9:
            return 1e10
        ratio = H_d / M
        return -ratio

    # Initial guess: uniform weights
    w0 = np.ones(19)
    
    best_ratio = 0.0
    best_w = w0.copy()
    
    # Try a few optimization runs with different initializations
    # 1. Uniform
    # 2. Random
    # 3. Centered (higher weight on middle diagonals)
    
    inits = [
        np.ones(19),
        np.random.dirichlet(np.ones(19)),
        np.exp(-np.linspace(-9, 9, 19)**2 / 10) # Gaussian-ish
    ]
    
    for init in inits:
        if time.time() > deadline - 60:
            break
            
        # Normalize init
        init = init / np.sum(init)
        
        # Use Nelder-Mead
        try:
            res = minimize(objective, init, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'fatol': 1e-6, 'xatol': 1e-6})
            if res.success or res.fun < -best_ratio:
                # Check if result is better
                curr_ratio = -res.fun
                if curr_ratio > best_ratio:
                    best_ratio = curr_ratio
                    best_w = res.x
        except Exception:
            pass
            
        # Also try a simple local search / random perturbation if optimization fails or to refine
        # Given time budget, we can do more restarts
        # Let's do a few random restarts of Nelder-Mead
        
    # Additional random restarts if time permits
    restarts = 0
    while time.time() < deadline - 120:
        restarts += 1
        w_rand = np.random.dirichlet(np.ones(19) * 0.5) # Sharp random
        try:
            res = minimize(objective, w_rand, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'fatol': 1e-5, 'xatol': 1e-5})
            if res.fun < -best_ratio:
                best_ratio = -res.fun
                best_w = res.x
        except Exception:
            pass
        
    # Construct final P
    P = build_P(best_w)
    
    # Fallback to uniform if something went wrong
    if P is None or not np.all(np.isfinite(P)) or np.sum(P) <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        # Ensure strict validity
        P[P < 0] = 0
        s = np.sum(P)
        if s > 0:
            P /= s
        else:
            P = np.ones((n, n), dtype=np.float64) / (n * n)
            
    return P

# EVOLVE-BLOCK-END
