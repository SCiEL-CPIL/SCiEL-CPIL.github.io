"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    pi = p[i]
    pj = p[j]
    bpj = b[:, pj][p]
    bpi = b[:, pi][p]
    brj = b[pj, :][p]
    bri = b[pi, :][p]
    delta = (a[:, i] * (bpj - bpi) +
             a[:, j] * (bpi - bpj) +
             a[i, :] * (brj - bri) +
             a[j, :] * (bri - brj))
    delta[i] = 0
    delta[j] = 0
    direct = a[i, j] * (b[pj, pi] - b[pi, pj]) + a[j, i] * (b[pi, pj] - b[pj, pi])
    return int(np.sum(delta) + direct)


def _first_improvement_ls(a, b, p, deadline, best_cost, best_p):
    """First-improvement 2-opt local search."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                return best_cost, best_p
            pi = p[i]
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    cur += d
                    improved = True
                    if cur < best_cost:
                        best_cost = cur
                        best_p = p.copy()
                    break
    return best_cost, best_p


def _anneal(a, b, p, deadline, best_cost, best_p, rng, t_start=5000.0, t_min=1.0, cooling=0.99999):
    """Simulated annealing with periodic restart from best."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    T = t_start
    while time.time() < deadline:
        if T < t_min:
            # Restart from best
            T = t_start
            p = best_p.copy()
            cur = best_cost
            continue
        
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        d = _swap_delta(a, b, p, i, j)
        if d < 0 or rng.random() < np.exp(-d / T):
            p[i], p[j] = p[j], p[i]
            cur += d
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
        T *= cooling
    
    # Final local search
    best_cost, best_p = _first_improvement_ls(a, b, p, deadline, best_cost, best_p)
    return best_cost, best_p


def solve(a, b):
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s kill
    
    rng = np.random.default_rng(42)
    
    # --- Constructive start ---
    # Rank matching: heavy flows to central locations
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # --- Phase 1: Local search to polish initial solution ---
    best_cost, best_p = _first_improvement_ls(a, b, p, deadline, best_cost, best_p)
    
    # --- Phase 2: Simulated annealing to escape local optima ---
    # Use remaining time for annealing
    remaining = deadline - time.time()
    if remaining > 5.0:
        # Allocate most time to annealing, keep buffer for final LS
        anneal_deadline = deadline - 5.0
        best_cost, best_p = _anneal(a, b, best_p, anneal_deadline, best_cost, best_p, rng)
    
    # --- Phase 3: Final local search polish ---
    if time.time() < deadline - 1.0:
        best_cost, best_p = _first_improvement_ls(a, b, best_p, deadline, best_cost, best_p)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
