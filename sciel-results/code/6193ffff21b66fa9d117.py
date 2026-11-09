# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        < 0.5 -> route the demand clockwise
        >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    deadline = time.time() + 600.0 - 15.0  # safety margin

    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)

    # Precompute the two route edge-sets per demand.
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod n).
    # Counter-clockwise uses all other ring edges.
    cw_edges = []
    ccw_edges = []
    for (s, t, w) in demands:
        cw = set()
        e = s
        while e != t:
            cw.add(e)
            e = (e + 1) % n
        all_edges = set(range(n))
        ccw = all_edges - cw
        cw_edges.append(cw)
        ccw_edges.append(ccw)

    # Edge incidence vectors: for each demand, a vector over edges for CW and CCW.
    cw_inc = np.zeros((D, n))
    ccw_inc = np.zeros((D, n))
    weights = np.array([w for (_, _, w) in demands], dtype=float)
    for i in range(D):
        for e in cw_edges[i]:
            cw_inc[i, e] = 1.0
        for e in ccw_edges[i]:
            ccw_inc[i, e] = 1.0

    # Load as a function of continuous x in [0,1] (0=CW, 1=CCW).
    # load_e = sum_i w_i * ( (1-x_i)*cw_inc[i,e] + x_i*ccw_inc[i,e] )
    # = sum_i w_i*cw_inc[i,e] + sum_i w_i*(ccw_inc[i,e]-cw_inc[i,e])*x_i
    base_load = (cw_inc * weights[:, None]).sum(axis=0)
    delta = ccw_inc - cw_inc
    delta_w = delta * weights[:, None]  # (D, n)

    def edge_loads(x):
        return base_load + delta_w.T @ x

    def soft_max(x, alpha):
        L = edge_loads(x)
        # log-sum-exp approximation of max
        Lmax = L.max()
        return Lmax + np.log(np.sum(np.exp(alpha * (L - Lmax)))) / alpha

    def objective(x):
        # smooth surrogate: log-sum-exp with alpha
        alpha = 30.0
        L = edge_loads(x)
        Lmax = L.max()
        return Lmax + np.log(np.sum(np.exp(alpha * (L - Lmax)))) / alpha

    def objective_grad(x):
        alpha = 30.0
        L = edge_loads(x)
        Lmax = L.max()
        exps = np.exp(alpha * (L - Lmax))
        p = exps / exps.sum()
        # d/dx_i of objective = alpha * sum_e p_e * delta_w[i,e]
        return alpha * (delta_w * p[None, :]).sum(axis=1)

    def true_max_load(x):
        return edge_loads(x).max()

    def evaluate_binary(bits):
        # bits: 0/1 array, 0=CW, 1=CCW
        x = bits.astype(float)
        L = edge_loads(x)
        return L.max(), L

    # Greedy heuristic to get a good initial binary solution.
    def greedy_initial():
        bits = np.zeros(D, dtype=int)
        # Start with all CW
        best_bits = bits.copy()
        best_load = evaluate_binary(best_bits)[0]
        # Try flipping each demand and keep improvements (hill climbing, multiple passes)
        improved = True
        passes = 0
        while improved and passes < 50 and time.time() < deadline:
            improved = False
            passes += 1
            for i in range(D):
                new_bits = best_bits.copy()
                new_bits[i] = 1 - new_bits[i]
                load = evaluate_binary(new_bits)[0]
                if load < best_load - 1e-9:
                    best_bits = new_bits
                    best_load = load
                    improved = True
        return best_bits, best_load

    init_bits, init_load = greedy_initial()

    best_bits = init_bits.copy()
    best_load = init_load

    # Continuous optimization from multiple starts.
    starts = [init_bits.astype(float)]
    # Add a few random and structured starts
    rng = np.random.default_rng(0)
    for _ in range(4):
        starts.append(rng.uniform(0, 1, D))
    starts.append(np.ones(D))
    starts.append(np.full(D, 0.5))

    for x0 in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                objective, x0, jac=objective_grad,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * D,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            x_opt = res.x
            # Round to binary
            bits = (x_opt >= 0.5).astype(int)
            load = evaluate_binary(bits)[0]
            if load < best_load - 1e-9:
                best_load = load
                best_bits = bits.copy()
        except Exception:
            pass

    # Local polish: single-flip and double-flip hill climbing from best_bits.
    def polish(bits):
        bits = bits.copy()
        load = evaluate_binary(bits)[0]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Single flips
            for i in range(D):
                nb = bits.copy()
                nb[i] = 1 - nb[i]
                l = evaluate_binary(nb)[0]
                if l < load - 1e-9:
                    bits = nb
                    load = l
                    improved = True
            if not improved:
                # Double flips
                for i in range(D):
                    for j in range(i + 1, D):
                        nb = bits.copy()
                        nb[i] = 1 - nb[i]
                        nb[j] = 1 - nb[j]
                        l = evaluate_binary(nb)[0]
                        if l < load - 1e-9:
                            bits = nb
                            load = l
                            improved = True
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
        return bits, load

    best_bits, best_load = polish(best_bits)

    # Convert to the required output convention:
    # entry >= 0.5 routes counter-clockwise, < 0.5 clockwise.
    # Our bits: 0=CW, 1=CCW.
    routing = best_bits.astype(float)
    return routing


# EVOLVE-BLOCK-END
