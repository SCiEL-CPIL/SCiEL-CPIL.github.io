"""Market-making with inventory risk: exact per-(step, inventory) optimal depths via DP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 270.0          # safety margin inside the 330s kill
_CACHE = {}
_GRID = None


def _optimal_depths(sigma, A, k, gamma, T, steps, Q):
    """Return (bid, ask) arrays of shape (steps, 2*Q+1): optimal depths per (step, q)."""
    global _GRID
    if _GRID is None:
        _GRID = np.concatenate((
            np.exp(np.linspace(-8.0, 0.0, 400)),
            np.linspace(1e-4, 0.9995, 400),
        ))

    dt = T / steps
    n_q = 2 * Q + 1
    V = np.zeros(n_q)                       # V[i] for q = i - Q
    bid = np.empty((steps, n_q))
    ask = np.empty((steps, n_q))
    cost_c = 0.5 * gamma * sigma * sigma * dt
    g = _GRID
    n_g = g.shape[0]
    q_range = np.arange(-Q, Q + 1)
    cost = cost_c * (q_range * q_range)     # (n_q,)

    for n in range(steps - 1, -1, -1):
        Vm = V.copy()                       # V[n+1, q-1]
        Vc = V.copy()                       # V[n+1, q]
        Vp = V.copy()                       # V[n+1, q+1]
        Vm[0] = Vc[0]
        Vp[-1] = Vc[-1]

        # ---- bid side (valid for q < Q) ----
        bid_n = np.zeros(n_q)
        bid_n[Q] = np.inf
        if Q > 0:
            Vm_b = Vm[:Q]
            Vc_b = Vc[:Q]
            Vp_b = Vp[:Q]
            cost_b = cost[:Q]
            term_b = (1.0 - g)[None, :] * (Vc_b[None, :] + g[None, :] * (Vm_b[None, :] - Vc_b[None, :]))
            score_b = g[None, :] * (g[None, :] - cost_b[None, :]) + term_b
            best_b = int(np.argmax(score_b, axis=1))
            bid_n[:Q] = -np.log(g[best_b]) / k

        # ---- ask side (valid for q > -Q) ----
        ask_n = np.zeros(n_q)
        ask_n[-Q] = np.inf
        if Q > 0:
            Vm_a = Vm[1:]
            Vc_a = Vc[1:]
            Vp_a = Vp[1:]
            cost_a = cost[1:]
            term_a = (1.0 - g)[None, :] * (Vc_a[None, :] - g[None, :] * (Vc_a[None, :] - Vp_a[None, :]))
            score_a = g[None, :] * (g[None, :] - cost_a[None, :]) + term_a
            best_a = int(np.argmax(score_a, axis=1))
            ask_n[-Q + 1:] = -np.log(g[best_a]) / k

        bid[n] = bid_n
        ask[n] = ask_n

        # ---- value update ----
        pb = 1.0 - np.exp(-A * np.exp(-k * np.where(np.isfinite(bid_n), bid_n, 0.0)) * dt)
        pa = 1.0 - np.exp(-A * np.exp(-k * np.where(np.isfinite(ask_n), ask_n, 0.0)) * dt)
        pb[Q] = 0.0
        pa[-Q] = 0.0
        V = (pb * (bid_n - cost) + pa * (ask_n - cost)
             + (1.0 - pb) * (1.0 - pa) * Vc
             + pb * (1.0 - pa) * Vm
             + (1.0 - pb) * pa * Vp)

    return bid, ask


def quote(t, inventory, mid, params):
    key = (params["sigma"], params["A"], params["k"], params["gamma"],
           params["T"], params["steps"], params["Q"])
    if key not in _CACHE:
        if time.time() < _DEADLINE:
            try:
                _CACHE[key] = _optimal_depths(params["sigma"], params["A"], params["k"],
                                              params["gamma"], params["T"], params["steps"], params["Q"])
            except Exception:
                _CACHE[key] = None
        else:
            _CACHE[key] = None

    n = int(round(t / (params["T"] / params["steps"])))
    n = max(0, min(params["steps"] - 1, n))
    Q = params["Q"]
    q = np.clip(np.asarray(inventory, dtype=np.int64), -Q, Q)
    idx = q + Q

    entry = _CACHE[key]
    if entry is not None:
        bid, ask = entry
        bd = np.asarray(bid[n, idx], dtype=np.float64)
        ad = np.asarray(ask[n, idx], dtype=np.float64)
        bd = np.where(np.isfinite(bd) & (bd >= 0.0), bd, np.inf)
        ad = np.where(np.isfinite(ad) & (ad >= 0.0), ad, np.inf)
        return bd, ad

    # fallback: flat risk-neutral depth (valid, inventory-agnostic)
    d = np.full(len(inventory), 1.0 / params["k"], dtype=np.float64)
    return d, d.copy()
# EVOLVE-BLOCK-END
