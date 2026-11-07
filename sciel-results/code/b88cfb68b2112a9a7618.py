# EVOLVE-BLOCK-START
import numpy as np

_Q = 30
_STEPS = 1000
_T = 1.0
_DT = _T / _STEPS

# Depth action grid: log-spaced, very fine near 0 (exponential sensitivity), capped at 10/k
_MIN_D = 1e-5
_MAX_D = 10.0 / 1.5          # 10/k for the smallest k (0.75); for k=1.5 -> 10/k=6.67
_LOG_N = 320
_GRID = np.exp(np.linspace(np.log(_MIN_D), np.log(_MAX_D), _LOG_N))
# Add an explicit "no quote" sentinel (very large depth -> p ~ 0)
_GRID = np.concatenate([_GRID, [1e9]])

# Precomputed tables per regime: bid_depths[q, tau], ask_depths[q, tau]
# q index: 0..60 corresponds to inventory -30..+30
# tau index: 0..999 corresponds to steps remaining 0..999
_TABLES = {}


def _build_table(sigma, A, k, gamma):
    Q = _Q
    nq = 2 * Q + 1
    ntau = _STEPS
    bid_tbl = np.empty((ntau, nq))
    ask_tbl = np.empty((ntau, nq))

    # V[q, tau] for tau = 0..ntau-1
    V = np.zeros((nq, ntau))
    V[:, 0] = 0.0

    cost = 0.5 * gamma * sigma * sigma * _DT

    for tau in range(1, ntau):
        Vq = V[:, tau - 1]
        # mid shift variance
        s2 = sigma * sigma * _DT
        # E[V(q+1)], E[V(q-1)] with no-fill-at-boundary
        Vq_up = Vq.copy()
        Vq_up[0] = Vq[0]      # q=-Q: no ask fill
        Vq_dn = Vq.copy()
        Vq_dn[-1] = Vq[-1]    # q=+Q: no bid fill

        base = -cost * Vq + Vq  # -cost*q^2 + (1-pb-pa)*Vq ; we'll add pb*Vq_dn + pa*Vq_up
        # Actually: base = Vq - cost*Vq  (since (1-pb-pa)*Vq + pb*Vq_dn + pa*Vq_up
        #           = Vq + pb*(Vq_dn - Vq) + pa*(Vq_up - Vq))
        Vq_minus = Vq_dn - Vq   # (V(q-1) - V(q))
        Vq_plus = Vq_up - Vq    # (V(q+1) - V(q))

        # For each depth d: p = 1 - exp(-A * exp(-k*d) * dt)
        # Bid value: Vq + p * Vq_minus
        # Ask value: Vq + p * Vq_plus
        # Maximize over d independently.
        kD = k * _GRID
        Ae = A * np.exp(-kD) * _DT
        p = 1.0 - np.exp(-Ae)  # shape (LOG_N+1,)

        # bid_vals[q, g] = Vq[q] + p[g] * Vq_minus[q]
        bid_vals = Vq[:, None] + p[None, :] * Vq_minus[:, None]
        ask_vals = Vq[:, None] + p[None, :] * Vq_plus[:, None]

        bi_idx = np.argmax(bid_vals, axis=1)
        ai_idx = np.argmax(ask_vals, axis=1)

        bid_tbl[tau, :] = _GRID[bi_idx]
        ask_tbl[tau, :] = _GRID[ai_idx]

        # Update V[:, tau] = Vq + p_bid * Vq_minus + p_ask * Vq_plus
        pb = p[bi_idx]
        pa = p[ai_idx]
        V[:, tau] = Vq + pb * Vq_minus + pa * Vq_plus

    return bid_tbl, ask_tbl


def _get_table(sigma, A, k, gamma):
    key = (sigma, A, k, gamma)
    if key not in _TABLES:
        _TABLES[key] = _build_table(sigma, A, k, gamma)
    return _TABLES[key]


def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]

    bid_tbl, ask_tbl = _get_table(sigma, A, k, gamma)

    n = int(round(t / _DT))
    n = max(0, min(n, _STEPS - 1))
    tau = _STEPS - 1 - n  # steps remaining

    q_idx = inventory + _Q
    bid_depths = bid_tbl[tau, q_idx]
    ask_depths = ask_tbl[tau, q_idx]

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
