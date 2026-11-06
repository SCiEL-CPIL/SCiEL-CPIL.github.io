"""Model-and-optimize 0/1 knapsack heuristic.

Approach: parameterize a scoring model  score(w, v, cap) = f(w, v, cap; params)
and fit params against the objective (maximize total value of a greedy completion)
on a small synthetic distribution. At runtime, pick the item with the best
model score among feasible items, then (if budget allows) verify with a short
greedy completion.
"""
import time
import numpy as np

_START = time.monotonic()
_TOTAL_BUDGET = 140.0  # 10s margin before 150s kill

def _remaining_budget():
    return max(0.0, _TOTAL_BUDGET - (time.monotonic() - _START))

# ---------------------------------------------------------------------------
# Greedy completion model: given a state (weights, values, cap), return the
# total value a greedy policy would collect. This is the OBJECTIVE we fit
# our parameters against.
# ---------------------------------------------------------------------------
def _greedy_total(w, v, cap, mode='density'):
    n = len(w)
    if n == 0 or cap <= 0:
        return 0.0
    safe_w = np.maximum(w, 1e-12)
    if mode == 'density':
        order = np.argsort(-v / safe_w)
    elif mode == 'value':
        order = np.argsort(-v)
    elif mode == 'hybrid':
        order = np.argsort(-(v / safe_w + 0.1 * v))
    else:
        order = np.argsort(-v / safe_w)
    total = 0.0
    c = float(cap)
    for i in order:
        wi = float(w[i])
        if wi <= c:
            total += float(v[i])
            c -= wi
            if c <= 1e-9:
                break
    return total

# ---------------------------------------------------------------------------
# Parameterized scoring model.
# score_i = (v_i / w_i^alpha) * (1 - beta * w_i / cap) + gamma * v_i
# We fit alpha, beta, gamma by maximizing mean greedy-completion value on
# synthetic instances.
# ---------------------------------------------------------------------------
def _model_score(w, v, cap, alpha, beta, gamma):
    safe_w = np.maximum(w, 1e-12)
    density_term = v / (safe_w ** alpha)
    size_penalty = 1.0 - beta * (w / max(cap, 1e-12))
    size_penalty = np.clip(size_penalty, 0.0, 1.0)
    return density_term * size_penalty + gamma * v

def _fit_parameters():
    """Fit alpha, beta, gamma by hill-climbing on synthetic instances."""
    rng = np.random.default_rng(42)
    # Synthetic distribution matching runtime: 100 items, capacity 12.5
    n_items = 100
    n_trials = 10
    instances = []
    for _ in range(n_trials):
        w = rng.uniform(0.05, 1.0, n_items)
        v = rng.uniform(0.05, 1.0, n_items)
        cap = 12.5
        instances.append((w, v, cap))

    # Initialize with reasonable defaults
    alpha = 1.0
    beta = 0.0
    gamma = 0.0

    def _evaluate(params):
        a, b, g = params
        total_val = 0.0
        for w, v, cap in instances:
            feasible = w <= cap
            if not np.any(feasible):
                continue
            scores = _model_score(w[feasible], v[feasible], cap, a, b, g)
            idx = int(np.argmax(scores))
            # Simulate greedy selection with this model, then measure total value
            total_val += _greedy_total_with_model(w[feasible], v[feasible], cap, a, b, g)
        return total_val

    best_score = _evaluate((alpha, beta, gamma))
    best_params = (alpha, beta, gamma)

    # Coordinate-wise hill climbing with smaller steps
    steps = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    for step in steps:
        improved = True
        while improved:
            improved = False
            # Try adjusting alpha
            for da in (-step, step):
                na = max(0.3, min(2.0, alpha + da))
                s = _evaluate((na, beta, gamma))
                if s > best_score:
                    best_score = s
                    alpha = na
                    best_params = (alpha, beta, gamma)
                    improved = True
            # Try adjusting beta
            for db in (-step, step):
                nb = max(-0.5, min(0.5, beta + db))
                s = _evaluate((alpha, nb, gamma))
                if s > best_score:
                    best_score = s
                    beta = nb
                    best_params = (alpha, beta, gamma)
                    improved = True
            # Try adjusting gamma
            for dg in (-step, step):
                ng = max(-0.5, min(0.5, gamma + dg))
                s = _evaluate((alpha, beta, ng))
                if s > best_score:
                    best_score = s
                    gamma = ng
                    best_params = (alpha, beta, gamma)
                    improved = True

    return best_params

def _greedy_total_with_model(w, v, cap, alpha, beta, gamma):
    """Simulate greedy selection using the model score, return total value."""
    n = len(w)
    if n == 0 or cap <= 0:
        return 0.0
    
    total = 0.0
    c = float(cap)
    remaining = np.ones(n, dtype=bool)
    
    for _ in range(n):
        if not np.any(remaining) or c <= 1e-9:
            break
        mask = remaining & (w <= c)
        if not np.any(mask):
            break
        scores = _model_score(w[mask], v[mask], c, alpha, beta, gamma)
        idx = int(np.argmax(scores))
        # Find the actual index
        mask_indices = np.where(mask)[0]
        actual_idx = int(mask_indices[idx])
        total += float(v[actual_idx])
        c -= float(w[actual_idx])
        remaining[actual_idx] = False
    
    return total

# Fit once at import time
_ALPHA, _BETA, _GAMMA = _fit_parameters()

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    cap = float(remaining_capacity)
    if cap <= 0:
        return -1

    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    n = len(w)
    if n == 0:
        return -1

    # Only consider feasible items
    feasible = w <= cap
    if not np.any(feasible):
        return -1

    # Fast path if near deadline: use model score directly
    if _remaining_budget() < 0.2:
        scores = _model_score(w[feasible], v[feasible], cap, _ALPHA, _BETA, _GAMMA)
        idx = int(np.argmax(scores))
        # Map back to original index
        feasible_indices = np.where(feasible)[0]
        return int(feasible_indices[idx])

    # Use the fitted model to score all feasible items
    scores = _model_score(w[feasible], v[feasible], cap, _ALPHA, _BETA, _GAMMA)
    feasible_indices = np.where(feasible)[0]

    # If budget allows, verify with a short greedy-completion lookahead
    # on the top few candidates to confirm the model's choice is good.
    if _remaining_budget() > 0.5 and n <= 200:
        k = min(8, len(feasible_indices))
        top_locals = np.argsort(-scores)[:k]
        best_est = -np.inf
        best_final = int(feasible_indices[top_locals[0]])
        for tl in top_locals:
            ci = int(feasible_indices[tl])
            used_w = float(w[ci])
            used_v = float(v[ci])
            new_cap = cap - used_w
            if new_cap < 0:
                continue
            # Build remaining arrays excluding ci
            mask = np.ones(n, dtype=bool)
            mask[ci] = False
            rw = w[mask]
            rv = v[mask]
            est = used_v + _greedy_total(rw, rv, new_cap, 'density')
            if est > best_est:
                best_est = est
                best_final = ci
            if _remaining_budget() < 0.3:
                break
        best_idx = best_final
    else:
        best_local = int(np.argmax(scores))
        best_idx = int(feasible_indices[best_local])

    return int(best_idx)
