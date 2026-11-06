"""Priority function for Maximum Independent Set via greedy decoder.

Structural change: formulate as Maximum Weight Independent Set on an auxiliary
view where each vertex's weight captures global structure via a spectral
relaxation (power iteration), blended with local degree heuristics. A fast
greedy-based ordering refinement (not full swap local search) adjusts the
final ordering. This captures long-range dependencies that pure degree
heuristics miss.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_is_size(adjacency, order):
    """Given a fixed order (list of vertex indices), greedily pick independent set.
    Returns the size of the independent set."""
    n = len(order)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if not available[v]:
            continue
        count += 1
        nbrs = np.where(adjacency[v])[0]
        available[nbrs] = False
    return count


def _spectral_score(adjacency, n, max_iter=60, tol=1e-7):
    """Power iteration to get a global score vector capturing structure.

    We use power iteration on (A + I) to get the leading eigenvector, which
    captures a global 'spread' signal. We also compute a second-order signal
    using a modified operator to capture independent-set-like structure.
    Returns a score vector (higher = more desirable to pick early).
    """
    A = adjacency.astype(float)
    D = A.sum(axis=1)

    # Leading eigenvector of A (via power iteration) - captures global structure
    x = np.ones(n) / np.sqrt(n)
    x /= (np.linalg.norm(x) + 1e-12)
    for _ in range(max_iter):
        x_new = A @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            break
        x_new /= norm
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    spec1 = x

    # A second signal: use the Laplacian L = D - A. Its leading eigenvector
    # (after removing the trivial constant) captures community spread.
    # We do power iteration on L but orthogonalize against the all-ones vector.
    L = np.diag(D) - A
    y = np.random.RandomState(42).randn(n)
    y -= y.mean()  # orthogonalize against constant
    ny = np.linalg.norm(y)
    if ny < 1e-12:
        y = np.ones(n) / np.sqrt(n)
        y -= y.mean()
    else:
        y /= ny
    for _ in range(max_iter):
        y_new = L @ y
        y_new -= y_new.mean()  # keep orthogonal to constant
        nrm = np.linalg.norm(y_new)
        if nrm < 1e-12:
            break
        y_new /= nrm
        if np.linalg.norm(y_new - y) < tol:
            y = y_new
            break
        y = y_new
    spec2 = y

    return spec1, spec2


def _fast_ordering_refine(adjacency, order, budget_secs=8.0):
    """Lightweight refinement: try moving high-potential vertices earlier.

    Instead of full adjacent-swap local search (O(n^2) evaluations), we do a
    limited number of 'insert' moves: for a few candidate vertices, try inserting
    them at different earlier positions and keep the best improvement.
    This is much cheaper and still captures the most impactful reorderings.
    """
    n = len(order)
    current_size = _greedy_is_size(adjacency, order)
    deadline = time.time() + budget_secs

    # Candidate vertices: those whose neighbors have high total degree (i.e.,
    # picking them 'saves' many potential picks). We try to move them earlier.
    degree = adjacency.sum(axis=1).astype(float)
    nbr_deg_sum = adjacency @ degree
    with np.errstate(divide='ignore', invalid='ignore'):
        potential = np.where(degree > 0, nbr_deg_sum / degree, 0.0)

    # Sort candidates by potential descending, take top ~n/5
    num_cand = max(10, n // 5)
    cand = np.argsort(-potential)[:num_cand]

    improved_any = False
    for v in cand:
        if time.time() > deadline:
            break
        # Current position of v in order
        pos = np.where(order == v)[0]
        if len(pos) == 0:
            continue
        pos = int(pos[0])
        # Try inserting v at a few earlier positions
        best_new = None
        best_size = current_size
        for new_pos in range(0, pos, max(1, pos // 5)):
            if time.time() > deadline:
                break
            order_new = order.copy()
            # Remove v from its current position
            order_new = np.delete(order_new, pos)
            # Insert v at new_pos
            order_new = np.insert(order_new, new_pos, v)
            sz = _greedy_is_size(adjacency, order_new)
            if sz > best_size:
                best_size = sz
                best_new = order_new
        if best_new is not None:
            order = best_new
            current_size = best_size
            improved_any = True

    return order, current_size


def _score_vertices(adjacency):
    """Compute a score for each vertex. Higher score = picked earlier."""
    n = adjacency.shape[0]
    degree = adjacency.sum(axis=1).astype(float)

    # Average degree of neighbors
    nbr_deg_sum = adjacency @ degree
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_nbr_deg = np.where(degree > 0, nbr_deg_sum / degree, 0.0)

    # Local score: prefer low degree and low neighbor-degree
    local_score = -(degree + 0.5 * avg_nbr_deg)

    # Global spectral scores
    spec1, spec2 = _spectral_score(adjacency, n)

    # Normalize spectral scores to comparable scale
    def _norm(z):
        std = z.std()
        if std < 1e-12:
            return z - z.mean()
        return (z - z.mean()) / std

    spec1_n = _norm(spec1)
    spec2_n = _norm(spec2)

    # Blend: local degree signal + spectral structure
    # We weight local heavily but add a spectral component to break ties
    # and capture long-range structure.
    w_local = 1.0
    w_spec = 0.15
    score = w_local * local_score + w_spec * (spec1_n + spec2_n)

    # Initial order: sort by score descending
    order = np.argsort(-score)

    # Fast refinement
    order, _ = _fast_ordering_refine(adjacency, order, budget_secs=8.0)

    # Convert order back to scores
    scores = np.zeros(n, dtype=float)
    for rank, v in enumerate(order):
        scores[v] = -rank

    return scores


def priority(adjacency):
    """Return one score per vertex. Vertices are selected greedily by descending score."""
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill

    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]

    if n == 0:
        return np.array([], dtype=float)

    # Check for timeout early
    if time.time() > deadline:
        degree = adjacency.sum(axis=1).astype(float)
        return -degree

    try:
        scores = _score_vertices(adjacency)
    except Exception:
        # Fallback
        degree = adjacency.sum(axis=1).astype(float)
        scores = -degree

    # Final safety check
    if time.time() > deadline:
        degree = adjacency.sum(axis=1).astype(float)
        return -degree

    return scores
# EVOLVE-BLOCK-END
