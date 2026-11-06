"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_size(adjacency, scores):
    """Simulate the fixed greedy decoder: pick vertices in descending score order,
    skipping neighbours of already-picked vertices. Return the number picked."""
    n = adjacency.shape[0]
    order = np.argsort(-scores, kind='stable')
    picked = 0
    blocked = np.zeros(n, dtype=bool)
    for v in order:
        if not blocked[v]:
            picked += 1
            # Mark v and its neighbours as blocked
            blocked[v] = True
            blocked[adjacency[v]] = True
    return picked


def _compute_scores(adjacency, deadline):
    """Compute a composite score vector for vertices.
    
    Strategy: combine multiple structural signals into a single priority.
    - Low degree is good (fewer neighbours to block).
    - Low 'contestation' (neighbours that are also low-degree) is good.
    - Fiedler vector magnitude can help separate vertices in sparse graphs.
    - Iterative refinement: penalize vertices whose neighbours have high scores.
    """
    n = adjacency.shape[0]
    adj = adjacency.astype(np.float64)
    
    # Base signal: negative degree (low degree = high priority)
    deg = adj.sum(axis=1)
    score = -deg.copy()
    
    # Second-order: penalty for being adjacent to low-degree vertices
    # (those are "contested" — if we pick a low-degree neighbour, it blocks us)
    # Actually, we want to AVOID vertices that are neighbours of many low-degree vertices
    # because those low-degree vertices will likely be picked first and block us.
    # So: penalty = sum over neighbours u of (1 / (deg[u] + 1))
    # Higher penalty = worse (subtract from score)
    inv_deg = 1.0 / (deg + 1.0)
    contest = adj @ inv_deg
    score = score - 0.5 * contest
    
    # Iterative refinement: a few rounds of "if my neighbours have high scores, 
    # I should lower my score" (smoothing/competition)
    t0 = time.time()
    for _ in range(3):
        if time.time() > deadline:
            break
        # Normalize scores to [0,1] for the refinement
        s_min, s_max = score.min(), score.max()
        if s_max - s_min > 1e-12:
            s_norm = (score - s_min) / (s_max - s_min)
        else:
            s_norm = np.zeros(n)
        # Average neighbour scores (how "contested" am I?)
        deg_safe = np.maximum(deg, 1.0)
        neigh_score = (adj @ s_norm) / deg_safe
        # Penalize vertices adjacent to high-scoring neighbours
        score = score - 0.3 * neigh_score * (s_max - s_min)
    
    # Fiedler vector signal: vertices with large |Fiedler| component are "central"
    # in the graph's bipartition; picking them early can be suboptimal.
    # So subtract |fiedler| from score (penalize central vertices).
    if n > 1:
        try:
            # Use sparse eigenvalue solver for efficiency
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh
            
            A = csr_matrix(adj)
            # Fiedler vector is the eigenvector for the second smallest eigenvalue
            # of the graph Laplacian L = D - A
            D = np.diag(deg)
            L = D - A
            # eigsh for smallest magnitude eigenvalues
            # We want the second smallest eigenvalue (Fiedler)
            # Use sigma=0 to find eigenvalues near 0
            try:
                # Find two smallest eigenvalues
                vals, vecs = eigsh(L, k=2, sigma=0.0, which='LM')
                # Sort by eigenvalue
                idx = np.argsort(vals)
                fiedler = np.abs(vecs[:, idx[1]])
                # Normalize
                f_max = fiedler.max()
                if f_max > 1e-12:
                    fiedler = fiedler / f_max
                score = score - 0.2 * fiedler
            except Exception:
                pass
        except ImportError:
            pass
    
    return score


def priority(adjacency):
    """Give every vertex a score; higher score = picked earlier by greedy decoder.
    
    The greedy decoder picks vertices in descending score order, skipping
    neighbours of already-picked vertices. We want to maximize the number
    of vertices picked (independent set size).
    """
    deadline = time.time() + 100.0  # Safety margin before 120s limit
    
    adjacency = np.asarray(adjacency)
    n = adjacency.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Compute the composite score
    score = _compute_scores(adjacency, deadline)
    
    # Ensure the scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
