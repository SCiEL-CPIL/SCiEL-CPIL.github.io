import numpy as np
import time

def _greedy_decode_fast(adj_bits, scores, n):
    """Fast greedy decode using integer bitmasks. adj_bits[v] is an int bitmask of neighbors of v."""
    order = np.argsort(-scores, kind='stable')
    forbidden_mask = 0
    count = 0
    for v in order:
        if forbidden_mask & (1 << v):
            continue
        count += 1
        forbidden_mask |= adj_bits[v]
        forbidden_mask |= (1 << v)
    return count

def _build_adj_bits(adj, n):
    """Convert boolean adjacency matrix to list of integer bitmasks."""
    adj_bits = []
    for v in range(n):
        bits = 0
        row = adj[v]
        # Use numpy to find set bits and construct integer
        indices = np.where(row)[0]
        for idx in indices:
            bits |= (1 << idx)
        adj_bits.append(bits)
    return adj_bits

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Build fast bitmask representation
    adj_bits = _build_adj_bits(adj, n)
    
    # 1. Compute Degrees
    degrees = adj.sum(axis=1).astype(float)
    
    # 2. Compute Average Neighbor Degree
    nbr_deg_sums = adj @ degrees
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_nbr_deg = np.where(degrees > 0, nbr_deg_sums / degrees, 0.0)
    
    # 3. Spectral components
    v1 = np.zeros(n)
    v2 = np.zeros(n)
    try:
        if time.time() < deadline - 20:
            eigvals, eigvecs = np.linalg.eigh(adj.astype(float))
            # Largest eigenvalue (Perron vector)
            v1 = eigvecs[:, -1]
            max_val = np.max(np.abs(v1))
            if max_val > 0:
                v1 = v1 / max_val
            else:
                v1 = np.zeros(n)
            
            # Second smallest eigenvalue of Laplacian (Fiedler vector)
            # L = D - A
            D = np.diag(degrees)
            L = D - adj.astype(float)
            eigvals_L, eigvecs_L = np.linalg.eigh(L)
            # Fiedler vector is the second smallest eigenvalue's eigenvector
            # (smallest is trivial all-ones)
            v2 = eigvecs_L[:, 1]
            max_val2 = np.max(np.abs(v2))
            if max_val2 > 0:
                v2 = v2 / max_val2
            else:
                v2 = np.zeros(n)
        else:
            v1 = np.zeros(n)
            v2 = np.zeros(n)
    except Exception:
        v1 = np.zeros(n)
        v2 = np.zeros(n)
    
    # 4. Combine Scores
    # Hub-blocking: low degree is good, high avg neighbor degree is good
    # Spectral: Fiedler vector gives bipartition; we want to pick one side
    # Use |v2| to identify which side, and bias toward the side with fewer connections
    w_deg = 1.0
    w_nbr = 1.0
    w_fiedler = 0.8
    
    # For Fiedler: vertices with positive v2 are on one side, negative on the other.
    # We want to pick vertices from the side that is more "independent" (less internal edges).
    # Simple heuristic: use the sign of v2 directly, weighted.
    # Actually, for independent set, we want to pick vertices that are NOT connected to each other.
    # The Fiedler vector separates the graph into two communities. Within each community,
    # there are fewer edges. So picking all vertices from one community gives a good independent set.
    # We should bias scores so that one community gets higher scores.
    # Determine which community has fewer internal edges:
    pos_mask = v2 > 0
    neg_mask = v2 < 0
    pos_edges = adj[np.ix_(pos_mask, pos_mask)].sum() / 2
    neg_edges = adj[np.ix_(neg_mask, neg_mask)].sum() / 2
    
    # Pick the community with fewer internal edges (more independent)
    if pos_edges <= neg_edges:
        # Boost positive side
        fiedler_score = np.where(v2 > 0, 1.0, -1.0)
    else:
        # Boost negative side
        fiedler_score = np.where(v2 < 0, 1.0, -1.0)
    
    scores = -w_deg * degrees + w_nbr * avg_nbr_deg + w_fiedler * fiedler_score
    
    # 5. Local Polish with fast decoder
    best_count = _greedy_decode_fast(adj_bits, scores, n)
    best_scores = scores.copy()
    
    max_polish_iters = 50
    for it in range(max_polish_iters):
        if time.time() > deadline:
            break
        improved = False
        # Try single-vertex score perturbations: for each vertex, try boosting it
        # above all others or below all others to see if greedy improves
        for v in range(n):
            if time.time() > deadline:
                break
            
            # Try boosting vertex v to the top
            orig_score = scores[v]
            scores[v] = np.max(scores) + 1.0
            count = _greedy_decode_fast(adj_bits, scores, n)
            if count > best_count:
                best_count = count
                best_scores = scores.copy()
                improved = True
            else:
                scores[v] = orig_score
            
            # Try demoting vertex v to the bottom
            scores[v] = np.min(scores) - 1.0
            count = _greedy_decode_fast(adj_bits, scores, n)
            if count > best_count:
                best_count = count
                best_scores = scores.copy()
                improved = True
            else:
                scores[v] = orig_score
        
        if not improved:
            break
    
    return best_scores
