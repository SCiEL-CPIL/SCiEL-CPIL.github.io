"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _all_flip_gains(adjacency, labels):
    """Compute flip gains for all vertices at once."""
    same_mask = (labels[:, None] == labels[None, :])
    same_counts = (adjacency & same_mask).sum(axis=1)
    opp_counts = (adjacency & ~same_mask).sum(axis=1)
    gains = same_counts - opp_counts
    return gains.astype(np.int32)


def _local_search(adjacency, labels, deadline, max_rounds=200):
    """Iterative local search: flip vertices if it improves cut size."""
    labels = labels.copy()
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        gains = _all_flip_gains(adjacency, labels)
        positive = np.where(gains > 0)[0]
        if len(positive) == 0:
            break
        labels[positive] = 1 - labels[positive]
    
    return labels


def _spectral_partition(adjacency, n):
    """Use Fiedler vector for initial partition."""
    A = adjacency.astype(float)
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        fiedler = eigenvectors[:, 1]
        labels = (fiedler >= 0).astype(np.int8)
        return labels
    except:
        return np.random.randint(0, 2, size=n).astype(np.int8)


def _greedy_partition(adjacency, n, deadline):
    """Greedy construction: add vertices one by one, choosing label that maximizes cut."""
    labels = np.zeros(n, dtype=np.int8)
    degrees = adjacency.sum(axis=1)
    order = np.argsort(-degrees)
    
    for v in order:
        if time.time() > deadline:
            break
        cut_if_0 = adjacency[v][labels == 1].sum()
        cut_if_1 = adjacency[v][labels == 0].sum()
        labels[v] = 0 if cut_if_0 >= cut_if_1 else 1
    
    return labels


def _random_labels(n, seed):
    """Generate random 0/1 labels."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=n).astype(np.int8)


def _adaptive_sa(adjacency, labels, deadline, start_temp=30.0, min_temp=0.01,
                 cooling=0.998, diversify_threshold=50, max_steps=500000):
    """
    Adaptive Simulated Annealing with dynamic cooling.
    Uses vectorized gain computation and adapts temperature based on
    acceptance ratio to escape local optima effectively.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    temp = start_temp
    rng = np.random.RandomState(789)
    adj_int = adjacency.astype(np.int8)
    
    accepted_bad = 0
    total_moves = 0
    steps_since_improve = 0
    last_improve_cut = current_cut
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Pick a random vertex to flip
        v = rng.randint(n)
        
        # Compute gain for flipping v
        same_count = adj_int[v][labels == labels[v]].sum()
        opp_count = adj_int[v][labels != labels[v]].sum()
        gain = same_count - opp_count
        
        total_moves += 1
        
        if gain >= 0:
            # Accept improving or neutral moves
            labels[v] = 1 - labels[v]
            current_cut += gain
            steps_since_improve = 0
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
                last_improve_cut = current_cut
        else:
            # Accept worsening moves with probability exp(gain/temp)
            if rng.random() < np.exp(gain / temp):
                labels[v] = 1 - labels[v]
                current_cut += gain
                accepted_bad += 1
                steps_since_improve += 1
        
        # Adaptive cooling: adjust temperature based on acceptance rate
        # If too many bad moves are accepted, cool faster to converge
        # If stuck (no improvement), heat up slightly to explore
        if total_moves > 100:
            bad_ratio = accepted_bad / total_moves
            if bad_ratio > 0.5:
                # Too many bad moves accepted - cool faster
                temp *= cooling * 0.95
            elif bad_ratio < 0.1:
                # Very few bad moves - cool slower to explore more
                temp *= cooling * 1.02
            
            # If stuck for a while, occasionally reheat to escape
            if steps_since_improve > diversify_threshold and rng.random() < 0.05:
                temp = min(temp * 2.0, start_temp * 0.5)
                steps_since_improve = 0
        
        temp = max(temp, min_temp)
        
        # Periodically run local search to polish
        if step % 5000 == 4999 and time.time() < deadline - 5.0:
            polished = _local_search(adjacency, labels, deadline)
            polished_cut = _cut_size(adjacency, polished)
            if polished_cut > best_cut:
                best_cut = polished_cut
                best_labels = polished.copy()
                labels = polished.copy()
                current_cut = polished_cut
                last_improve_cut = polished_cut
                steps_since_improve = 0
    
    return best_labels, best_cut


def _walksat(adjacency, labels, deadline, max_iters=100000, clause_size=3):
    """
    WalkSAT-style stochastic local search.
    Randomly flip vertices, biased toward improving moves.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    rng = np.random.RandomState(321)
    adj_int = adjacency.astype(np.int8)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute gains for all vertices
        gains = _all_flip_gains(adjacency, labels)
        
        # Find best improving move
        best_gain = np.max(gains)
        
        if best_gain > 0:
            # Flip the best improving vertex
            candidates = np.where(gains == best_gain)[0]
            v = candidates[rng.randint(len(candidates))]
        else:
            # No improving move - pick randomly among vertices with highest gains
            top_k = min(clause_size, n)
            top_indices = np.argsort(gains)[-top_k:]
            v = top_indices[rng.randint(len(top_indices))]
        
        gain = gains[v]
        labels[v] = 1 - labels[v]
        current_cut += gain
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels, best_cut


def _multi_start_sa(adjacency, n, deadline):
    """
    Multi-start Simulated Annealing with diverse initializations.
    """
    start_time = time.time()
    remaining = deadline - start_time
    
    # Divide time among multiple SA runs
    num_starts = max(2, int(remaining / 20.0))  # ~20s per start
    num_starts = min(num_starts, 5)  # Cap at 5
    
    per_start_time = remaining / num_starts
    
    best_labels = None
    best_cut = -1
    
    # Different initializations for diversity
    inits = []
    
    # Spectral
    inits.append(_spectral_partition(adjacency, n))
    
    # Greedy
    inits.append(_greedy_partition(adjacency, n, deadline))
    
    # Alternating
    inits.append((np.arange(n) % 2).astype(np.int8))
    
    # Random
    for seed in range(5):
        inits.append(_random_labels(n, 500 + seed))
    
    # Shuffle and take first num_starts
    rng = np.random.RandomState(42)
    rng.shuffle(inits)
    inits = inits[:num_starts]
    
    for i, init_labels in enumerate(inits):
        if time.time() > deadline - 2.0:
            break
        
        # First do local search
        ls_labels = _local_search(adjacency, init_labels, deadline)
        ls_cut = _cut_size(adjacency, ls_labels)
        
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels.copy()
        
        # Then SA from the locally-optimal solution
        sa_deadline = deadline - 2.0
        sa_labels, sa_cut = _adaptive_sa(adjacency, ls_labels, sa_deadline)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    return best_labels, best_cut


def _final_polish(adjacency, labels, deadline):
    """Final high-precision polish: aggressive local search with multiple passes."""
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Multiple local search passes with occasional perturbations
    for _ in range(20):
        if time.time() > deadline:
            break
        
        # Local search
        labels = _local_search(adjacency, labels, deadline)
        current_cut = _cut_size(adjacency, labels)
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
        
        # Check if stuck
        gains = _all_flip_gains(adjacency, labels)
        if np.max(gains) <= 0:
            # Stuck at local optimum - perturb
            n = len(labels)
            rng = np.random.RandomState(int(time.time() * 1000) % 100000)
            num_flips = max(1, n // 50)
            flip_indices = rng.choice(n, size=num_flips, replace=False)
            labels[flip_indices] = 1 - labels[flip_indices]
            current_cut = _cut_size(adjacency, labels)
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Phase 1: Multi-start SA with diverse initializations
    best_labels, best_cut = _multi_start_sa(adjacency, n, deadline)
    
    # Phase 2: WalkSAT from best result
    if time.time() < deadline - 10.0:
        ws_labels, ws_cut = _walksat(adjacency, best_labels, deadline - 5.0)
        if ws_cut > best_cut:
            best_cut = ws_cut
            best_labels = ws_labels
    
    # Phase 3: Final polish
    if time.time() < deadline - 2.0:
        polish_labels, polish_cut = _final_polish(adjacency, best_labels, deadline)
        if polish_cut > best_cut:
            best_cut = polish_cut
            best_labels = polish_labels
    
    return best_labels
# EVOLVE-BLOCK-END
