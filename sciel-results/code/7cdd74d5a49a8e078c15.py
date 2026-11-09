# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _flip_gain(adjacency, labels, v):
    """Compute the change in cut size if we flip vertex v."""
    same_count = adjacency[v][labels == labels[v]].sum()
    opp_count = adjacency[v][labels != labels[v]].sum()
    return int(same_count - opp_count)


def _two_flip_gain(adjacency, labels, u, v):
    """Compute the change in cut size if we flip both vertices u and v."""
    lu = labels[u]
    lv = labels[v]
    
    # If u and v are on the same side, flipping both changes the edge between them from non-cut to cut or vice versa
    # Actually, let's just compute the gain by considering each flip independently and the interaction
    
    # Gain from flipping u alone
    gain_u = _flip_gain(adjacency, labels, u)
    # Gain from flipping v alone
    gain_v = _flip_gain(adjacency, labels, v)
    
    # The interaction term: 
    # If u and v are connected, flipping both of them:
    # - If they were on the same side: the edge (u,v) was not in the cut. After flipping both, it's still not in the cut. No change to this edge.
    # - If they were on different sides: the edge (u,v) was in the cut. After flipping both, it's still in the cut. No change to this edge.
    # Wait, that's not right. Let's think carefully.
    
    # Let's just compute it directly for safety, since n=100 is small.
    # But for efficiency in a loop, the independent gains plus a correction is better.
    
    # Actually, the gain of flipping two vertices is:
    # sum of gains of flipping each one individually, MINUS 2 * (interaction)
    # The interaction is: if u and v are connected, and they are on the SAME side, then flipping both doesn't change the status of edge (u,v) (was non-cut, still non-cut). 
    # But wait, the gain of flipping u alone assumes v stays put.
    
    # Let's just use the direct formula for two flips.
    # The change in cut for edge (i, j) when flipping set S:
    # If i and j are both in S: no change (both flip, relative side stays same)
    # If i in S, j not in S: flips (was cut -> not cut, or not cut -> cut)
    # If i not in S, j in S: flips
    # If neither in S: no change
    
    # So for flipping {u, v}:
    # Edges from u to V\{u,v} flip their cut status
    # Edges from v to V\{u,v} flip their cut status
    # Edge (u,v): if it exists, its cut status does NOT change (both endpoints flip)
    
    # Gain = (number of edges from u to V\{u,v} that are currently NOT cut) 
    #       - (number of edges from u to V\{u,v} that are currently cut)
    #       + (number of edges from v to V\{u,v} that are currently NOT cut)
    #       - (number of edges from v to V\{u,v} that are currently cut)
    
    # This is equivalent to:
    # For u: (neighbors of u not in S) that are on same side as u - (neighbors of u not in S) that are on opposite side
    # But S = {u, v}. So neighbors of u not in S = neighbors of u excluding v.
    
    # Let's just compute it the safe way for n=100.
    labels_new = labels.copy()
    labels_new[u] = 1 - labels_new[u]
    labels_new[v] = 1 - labels_new[v]
    
    old_cut = _cut_size(adjacency, labels)
    new_cut = _cut_size(adjacency, labels_new)
    
    return new_cut - old_cut


def _local_search(adjacency, labels, deadline):
    """Iterative local search with running cut count and shuffled order."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        order = np.random.permutation(n)
        for v in order:
            if time.time() > deadline:
                break
            gain = _flip_gain(adjacency, labels, v)
            if gain > 0:
                labels[v] = 1 - labels[v]
                current_cut += gain
                improved = True
    
    return labels, current_cut


def _two_flip_local_search(adjacency, labels, deadline):
    """Local search that allows 2-flip moves to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        # First try single flips
        order = np.random.permutation(n)
        for v in order:
            if time.time() > deadline:
                break
            gain = _flip_gain(adjacency, labels, v)
            if gain > 0:
                labels[v] = 1 - labels[v]
                current_cut += gain
                improved = True
        
        # Then try 2-flip moves (sampled to keep it fast)
        if time.time() > deadline:
            break
            
        num_pairs_to_try = min(50, n * (n - 1) // 2)
        tried = set()
        for _ in range(num_pairs_to_try):
            if time.time() > deadline:
                break
            u, v = np.random.choice(n, size=2, replace=False)
            if u > v:
                u, v = v, u
            if (u, v) in tried:
                continue
            tried.add((u, v))
            
            gain = _two_flip_gain(adjacency, labels, u, v)
            if gain > 0:
                labels[u] = 1 - labels[u]
                labels[v] = 1 - labels[v]
                current_cut += gain
                improved = True
    
    return labels, current_cut


def _spectral_partition(adjacency, n):
    """Spectral partition using the Fiedler vector."""
    try:
        A = adjacency.astype(np.float64)
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        try:
            vals, vecs = np.linalg.eigh(L)
            idx = np.argsort(vals)
            fiedler_vec = vecs[:, idx[1]]
        except Exception:
            return np.arange(n) % 2
        
        labels = (fiedler_vec > 0).astype(np.int8)
        
        if labels.sum() == 0 or labels.sum() == n:
            median = np.median(fiedler_vec)
            labels = (fiedler_vec > median).astype(np.int8)
            if labels.sum() == 0 or labels.sum() == n:
                labels = np.arange(n) % 2
        
        return labels
    except Exception:
        return np.arange(n) % 2


def _greedy_partition(adjacency, n, deadline):
    """Greedy construction: assign vertices to maximize cut, ordered by degree."""
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


def _perturb(labels, n, num_flips, rng):
    """Perturb a solution by flipping a few random vertices."""
    perturbed = labels.copy()
    indices = rng.choice(n, size=min(num_flips, n), replace=False)
    perturbed[indices] = 1 - perturbed[indices]
    return perturbed


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin within 150s limit
    
    best_labels = None
    best_cut = -1
    
    def _update_best(labels, cut):
        nonlocal best_labels, best_cut
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Phase 1: Multiple diverse initializations
    initializations = []
    
    # 1. Spectral partition
    initializations.append(_spectral_partition(adjacency, n))
    
    # 2. Greedy partition (degree-ordered)
    initializations.append(_greedy_partition(adjacency, n, deadline))
    
    # 3. Alternating
    initializations.append(np.arange(n) % 2)
    
    # 4-12. Random initializations with different seeds
    for seed in range(9):
        initializations.append(_random_labels(n, 100 + seed))
    
    # Phase 2: Local search from each initialization
    for init_labels in initializations:
        if time.time() > deadline:
            break
        labels, cut = _local_search(adjacency, init_labels, deadline)
        _update_best(labels, cut)
    
    # Phase 3: 2-flip local search on the best solution
    if best_labels is not None:
        if time.time() < deadline - 10:
            labels, cut = _two_flip_local_search(adjacency, best_labels, deadline)
            _update_best(labels, cut)
    
    # Phase 4: Perturbation-based restarts from best solution
    rng = np.random.RandomState(42)
    num_perturbations = 50
    for i in range(num_perturbations):
        if time.time() > deadline:
            break
        # Increase perturbation size over time
        num_flips = 1 + (i % 5)
        perturbed = _perturb(best_labels, n, num_flips, rng)
        labels, cut = _local_search(adjacency, perturbed, deadline)
        _update_best(labels, cut)
        
        # Occasionally apply 2-flip search on the perturbed result
        if i % 3 == 0 and time.time() < deadline - 5:
            labels2, cut2 = _two_flip_local_search(adjacency, labels, deadline)
            _update_best(labels2, cut2)
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
