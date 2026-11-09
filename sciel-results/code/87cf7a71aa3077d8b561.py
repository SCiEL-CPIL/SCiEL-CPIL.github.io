"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # Edge (i,j) crosses if labels[i] != labels[j]
    # Vectorized computation
    mask = (labels[:, None] != labels[None, :]) & adj
    return int(mask.sum()) // 2


def _greedy_improve(adj, labels, deadline):
    """Greedy local improvement: flip vertices if it increases cut size."""
    n = len(labels)
    adj = adj.astype(np.int8)  # for faster summing
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for v in range(n):
            same_count = int(adj[v][labels == labels[v]].sum())
            other_count = int(adj[v][labels != labels[v]].sum())
            if same_count > other_count:
                labels[v] = 1 - labels[v]
                improved = True
    return labels


def _random_labels(n, seed):
    """Generate random 0/1 labels."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=n).astype(np.int8)


def _alternating_labels(n):
    """Generate alternating 0/1 labels."""
    return (np.arange(n) % 2).astype(np.int8)


def _biased_random_labels(n, seed, bias=0.5):
    """Generate random 0/1 labels with a bias."""
    rng = np.random.RandomState(seed)
    probs = np.full(n, bias)
    return (rng.random(n) < probs).astype(np.int8)


def _perturbed_alternating(n, seed, perturb_rate=0.1):
    """Generate alternating labels with random perturbations."""
    labels = (np.arange(n) % 2).astype(np.int8)
    rng = np.random.RandomState(seed)
    flip_mask = rng.random(n) < perturb_rate
    labels[flip_mask] = 1 - labels[flip_mask]
    return labels


def partition(adjacency):
    n = len(adjacency)
    adj = np.array(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Generate diverse initializations
    initializations = []
    
    # Alternating
    initializations.append(_alternating_labels(n))
    
    # Multiple random starts with different seeds
    for seed in range(10):
        initializations.append(_random_labels(n, seed))
    
    # Biased random starts
    for seed in range(5):
        initializations.append(_biased_random_labels(n, seed, bias=0.3))
        initializations.append(_biased_random_labels(n, seed, bias=0.7))
    
    # Perturbed alternating
    for seed in range(5):
        initializations.append(_perturbed_alternating(n, seed, perturb_rate=0.05))
        initializations.append(_perturbed_alternating(n, seed, perturb_rate=0.1))
        initializations.append(_perturbed_alternating(n, seed, perturb_rate=0.2))
    
    # Try each initialization with greedy improvement
    for init_labels in initializations:
        if time.time() > deadline:
            break
        
        # Make a copy to avoid modifying the original
        labels = init_labels.copy()
        
        # Greedy improvement
        labels = _greedy_improve(adj, labels, deadline)
        
        # Compute cut size
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # If we didn't find any valid solution (shouldn't happen), use alternating
    if best_labels is None:
        best_labels = _alternating_labels(n)
    
    return best_labels
# EVOLVE-BLOCK-END
