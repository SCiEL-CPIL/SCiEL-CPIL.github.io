"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Safety margin before the 150 s wall-clock kill.
DEADLINE_MARGIN = 5.0


def _cut_size(labels, adj):
    """Compute the cut size (number of crossing edges) for a labeling."""
    diff = labels[None, :] != labels[:, None]
    return int(np.sum(adj & diff))


def _vectorized_gains(labels, adj, Ax):
    """
    Compute flip gains for all vertices simultaneously.
    Ax = adj @ labels (vector of neighbor label sums).
    For vertex v with label l:
      same_side_neighbors = Ax[v] - l * Ax[v] ... wait, let me think.
    
    Actually: if labels are 0/1:
      Ax[v] = sum of labels of neighbors = (# neighbors with label 1)
      deg[v] = total degree
      same_side = if labels[v]==1: Ax[v], else: deg[v] - Ax[v]
      other_side = deg[v] - same_side
      gain = same_side - other_side
    
    Let's use a unified formula:
      If labels[v] = 1: same = Ax[v], other = deg[v] - Ax[v], gain = Ax[v] - (deg[v] - Ax[v]) = 2*Ax[v] - deg[v]
      If labels[v] = 0: same = deg[v] - Ax[v], other = Ax[v], gain = (deg[v] - Ax[v]) - Ax[v] = deg[v] - 2*Ax[v]
    
    So: gain[v] = (2*labels[v] - 1) * (2*Ax[v] - deg[v])
    Wait, let's verify:
      labels[v]=1: (2*1-1)*(2*Ax[v]-deg[v]) = 1*(2*Ax[v]-deg[v]) = 2*Ax[v]-deg[v] ✓
      labels[v]=0: (2*0-1)*(2*Ax[v]-deg[v]) = -1*(2*Ax[v]-deg[v]) = deg[v]-2*Ax[v] ✓
    """
    deg = adj.sum(axis=1)
    sign = 2.0 * labels.astype(np.float64) - 1.0
    return sign * (2.0 * Ax - deg)


def _update_Ax(v, labels, adj, Ax):
    """Update Ax when vertex v is flipped. Ax = adj @ labels."""
    # Flipping v changes labels[v] by (1 - 2*labels[v])
    # This affects Ax[u] for all neighbors u of v: Ax[u] += (1 - 2*old_label) * adj[u,v]
    # Since adj is symmetric: adj[u,v] = adj[v,u]
    delta = 1.0 - 2.0 * labels[v]
    Ax[adj[v, :]] += delta
    return Ax


def _greedy_polish(labels, adj, deadline):
    """
    Fast greedy local search using vectorized gain computation.
    """
    n = len(labels)
    labels = labels.copy().astype(np.float64)
    Ax = adj.astype(np.float64) @ labels
    best_cut = _cut_size(labels.astype(int), adj)
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        gains = _vectorized_gains(labels, adj, Ax)
        best_gain = np.max(gains)
        if best_gain > 1e-10:
            best_v = int(np.argmax(gains))
            _update_Ax(best_v, labels, adj, Ax)
            labels[best_v] = 1.0 - labels[best_v]
            best_cut += int(best_gain)
            improved = True
    
    return labels.astype(int), best_cut


def _simulated_annealing(labels, adj, deadline, t0=10.0, t1=0.1, alpha=0.99):
    """
    Simulated annealing for MaxCut.
    labels: 0/1 array (will be converted to float internally)
    Returns improved labels and cut size.
    """
    n = len(labels)
    labels = labels.copy().astype(np.float64)
    Ax = adj.astype(np.float64) @ labels
    current_cut = _cut_size(labels.astype(int), adj)
    best_cut = current_cut
    best_labels = labels.copy()
    
    t = t0
    rng = np.random.default_rng()
    
    while t > t1:
        if time.time() > deadline:
            break
        
        # Pick a random vertex to flip
        v = rng.integers(n)
        
        # Compute gain for flipping v
        gains = _vectorized_gains(labels, adj, Ax)
        gain = gains[v]
        
        # Accept if gain > 0, or with probability exp(gain/t) if gain < 0
        if gain >= 0 or rng.random() < np.exp(gain / t):
            _update_Ax(v, labels, adj, Ax)
            labels[v] = 1.0 - labels[v]
            current_cut += int(gain)
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        t *= alpha
    
    # Final greedy polish from best state
    best_labels_int, best_cut = _greedy_polish(best_labels.astype(int), adj, deadline)
    
    return best_labels_int, best_cut


def partition(adjacency):
    """
    Partition vertices into two sets to maximize the number of crossing edges.
    Uses simulated annealing with greedy polish and multiple restarts.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_labels = None
    best_cut = -1
    
    # Strategy 1: Start with alternating initialization + greedy polish
    labels_alt = (np.arange(n) % 2).astype(int)
    labels_alt, cut_alt = _greedy_polish(labels_alt, adj, deadline)
    best_labels = labels_alt
    best_cut = cut_alt
    
    # Strategy 2: Multiple SA restarts with random initialization
    # Each SA run is followed by greedy polish
    n_sa_restarts = 5
    for r in range(n_sa_restarts):
        if time.time() > deadline:
            break
        
        # Random initial labeling
        labels = np.random.randint(0, 2, size=n).astype(int)
        
        # Run simulated annealing
        labels_sa, cut_sa = _simulated_annealing(labels, adj, deadline, t0=5.0, t1=0.05, alpha=0.995)
        
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa.copy()
    
    # Strategy 3: Use remaining time for additional greedy polish from best solution
    if time.time() < deadline - 1.0:
        labels_final, cut_final = _greedy_polish(best_labels, adj, deadline)
        if cut_final > best_cut:
            best_cut = cut_final
            best_labels = labels_final
    
    return best_labels
# EVOLVE-BLOCK-END
