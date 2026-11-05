"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_polish(labels, adj, n, max_passes=50, deadline=None):
    """Vectorized greedy local search (flip if it improves cut)."""
    labels = labels.copy()
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        # For each vertex, compute gain of flipping
        # gain[v] = (# neighbors opposite) - (# neighbors same)
        # If gain[v] > 0, flipping improves cut.
        # neighbors_same[v] = sum over u where labels[u]==labels[v] of adj[v,u]
        # neighbors_opp[v] = sum over u where labels[u]!=labels[v] of adj[v,u]
        # total_degree[v] = neighbors_same[v] + neighbors_opp[v]
        # gain[v] = neighbors_opp[v] - neighbors_same[v] = 2*neighbors_opp[v] - total_degree[v]
        
        # Vectorized: for each v, neighbors_opp[v] = dot(adj[v], 1 - 2*labels + 1) ... 
        # Actually: labels are 0/1. Let s = 2*labels - 1 (spin +1/-1).
        # neighbors_opp[v] = sum_u adj[v,u] * (1 if s[u] != s[v] else 0)
        #                   = sum_u adj[v,u] * (1 - s[u]*s[v])/2
        #                   = (deg[v] - s[v] * sum_u adj[v,u]*s[u]) / 2
        # gain[v] = 2*neighbors_opp[v] - deg[v] = deg[v] - s[v]*A@ s - deg[v] = -s[v]*(A@ s)[v]
        # So flip v if -s[v]*(A@ s)[v] > 0, i.e., s[v]*(A@ s)[v] < 0.
        
        s = 2 * labels - 1  # +1 / -1
        As = adj @ s
        # Flip vertices where s[v] * As[v] < 0
        flip_mask = (s * As) < 0
        if not flip_mask.any():
            break
        s[flip_mask] = -s[flip_mask]
        labels = (s + 1) // 2
    return labels


def _simulated_annealing(adj, n, start_labels, t_start=1.0, t_end=0.01, n_iter=500, 
                         deadline=None, rng=None):
    """Simulated annealing for MaxCut."""
    if rng is None:
        rng = np.random.default_rng()
    
    labels = start_labels.copy()
    s = 2 * labels - 1  # spins
    # Precompute degrees
    deg = adj.sum(axis=1)
    
    # Current cut size
    # cut = 0.5 * sum_{i,j} adj[i,j] * (1 - s[i]*s[j])/2 ... 
    # Actually cut = sum_{i<j, adj[i,j]=1, s[i]!=s[j]} 1
    # = 0.25 * sum_{i,j} adj[i,j]*(1 - s[i]*s[j]) for i!=j... 
    # Simpler: cut = 0.5 * (sum_i s[i]*(A@ s)[i] is negative part)... 
    # Let's just compute: for each edge (i,j) with adj[i,j]=1, contributes 1 if s[i]!=s[j].
    # cut = 0.5 * sum_{i,j} adj[i,j] * (1 - s[i]*s[j]) / 2 * 2 ... 
    # Actually: sum_{i,j} adj[i,j]*(1-s[i]*s[j]) counts each edge twice (once for i->j, once j->i).
    # For an edge with s[i]!=s[j]: (1-(-1))=2, counted twice => 4.
    # For an edge with s[i]==s[j]: (1-1)=0.
    # So cut = 0.25 * sum_{i,j} adj[i,j]*(1-s[i]*s[j])
    
    def cut_value(sp):
        return 0.25 * np.sum(adj * (1 - np.outer(sp, sp)))
    
    cur_cut = cut_value(s)
    best_cut = cur_cut
    best_s = s.copy()
    
    # Cooling schedule
    coolings = np.linspace(t_start, t_end, n_iter)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        T = coolings[it]
        
        # Pick a random vertex to flip
        v = rng.integers(n)
        old_s_v = s[v]
        As_v = (adj @ s)[v]
        # Gain in cut if we flip v:
        # New contribution of v to sum: -old_s_v * As_v becomes +old_s_v * As_v
        # Delta in sum_{i,j} adj*(1-s_i*s_j) for terms involving v:
        # Terms (v,u) and (u,v): each changes by -2*old_s_v*s[u] -> wait
        # Let's use: cut = 0.25 * sum_{i,j} adj[i,j]*(1-s[i]*s[j])
        # d(cut)/d(s[v]) flip: 
        # Before: terms with v: sum_u adj[v,u]*(1-s[v]*s[u]) + sum_u adj[u,v]*(1-s[u]*s[v])
        #   = 2 * sum_u adj[v,u]*(1-s[v]*s[u]) (symmetric)
        # After flip s[v]->-s[v]:
        #   = 2 * sum_u adj[v,u]*(1+ s[v]*s[u])
        # Delta = 2 * sum_u adj[v,u]*(1+s[v]*s[u] - 1 + s[v]*s[u]) = 2*2*s[v]*sum_u adj[v,u]*s[u] = 4*s[v]*As_v
        # Delta_cut = 0.25 * 4 * s[v] * As_v = s[v] * As_v
        
        delta = old_s_v * As_v
        if delta >= 0 or rng.random() < np.exp(delta / T):
            s[v] = -old_s_v
            new_cut = cur_cut + delta
            cur_cut = new_cut
            if cur_cut > best_cut:
                best_cut = cur_cut
                best_s = s.copy()
    
    best_labels = (best_s + 1) // 2
    return best_labels, best_cut


def _quantum_annealing(adj, n, deadline=None, rng=None):
    """
    Quantum-inspired annealing: use a transverse field to allow tunneling.
    We maintain a 'soft' spin vector and periodically project to hard spins.
    This is a heuristic inspired by QAOA/QA but implemented classically.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Initialize with random spins
    s = rng.choice([-1, 1], size=n)
    best_s = s.copy()
    
    # Compute current cut
    def cut_value(sp):
        return 0.25 * np.sum(adj * (1 - np.outer(sp, sp)))
    
    best_cut = cut_value(s)
    
    # Quantum annealing schedule: transverse field gamma(t) decreases
    # We use a soft spin approach: each vertex has a probability of being +1
    # p[i] = sigmoid(h[i]/T) where h is the effective field
    
    t_start = 1.0
    t_end = 0.001
    n_iter = 300
    coolings = np.linspace(t_start, t_end, n_iter)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        T = coolings[it]
        
        # Compute effective field for each vertex
        # h[v] = sum_u adj[v,u] * s[u] (the Ising field)
        As = adj @ s
        
        # Soft update: with some probability, flip based on field and temperature
        # This mimics quantum tunneling by allowing flips even against the field
        # with probability depending on T
        
        # For each vertex, compute probability of being +1
        # P(s[v]=+1) = sigmoid(As[v] / T)
        logits = As / T
        # Clip to avoid overflow
        logits = np.clip(logits, -50, 50)
        p_pos = 1.0 / (1.0 + np.exp(-logits))
        
        # Sample new spins
        new_s = np.where(rng.random(n) < p_pos, 1, -1)
        
        # Compute new cut
        new_cut = cut_value(new_s)
        
        if new_cut >= best_cut:
            best_cut = new_cut
            best_s = new_s.copy()
        
        # Accept with Metropolis criterion
        delta = new_cut - cut_value(s)
        if delta >= 0 or rng.random() < np.exp(delta / T):
            s = new_s
    
    best_labels = (best_s + 1) // 2
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Deadline: leave 10s margin from 120s limit
    deadline = time.time() + 110.0
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different strategies
    rng = np.random.default_rng(42)
    
    # Strategy 1: Greedy from alternating start
    labels_alt = np.arange(n) % 2
    polished = _greedy_polish(labels_alt, adj, n, max_passes=50, deadline=deadline)
    cut_val = 0.25 * np.sum(adj * (1 - np.outer(2*polished-1, 2*polished-1)))
    if cut_val > best_cut:
        best_cut = cut_val
        best_labels = polished
    
    # Strategy 2: SA from random starts
    for restart in range(5):
        if time.time() > deadline:
            break
        start = rng.integers(2, size=n)
        sa_labels, sa_cut = _simulated_annealing(
            adj, n, start, t_start=1.0, t_end=0.005, n_iter=800,
            deadline=deadline, rng=rng
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
        # Polish the SA result
        polished = _greedy_polish(sa_labels, adj, n, max_passes=30, deadline=deadline)
        cut_val = 0.25 * np.sum(adj * (1 - np.outer(2*polished-1, 2*polished-1)))
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = polished
    
    # Strategy 3: Quantum-inspired annealing
    for restart in range(3):
        if time.time() > deadline:
            break
        qa_labels, qa_cut = _quantum_annealing(adj, n, deadline=deadline, rng=rng)
        if qa_cut > best_cut:
            best_cut = qa_cut
            best_labels = qa_labels
        # Polish
        polished = _greedy_polish(qa_labels, adj, n, max_passes=30, deadline=deadline)
        cut_val = 0.25 * np.sum(adj * (1 - np.outer(2*polished-1, 2*polished-1)))
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = polished
    
    # Final polish on best result
    if best_labels is not None:
        best_labels = _greedy_polish(best_labels, adj, n, max_passes=100, deadline=deadline)
    
    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
