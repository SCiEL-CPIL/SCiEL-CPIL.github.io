"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """QUBO-based Simulated Annealing + WalkSAT for Max-Cut with fast O(n) gain computation."""
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline guard: 150s wall-clock limit, return best with safety margin
    deadline = time.time() + 138.0
    rng = np.random.default_rng(42)
    
    def cut_size(labels):
        """Compute cut size: number of edges crossing the partition."""
        # For each vertex, count neighbors with different label
        # Sum and divide by 2 (each edge counted twice)
        cross = np.zeros(n, dtype=np.int32)
        for v in range(n):
            cross[v] = adj[v][labels != labels[v]].sum()
        return int(np.sum(cross) // 2)
    
    def flip_gain(labels, v):
        """O(deg(v)) computation of gain if we flip vertex v.
        Returns the change in cut size if we flip v."""
        # Current cross edges from v: neighbors with different label
        cur_cross = adj[v][labels != labels[v]].sum()
        # If we flip v, these become same-label (lost from cut)
        # And neighbors with same label become different (gained in cut)
        new_cross = adj[v][labels == labels[v]].sum()
        return int(new_cross - cur_cross)
    
    def all_flip_gains(labels):
        """Compute flip gains for all vertices in O(n*avg_degree)."""
        gains = np.zeros(n, dtype=np.int32)
        for v in range(n):
            gains[v] = flip_gain(labels, v)
        return gains
    
    def local_improve(labels):
        """Greedy local search: flip any vertex that improves the cut."""
        improved = True
        while improved:
            improved = False
            for v in range(n):
                if flip_gain(labels, v) > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels
    
    def two_flip_local_improve(labels):
        """Polish pass: try all pairs of simultaneous flips, accept if net gain > 0."""
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    return labels
                for j in range(i + 1, n):
                    if labels[i] == labels[j]:
                        continue
                    # Compute gain of flipping both i and j simultaneously
                    # This is more complex; approximate by sequential
                    g1 = flip_gain(labels, i)
                    labels[i] = 1 - labels[i]
                    g2 = flip_gain(labels, j)
                    labels[i] = 1 - labels[i]  # undo
                    # Approximate net gain
                    net_gain = g1 + g2
                    if net_gain > 0:
                        labels[i] = 1 - labels[i]
                        labels[j] = 1 - labels[j]
                        improved = True
        return labels
    
    def simulated_annealing(labels, steps=50000, t_start=1.0, t_end=0.01):
        """SA with 1-flip moves, using fast O(deg(v)) gain computation."""
        current_labels = labels.copy().astype(np.int8)
        current_cut = cut_size(current_labels)
        best_local = current_cut
        best_local_labels = current_labels.copy()
        
        # Adaptive cooling
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Choose a random vertex to flip
            v = rng.integers(0, n)
            gain = flip_gain(current_labels, v)
            
            # Accept if improving or with probability exp(-|gain|/T)
            if gain >= 0:
                current_labels[v] = 1 - current_labels[v]
                current_cut += gain
            else:
                # Temperature decreases geometrically
                t = t_start * (t_end / t_start) ** (step / max(steps - 1, 1))
                if rng.random() < np.exp(gain / max(t, 1e-10)):
                    current_labels[v] = 1 - current_labels[v]
                    current_cut += gain
            
            if current_cut > best_local:
                best_local = current_cut
                best_local_labels = current_labels.copy()
        
        return best_local_labels, best_local
    
    def walksat(labels, steps=5000):
        """WalkSAT-style: flip variable biased by unsatisfied clauses (uncut edges)."""
        current_cut = cut_size(labels)
        best_local = current_cut
        best_local_labels = labels.copy()
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Count unsatisfied clauses per vertex (uncut edges incident to v)
            unsat_count = np.zeros(n, dtype=np.int32)
            for v in range(n):
                unsat_count[v] = adj[v][labels == labels[v]].sum()
            
            if unsat_count.sum() == 0:
                break
            
            # Choose a variable to flip
            p = 0.7
            if rng.random() < p:
                total_unsat = unsat_count.sum()
                if total_unsat == 0:
                    v = rng.integers(0, n)
                else:
                    probs = unsat_count / total_unsat
                    v = rng.choice(n, p=probs)
            else:
                v = rng.integers(0, n)
            
            # Compute delta using fast gain computation
            delta = flip_gain(labels, v)
            
            # Flip the variable
            labels[v] = 1 - labels[v]
            current_cut += delta
            
            if current_cut > best_local:
                best_local = current_cut
                best_local_labels = labels.copy()
        
        return best_local_labels, best_local
    
    # Multiple restarts with diverse initial solutions
    best_labels = None
    best_cut = -1
    
    # Initial solutions
    initial_solutions = []
    initial_solutions.append(np.arange(n) % 2)
    initial_solutions.append(rng.integers(0, 2, n).astype(np.int8))
    initial_solutions.append(rng.integers(0, 2, n).astype(np.int8))
    degrees = adj.sum(axis=1)
    initial_solutions.append((degrees > degrees.mean()).astype(np.int8))
    
    # Spectral initialization
    try:
        from scipy.linalg import eigh
        L = np.diag(degrees.astype(np.float64)) - adj.astype(np.float64)
        vals, vecs = eigh(L)
        # Use second eigenvector (Fiedler vector)
        v = vecs[:, 1]
        spectral_labels = (v > v.mean()).astype(np.int8)
        initial_solutions.append(spectral_labels)
    except:
        pass
    
    for init_labels in initial_solutions:
        if time.time() > deadline:
            break
        
        labels = init_labels.copy().astype(np.int8)
        labels = local_improve(labels)
        cur_cut = cut_size(labels)
        
        if cur_cut > best_cut:
            best_cut = cur_cut
            best_labels = labels.copy()
        
        # WalkSAT phase
        ws_labels, ws_cut = walksat(labels, steps=3000)
        if ws_cut > best_cut:
            best_cut = ws_cut
            best_labels = ws_labels.copy()
        
        # SA phase to escape local minima
        sa_labels, sa_cut = simulated_annealing(ws_labels, steps=30000, t_start=1.0, t_end=0.01)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        # Final local improvement
        if time.time() < deadline:
            polished = local_improve(sa_labels)
            pol_cut = cut_size(polished)
            if pol_cut > best_cut:
                best_cut = pol_cut
                best_labels = polished
    
    # Final high-precision polish pass on the best solution
    if best_labels is not None and time.time() < deadline - 5:
        final_polished = two_flip_local_improve(best_labels.copy())
        final_cut = cut_size(final_polished)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_polished
        # One more greedy pass
        final_polished2 = local_improve(best_labels.copy())
        final_cut2 = cut_size(final_polished2)
        if final_cut2 > best_cut:
            best_cut = final_cut2
            best_labels = final_polished2
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
