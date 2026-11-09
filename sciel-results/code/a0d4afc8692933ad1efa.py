import numpy as np
import time
import random
from scipy.linalg import eigh

def _spectral_init(adj):
    """
    Use spectral method to get an initial partition.
    """
    n = len(adj)
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj
    
    try:
        eigvals, eigvecs = eigh(L)
        fiedler = eigvecs[:, 1]
        labels = (fiedler > 0).astype(np.int32)
        return labels
    except Exception:
        return np.random.randint(0, 2, n).astype(np.int32)

def _greedy_1flip(adj, labels, neighbors, time_limit, start_time):
    """
    Fast greedy local search: repeatedly flip vertices that improve the cut.
    """
    n = len(labels)
    labels = labels.copy()
    x = 2 * labels - 1  # -1 or 1
    
    # S[v] = sum of x[u] for u in neighbors(v)
    S = np.zeros(n)
    for v in range(n):
        S[v] = np.dot(adj[v], x)
        
    improved = True
    while improved:
        if time.time() - start_time > time_limit - 5:
            break
        improved = False
        for v in range(n):
            # Gain for flipping v: 2 * x[v] * S[v]
            gain = 2 * x[v] * S[v]
            if gain > 0:
                # Flip v
                x[v] = -x[v]
                labels[v] = 1 - labels[v]
                improved = True
                # Update S for neighbors of v
                # S[u] includes x[v]. x[v] changed by -2*x[v]_old (which is 2*x[v]_new)
                # Actually, x[v] became -x[v]_old.
                # Change in x[v] is -2*x[v]_old.
                # So S[u] += -2 * x[v]_old
                delta = -2 * (-x[v]) # x[v] is new, so old was -x[v]. Wait.
                # Let old_x = x_old. new_x = -x_old.
                # Change = new_x - old_x = -2 * old_x.
                # We stored new_x in x[v]. So old_x = -x[v].
                # Change = -2 * (-x[v]) = 2 * x[v].
                # So S[u] += 2 * x[v] (where x[v] is the NEW value)
                for u in neighbors[v]:
                    S[u] += 2 * x[v]
                # S[v] itself does not change because it depends on neighbors, not self.
                # But wait, does S[v] depend on x[v]? No, S[v] = sum_{u in N(v)} x[u].
                # So S[v] is unchanged.
                
    return labels

def _tabu_search(adj, labels, neighbors, time_limit, start_time, tabu_size=10, patience=500):
    """
    Cerny Tabu Search.
    """
    n = len(labels)
    labels = labels.copy()
    x = 2 * labels - 1
    
    # Initialize S
    S = np.zeros(n)
    for v in range(n):
        S[v] = np.dot(adj[v], x)
        
    # Current cut
    def calc_cut(x):
        return 0.5 * (np.sum(adj * (1 - np.outer(x, x)))) # This is O(N^2) each time, too slow.
        # Better: Cut = 0.5 * (sum_edges - sum_alignment)
        # Alignment = sum_{i,j} A[i,j] x[i] x[j]
        # We can maintain alignment.
    
    # Let's maintain alignment = sum_{i<j} 2*A[i,j]*x[i]*x[j] ? No.
    # Alignment A = x^T A x.
    # Cut = 0.5 * (sum(A) - x^T A x).
    # Let's just compute cut once at start and update incrementally?
    # Gain of flipping v is 2 * x[v] * S[v].
    # So new_cut = old_cut + gain.
    
    current_cut = 0.5 * (adj @ (1 - (2*labels - 1) * (2*labels - 1))).sum() # Just compute properly once
    # Actually, simpler:
    current_cut = np.sum(adj * (1 - labels)[:, None] * labels[None, :]) * 0.5 # Wait, labels are 0/1.
    # If labels 0/1:
    # Edge (i,j) crosses if labels[i] != labels[j].
    # cut = sum_{i,j} A[i,j] * (labels[i] != labels[j]) / 2
    # (labels[i] != labels[j]) is 1 if different, 0 if same.
    # This is 1 - (labels[i] == labels[j]).
    # labels[i] == labels[j] is 1 if both 0 or both 1.
    # Let's stick to x formulation for gain, but compute cut from x.
    # Cut = 0.5 * (sum_{i,j} A[i,j] - sum_{i,j} A[i,j] x[i] x[j])
    # sum_{i,j} A[i,j] is constant (2 * E).
    # Alignment = x^T A x.
    
    total_weight = adj.sum()
    alignment = x @ (adj @ x)
    current_cut = 0.5 * (total_weight - alignment)
    
    best_cut = current_cut
    best_labels = labels.copy()
    
    tabu_list = [-1] * tabu_size
    tabu_idx = 0
    
    no_improve_count = 0
    
    while True:
        if time.time() - start_time > time_limit - 5:
            break
        
        # Calculate gains for all v
        gains = 2 * x * S
        
        # Find best move
        # We want max gain.
        # If best is tabu, we might still take it if it improves best_cut (aspiration)
        # Otherwise, find best non-tabu.
        
        # Sort indices by gain descending
        # N=100, so sorting is fast.
        idxs = np.argsort(gains)[::-1]
        
        best_v = -1
        best_gain = -np.inf
        
        for v in idxs:
            if tabu_list[tabu_idx % tabu_size] == v:
                # Check aspiration
                new_cut = current_cut + gains[v]
                if new_cut > best_cut:
                    best_v = v
                    best_gain = gains[v]
                    break
            else:
                best_v = v
                best_gain = gains[v]
                break
        
        if best_v == -1:
            # No move found? Should not happen unless all tabu and no aspiration.
            # In that case, break or pick random.
            break
            
        # Perform move
        old_x_v = x[best_v]
        x[best_v] = -old_x_v
        labels[best_v] = 1 - labels[best_v]
        
        # Update alignment and cut
        # alignment changes by -2 * old_x_v * S[best_v] ?
        # Let's derive:
        # alignment = sum_{i,j} A[i,j] x[i] x[j]
        # Change in x[best_v] is delta = -2 * old_x_v.
        # New alignment = old_alignment + 2 * delta * S[best_v] + A[best_v, best_v] * delta^2
        # A[best_v, best_v] is 0.
        # So delta_alignment = 2 * (-2 * old_x_v) * S[best_v] = -4 * old_x_v * S[best_v].
        # Gain in cut = - delta_alignment / 2 = 2 * old_x_v * S[best_v].
        # This matches gains[best_v] = 2 * old_x_v * S[best_v].
        
        current_cut += best_gain
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > patience:
            break
            
        # Update S for neighbors
        # S[u] += -2 * old_x_v
        for u in neighbors[best_v]:
            S[u] -= 2 * old_x_v
            
        # Update tabu list
        tabu_list[tabu_idx % tabu_size] = best_v
        tabu_idx += 1
        
    return best_labels

def partition(adjacency):
    adj = np.array(adjacency, dtype=np.float64)
    n = len(adj)
    
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Precompute neighbors
    neighbors = []
    for i in range(n):
        neighbors.append(np.where(adj[i] != 0)[0])
    
    best_labels = None
    best_cut = -1
    
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Initialization
        if restart % 3 == 0:
            labels = _spectral_init(adj)
        elif restart % 3 == 1:
            labels = np.random.randint(0, 2, n).astype(np.int32)
        else:
            labels = np.arange(n) % 2
        
        # Phase 1: Greedy
        labels = _greedy_1flip(adj, labels, neighbors, time_limit, start_time)
        
        # Phase 2: Tabu
        labels = _tabu_search(adj, labels, neighbors, time_limit, start_time)
        
        # Evaluate
        cut = 0.5 * (adj @ (1 - labels)).sum()
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
    if best_labels is None:
        best_labels = np.arange(n) % 2
        
    return best_labels
