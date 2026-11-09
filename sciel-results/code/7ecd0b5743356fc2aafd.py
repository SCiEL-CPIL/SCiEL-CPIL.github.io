"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size_fast(A_int, labels):
    """Compute cut size: number of edges crossing the partition."""
    # labels: 0/1 array of length n
    # An edge (i,j) is cut if labels[i] != labels[j]
    # Cut size = sum over all edges where labels differ
    # = (1/2) * sum_i sum_j A[i,j] * (labels[i] != labels[j])
    # = (1/2) * sum_i sum_j A[i,j] * (labels[i] - labels[j])^2  [since binary]
    # More efficiently: for each vertex i, count neighbors in opposite partition
    # cut = sum_i sum_j A[i,j] * (1 - labels[i]*labels[j] - (1-labels[i])*(1-labels[j]))
    # Simpler: cut = sum_i sum_j A[i,j] * (labels[i] XOR labels[j])
    # For binary labels: XOR = labels[i] + labels[j] - 2*labels[i]*labels[j]
    # cut = sum_i sum_j A[i,j]*(labels[i]+labels[j]) - 2*sum_i sum_j A[i,j]*labels[i]*labels[j]
    # = 2*sum_i labels[i]*deg[i] - 2*labels^T A labels
    # Wait, let me just do it directly with matrix ops
    l = labels.astype(np.float64)
    # cut = 0.5 * sum_{i,j} A[i,j] * |l[i] - l[j]|  but for binary this is:
    # For binary: |l[i]-l[j]| = l[i] + l[j] - 2*l[i]*l[j] when they differ, 0 when same
    # Actually for binary 0/1: l[i]+l[j]-2*l[i]*l[j] = 1 if different, 0 if same. Yes!
    # So cut = 0.5 * sum_{i,j} A[i,j] * (l[i]+l[j]-2*l[i]*l[j])
    # = 0.5 * (2*l^T D l - 2*l^T A l) where D is degree matrix... no
    # Let me just compute directly:
    # sum_i sum_j A[i,j]*(l[i]+l[j]-2*l[i]*l[j])
    # = sum_i l[i]*sum_j A[i,j] + sum_j l[j]*sum_i A[i,j] - 2*sum_i sum_j A[i,j]*l[i]*l[j]
    # = 2*l^T*d - 2*l^T*A*l
    # where d = A.sum(axis=1)
    d = A_int.sum(axis=1).astype(np.float64)
    cut = 2.0 * (l @ d) - 2.0 * (l @ (A_int @ l))
    return int(cut // 2)


def _spectral_partition(adjacency):
    """Use spectral method (Fiedler vector) to get initial partition."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return np.random.randint(0, 2, size=n)
    
    idx = np.argsort(eigvals)
    fiedler = eigvecs[:, idx[1]]
    
    labels = (fiedler >= 0).astype(np.int32)
    
    if labels.sum() == 0 or labels.sum() == n:
        labels = np.random.randint(0, 2, size=n)
    
    return labels


def _random_labels(n):
    """Generate random initial labels."""
    return np.random.randint(0, 2, size=n)


def _fast_local_search(A_int, labels, deadline, max_rounds=100):
    """Iterative local search using vectorized gain computation."""
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    
    # For each vertex v, the gain of flipping v:
    # gain[v] = (number of neighbors in opposite partition) - (number of neighbors in same partition)
    # If gain[v] > 0, flipping v increases cut size.
    # 
    # Let's define: for vertex v with label l[v]:
    #   same_count = sum_{u: l[u]==l[v]} A[v,u]
    #   opp_count = sum_{u: l[u]!=l[v]} A[v,u]
    #   gain = opp_count - same_count
    #   total_degree = same_count + opp_count
    #   gain = total_degree - 2*same_count
    #
    # same_count for v = sum_u A[v,u] * (l[u] == l[v])
    # = sum_u A[v,u] * (1 - |l[u] - l[v]|)  ... not quite
    # For binary: (l[u]==l[v]) = 1 - l[u] + l[v] - 2*l[u]*l[v] ... no
    # Simpler: (l[u]==l[v]) = 1 - (l[u] XOR l[v]) = 1 - l[u] - l[v] + 2*l[u]*l[v]
    # Hmm, let me just compute directly.
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        # Compute gains for all vertices
        # same_count[v] = sum_u A[v,u] * (l[u] == l[v])
        # For binary labels: (l[u]==l[v]) = 1 - l[u] - l[v] + 2*l[u]*l[v]
        # Actually simpler: (l[u]==l[v]) = 1 - (l[u]+l[v]-2*l[u]*l[v]) = 1 - l[u] - l[v] + 2*l[u]*l[v]
        # Wait: if l[u]=0,l[v]=0: 1-0-0+0=1 ✓
        # if l[u]=1,l[v]=1: 1-1-1+2=1 ✓
        # if l[u]=0,l[v]=1: 1-0-1+0=0 ✓
        # if l[u]=1,l[v]=0: 1-1-0+0=0 ✓
        # Good.
        
        l = labels.astype(np.float64)
        # same_indicator[u,v] = 1 - l[u] - l[v] + 2*l[u]*l[v]
        # same_count[v] = sum_u A[v,u] * (1 - l[u] - l[v] + 2*l[u]*l[v])
        # = sum_u A[v,u] - l[v]*sum_u A[v,u] - sum_u A[v,u]*l[u] + 2*l[v]*sum_u A[v,u]*l[u]
        # = deg[v] - l[v]*deg[v] - (A@l)[v] + 2*l[v]*(A@l)[v]
        # = deg[v]*(1 - l[v]) + (A@l)[v]*(2*l[v] - 1)
        
        deg = A_int.sum(axis=1).astype(np.float64)
        Al = A_int @ l  # (A @ l)[v] = sum_u A[v,u]*l[u]
        
        # same_count[v] = deg[v]*(1-l[v]) + Al[v]*(2*l[v]-1)
        same_count = deg * (1.0 - l) + Al * (2.0 * l - 1.0)
        
        # gain[v] = deg[v] - 2*same_count[v]
        gain = deg - 2.0 * same_count
        
        # Find vertex with maximum positive gain
        max_gain = gain.max()
        if max_gain <= 0:
            break
        
        # Flip the vertex with max gain (or a random one among top gainers for diversity)
        v = int(np.argmax(gain))
        labels[v] = 1 - labels[v]
        improved = True
        
        if not improved:
            break
    
    return labels


def _walksat_search(A_int, labels, deadline, max_iterations=20000, p_flip=0.25):
    """WalkSAT-style stochastic local search for Max-Cut.
    
    In Max-Cut as Max-2-SAT:
    - Each edge (i,j) is a clause: (l[i] XOR l[j])
    - "Unsatisfied clause" = uncut edge (both endpoints in same partition)
    - WalkSAT: pick a variable involved in unsatisfied clauses, flip it
      with probability biased by whether it satisfies or breaks clauses.
    
    Here we use:
    - Select vertex weighted by number of uncut incident edges (unsatisfied clauses)
    - Flip with probability based on net gain
    """
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    
    current_cut = _cut_size_fast(A_int, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Precompute degree
    deg = A_int.sum(axis=1).astype(np.int32)
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Compute uncut edges per vertex: for vertex v, uncut[v] = number of neighbors in same partition
        l = labels.astype(np.float64)
        # same_count[v] = sum_u A[v,u] * (l[u]==l[v])
        # Using the formula: same_count[v] = deg[v]*(1-l[v]) + (A@l)[v]*(2*l[v]-1)
        Al = A_int @ l
        same_count = deg.astype(np.float64) * (1.0 - l) + Al * (2.0 * l - 1.0)
        uncut = same_count  # number of uncut incident edges
        
        # Total uncut edges (each counted twice)
        total_uncut = uncut.sum()
        
        if total_uncut < 1e-9:
            break  # Perfect cut found
        
        # Select vertex: with probability p_flip, pick randomly from vertices with uncut edges
        # Otherwise, pick the vertex with the most uncut edges (greedy)
        # For WalkSAT: pick a random unsatisfied clause, then pick a variable from it
        # Here: pick a random uncut edge, then pick one of its endpoints
        if np.random.random() < p_flip:
            # Random walk: pick a random vertex weighted by uncut
            uncut_weights = uncut / total_uncut
            v = np.random.choice(n, p=uncut_weights)
        else:
            # Greedy: pick vertex with most uncut edges
            v = int(np.argmax(uncut))
        
        # Compute gain of flipping v
        lv = labels[v]
        same_v = same_count[v]
        opp_v = deg[v] - same_v
        gain = opp_v - same_v  # = deg[v] - 2*same_v
        
        # Acceptance: always accept if gain > 0
        # If gain < 0, accept with some probability (to escape local optima)
        if gain >= 0:
            labels[v] = 1 - lv
            current_cut += int(gain)
        else:
            # Accept worsening move with probability based on how bad it is
            # Use a simple heuristic: accept with prob 0.3 if gain >= -2, else 0.1
            if gain >= -2:
                accept_prob = 0.3
            else:
                accept_prob = 0.1
            if np.random.random() < accept_prob:
                labels[v] = 1 - lv
                current_cut += int(gain)
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
        
        # Periodic local search polish
        if it % 2000 == 1999 and time.time() < deadline - 5.0:
            polished = _fast_local_search(A_int, labels, deadline - 2.0, max_rounds=20)
            polished_cut = _cut_size_fast(A_int, polished)
            if polished_cut > best_cut:
                best_cut = polished_cut
                best_labels = polished.copy()
            labels = polished
            current_cut = polished_cut
    
    return best_labels, best_cut


def partition(adjacency):
    """Main partition function using spectral init, fast local search, and WalkSAT."""
    n = len(adjacency)
    
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    A_int = adjacency.astype(np.int32)
    
    # Generate multiple initial candidates
    candidates = []
    
    # 1. Spectral partition
    spec_labels = _spectral_partition(adjacency)
    candidates.append(spec_labels)
    
    # 2. A few random partitions
    for _ in range(5):
        candidates.append(_random_labels(n))
    
    # 3. Alternating
    candidates.append(np.arange(n) % 2)
    
    # Run fast local search on each candidate
    best_labels = None
    best_cut = -1
    
    for cand in candidates:
        if time.time() > deadline - 30.0:
            break
        
        ls_labels = _fast_local_search(A_int, cand, deadline - 30.0, max_rounds=50)
        ls_cut = _cut_size_fast(A_int, ls_labels)
        
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels.copy()
    
    # If we have time, run WalkSAT from the best local search result
    if time.time() < deadline - 10.0 and best_labels is not None:
        ws_labels, ws_cut = _walksat_search(A_int, best_labels, deadline - 5.0, 
                                             max_iterations=30000, p_flip=0.3)
        if ws_cut > best_cut:
            best_cut = ws_cut
            best_labels = ws_labels
        
        # Final local search polish
        if time.time() < deadline - 3.0:
            final_labels = _fast_local_search(A_int, best_labels, deadline - 1.0, max_rounds=30)
            final_cut = _cut_size_fast(A_int, final_labels)
            if final_cut > best_cut:
                best_cut = final_cut
                best_labels = final_labels
    
    # Ensure valid output
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(np.int32)
# EVOLVE-BLOCK-END
