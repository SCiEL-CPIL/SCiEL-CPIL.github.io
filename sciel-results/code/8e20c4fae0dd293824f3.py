"""Rewritten program for kissing number in dimension 11."""
import time
import numpy as np
import itertools
from scipy.sparse import csr_matrix

D = 11
DEADLINE = 145.0  # seconds, safety margin under 150s

def _gen_candidates(max_w=5, max_nz=5):
    """Generate candidate nonzero integer vectors with bounded weight and entries in {-2,-1,0,1,2}."""
    seen = set()
    vecs = []
    # weight 1..max_nz, entries in {-1,0,1} first (classic), then allow ±2 for small weights
    for w in range(1, max_nz + 1):
        for positions in itertools.combinations(range(D), w):
            # For each position, try entries from a small set
            # To keep it manageable, for w<=3 allow ±2, else ±1
            if w <= 3:
                vals = [-2, -1, 1, 2]
            else:
                vals = [-1, 1]
            for combo in itertools.product(vals, repeat=w):
                v = [0] * D
                for p, val in zip(positions, combo):
                    v[p] = val
                # canonical: ensure not all zero (already true since w>=1)
                # dedup via tuple
                t = tuple(v)
                if t in seen:
                    continue
                seen.add(t)
                vecs.append(v)
    return np.array(vecs, dtype=np.int32)

def _norms_sq(A):
    return np.sum(A * A, axis=1)

def _build_conflict_graph(A, norms_sq, R2):
    """
    Build adjacency where edge i-j means distance^2 < R2.
    distance^2 = ||a||^2 + ||b||^2 - 2 a.b
    Conflict if ||a||^2 + ||b||^2 - 2 a.b < R2
    """
    n = len(A)
    # Compute dot product matrix
    dots = A @ A.T
    # dist2[i,j] = ns[i] + ns[j] - 2*dots[i,j]
    dist2 = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    # Conflict: dist2 < R2, excluding self
    conflict = (dist2 < R2)
    np.fill_diagonal(conflict, False)
    # Convert to sparse
    # Use only upper triangle to save memory, but we need full for independent set
    # Actually for max independent set, we need the graph
    # Let's build CSR from the boolean matrix
    rows, cols = np.where(conflict)
    data = np.ones(len(rows), dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    return graph

def _greedy_max_independent_set(graph, rng):
    """
    Greedy max independent set with min-degree ordering.
    """
    n = graph.shape[0]
    adj = graph.tocsr()
    in_set = np.zeros(n, dtype=bool)
    remaining = list(range(n))
    degrees = np.array([adj[i].sum() for i in remaining], dtype=np.int64)
    # Use a simple approach: repeatedly pick node with min degree among remaining,
    # add to independent set, remove it and its neighbors from remaining.
    # For efficiency, we'll use a list and track degrees.
    # This is O(n^2) worst case, but with pruning it's faster.
    # For large n, we need a better approach. Let's use a heap or similar.
    # Given time constraints, let's do a simple iterative version.
    indep = []
    removed = np.zeros(n, dtype=bool)
    rem_deg = np.array([adj[i].sum() for i in range(n)], dtype=np.int64)
    rem_nodes = list(range(n))
    node_idx = {node: i for i, node in enumerate(rem_nodes)}
    
    while rem_nodes:
        if time.time() > DEADLINE - 5:
            break
        # Find min degree node
        min_idx = 0
        min_deg = rem_deg[rem_nodes[0]]
        for i in range(1, len(rem_nodes)):
            d = rem_deg[rem_nodes[i]]
            if d < min_deg:
                min_deg = d
                min_idx = i
        node = rem_nodes[min_idx]
        indep.append(node)
        removed[node] = True
        # Remove neighbors
        neighbors = adj[node].nonzero()[1]
        new_rem = []
        for nb in neighbors:
            if not removed[nb]:
                removed[nb] = True
                # Update degrees of remaining nodes
                for other in adj[nb].nonzero()[1]:
                    if not removed[other]:
                        rem_deg[other] -= 1
        # Rebuild rem_nodes
        rem_nodes = [nd for nd in rem_nodes if not removed[nd]]
    return indep

def _branch_and_bound(graph, start_set, max_time=10):
    """
    Try to expand start_set using branch and bound with max-degree branching.
    """
    n = graph.shape[0]
    adj = graph.tocsr()
    in_set = set(start_set)
    candidates = [i for i in range(n) if i not in in_set and all(j not in in_set for j in adj[i].nonzero()[1])]
    
    def dfs(cands, current_set, depth=0):
        if time.time() > max_time:
            return current_set
        if not cands:
            return current_set
        # Branch on node with max degree
        # Pick the candidate with highest degree among candidates
        best = max(cands, key=lambda x: adj[x].sum())
        # Branch 1: include best
        new_cands_1 = [c for c in cands if c != best and best not in adj[c].nonzero()[1]]
        result1 = dfs(new_cands_1, current_set + [best], depth + 1)
        # Branch 2: exclude best
        result2 = dfs([c for c in cands if c != best], current_set, depth + 1)
        # Return the better one
        if len(result1) >= len(result2):
            return result1
        else:
            return result2
    
    return dfs(candidates, list(start_set))

def construct():
    start_time = time.time()
    
    # Step 1: Generate candidate pool
    A = _gen_candidates(max_w=5, max_nz=5)
    norms_sq = _norms_sq(A)
    n = len(A)
    
    # Step 2: Try different R values
    # R is the minimum pairwise distance. We want to maximize |independent set|
    # R2 = R^2
    # Good R2 values: 2, 3, 4, 5, 6, 7, 8
    best_set = None
    best_size = 0
    
    # First, get a baseline from the seed (weight 2 vectors)
    # The seed uses vectors with exactly two non-zero entries of ±1
    seed_mask = (norms_sq == 2) & (np.all(np.abs(A) <= 1, axis=1))
    seed_indices = np.where(seed_mask)[0]
    if len(seed_indices) > 0:
        seed_A = A[seed_indices]
        seed_ns = norms_sq[seed_indices]
        R2 = 2
        dist2 = seed_ns[:, None] + seed_ns[None, :] - 2 * (seed_A @ seed_A.T)
        conflict = (dist2 < R2)
        np.fill_diagonal(conflict, False)
        # Simple greedy on seed
        n_s = len(seed_indices)
        adj_s = csr_matrix(conflict)
        indep_s = _greedy_max_independent_set(adj_s, None)
        best_set = [seed_indices[i] for i in indep_s]
        best_size = len(best_set)
    
    # Now try to improve with larger pool
    for R2 in [2, 3, 4, 5]:
        if time.time() > DEADLINE - 10:
            break
        try:
            graph = _build_conflict_graph(A, norms_sq, R2)
            # Greedy independent set
            indep = _greedy_max_independent_set(graph, None)
            size = len(indep)
            if size > best_size:
                best_set = indep
                best_size = size
            
            # Try to improve with local search / branch and bound on a subset
            # Take the greedy solution and try to expand
            if time.time() < DEADLINE - 20:
                # Try branch and bound on a random subset of candidates for speed
                rng = np.random.default_rng(42)
                sample_size = min(500, n)
                sample_idx = rng.choice(n, size=sample_size, replace=False)
                sub_graph = graph[np.ix_(sample_idx, sample_idx)]
                # Greedy on subset
                sub_indep = _greedy_max_independent_set(sub_graph, None)
                sub_set = [sample_idx[i] for i in sub_indep]
                if len(sub_set) > best_size:
                    best_set = sub_set
                    best_size = len(sub_set)
        except Exception:
            continue
    
    if best_set is None:
        # Fallback: return empty or seed
        return np.array([], dtype=np.int64)
    
    # Convert to integer array
    result = A[best_set].astype(np.int64)
    return result
