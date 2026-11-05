"""Program to find a large subset of [1,256] where no pair's product is divisible by their sum."""
import time
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

DEADLINE = 85.0  # safety margin before 90s kill


def _build_conflict_matrix():
    """Build conflict matrix: edge (i,j) exists if (i*j) % (i+j) == 0."""
    n = 256
    rows, cols, vals = [], [], []
    for i in range(1, 257):
        for j in range(i + 1, 257):
            if (i * j) % (i + j) == 0:
                rows.append(i - 1)
                cols.append(j - 1)
                vals.append(1)
                rows.append(j - 1)
                cols.append(i - 1)
                vals.append(1)
    if not rows:
        return csr_matrix((256, 256), dtype=np.int8)
    mat = csr_matrix((vals, (rows, cols)), shape=(256, 256), dtype=np.int8)
    return mat


def _spectral_seed(conflict_mat, k=10):
    """Use eigenvector of conflict graph to find low-conflict core."""
    n = 256
    # Use the smallest non-trivial eigenvector (Fiedler-like) or just the principal one
    try:
        # Get the eigenvector corresponding to the smallest eigenvalue (most uniform)
        # Actually, for independent set, we want vertices with LOW degree/conflict
        # Let's use the principal eigenvector and pick vertices with low values
        vals, vecs = eigsh(conflict_mat, k=min(k, n - 1), which='LA')
        # Use the eigenvector with largest eigenvalue
        v = vecs[:, 0]
        # Normalize to [0, 1]
        v_min, v_max = v.min(), v.max()
        if v_max - v_min > 1e-10:
            v_norm = (v - v_min) / (v_max - v_min)
        else:
            v_norm = np.ones(n) / n
        # Pick vertices with LOW eigenvector values (low conflict)
        # Sort by eigenvector value, take the lowest
        order = np.argsort(v_norm)
        seed = list(order[:k])
        return seed, v_norm
    except Exception:
        # Fallback: random seed
        seed = list(random.sample(range(n), k))
        return seed, np.ones(n) / n


def _greedy_expand(seed, conflict_mat, v_norm):
    """Greedily expand the seed set with vertices having low conflict."""
    n = 256
    selected = set(seed)
    
    # Precompute conflict sets for fast lookup
    conflict_sets = []
    for i in range(n):
        row = conflict_mat.getrow(i)
        conflict_sets.append(set(row.indices))
    
    # Build candidate list sorted by eigenvector value (low = preferred)
    candidates = list(range(n))
    candidates.sort(key=lambda x: v_norm[x])
    
    for v in candidates:
        if v in selected:
            continue
        # Check if v conflicts with any selected vertex
        can_add = True
        for s in conflict_sets[v]:
            if s in selected:
                can_add = False
                break
        if can_add:
            selected.add(v)
    
    return selected


def _local_search(selected, conflict_mat, max_iters=5000):
    """Improve the set via local search: try to add/remove/swap vertices."""
    n = 256
    selected = set(selected)
    
    # Precompute conflict sets
    conflict_sets = []
    for i in range(n):
        row = conflict_mat.getrow(i)
        conflict_sets.append(set(row.indices))
    
    improved = True
    iters = 0
    start_time = time.time()
    
    while improved and iters < max_iters and (time.time() - start_time) < (DEADLINE - time.time() if False else 50):
        improved = False
        iters += 1
        if time.time() > DEADLINE - 5:
            break
        
        # Try adding a vertex not in selected
        for v in range(n):
            if v in selected:
                continue
            can_add = True
            for s in conflict_sets[v]:
                if s in selected:
                    can_add = False
                    break
            if can_add:
                selected.add(v)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: remove one, add one (net gain 0 but might enable more)
        # Actually, try 2-opt: remove two, add two if net gain >= 0
        # Simpler: try removing a vertex that blocks many, then re-add
        for v in list(selected):
            # Count how many vertices are blocked by v
            blocked = 0
            for u in conflict_sets[v]:
                if u not in selected:
                    blocked += 1
            if blocked > 0:
                # Try removing v and see if we can add more
                selected.discard(v)
                added = 0
                for u in conflict_sets[v]:
                    if u not in selected:
                        can_add = True
                        for s in conflict_sets[u]:
                            if s in selected:
                                can_add = False
                                break
                        if can_add:
                            selected.add(u)
                            added += 1
                if added > 0:
                    improved = True
                    break
                else:
                    # Put v back
                    selected.add(v)
    
    return selected


def _verify(selected):
    """Verify the set is valid: no pair has (a*b) % (a+b) == 0."""
    sel = list(selected)
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            a, b = sel[i], sel[j]
            if (a * b) % (a + b) == 0:
                return False
    return True


def construct():
    start_time = time.time()
    
    # Build conflict matrix
    conflict_mat = _build_conflict_matrix()
    
    # Spectral seeding
    seed, v_norm = _spectral_seed(conflict_mat, k=15)
    
    # Greedy expansion
    selected = _greedy_expand(seed, conflict_mat, v_norm)
    
    # Local search
    selected = _local_search(selected, conflict_mat)
    
    # Try multiple random restarts if time permits
    best = set(selected)
    best_size = len(best)
    
    while time.time() < DEADLINE - 10:
        # Random restart with different seed size
        k = random.randint(5, 25)
        seed2 = list(random.sample(range(256), k))
        selected2 = _greedy_expand(seed2, conflict_mat, v_norm)
        selected2 = _local_search(selected2, conflict_mat, max_iters=2000)
        
        if len(selected2) > best_size:
            best = set(selected2)
            best_size = len(selected2)
        
        if time.time() > DEADLINE - 5:
            break
    
    # Final verification
    if not _verify(best):
        # Fallback to odd numbers
        best = set(range(1, 257, 2))
    
    return list(best)
