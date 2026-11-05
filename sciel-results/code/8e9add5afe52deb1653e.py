"""Difference triangle set (7,5): minimize the largest entry.
Spectral clustering on bipartite row-compatibility graph with eigenvector centrality,
plus backtracking and local search refinement."""
import time
import random
import math
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _generate_candidate_rows(k, max_val, count, rng):
    """Generate a diverse pool of candidate rows using randomized greedy construction."""
    candidates = []
    for _ in range(count):
        used = set()
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Randomized step selection
            step = rng.randint(1, max(1, max_val // (k + 1) + 3))
            x = row[-1] + step
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                # Try smaller steps
                found = False
                for s in range(1, step):
                    x_try = row[-1] + s
                    nd = {x_try - a for a in row}
                    if not nd & used:
                        x = x_try
                        new_diffs = nd
                        found = True
                        break
                if not found:
                    # Fallback: find next valid value
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                        x += 1
                    if x > max_val:
                        break
            if x > max_val:
                break
            row.append(x)
            used |= new_diffs
        if len(row) == k + 1:
            candidates.append(row)
    return candidates


def _all_diffs(rows):
    """Return the complete set of all differences across all rows."""
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used


def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _spectral_row_selection(k, n_rows, max_val, deadline, rng, pool_size=500):
    """Spectral clustering approach: generate candidate rows, build bipartite compatibility
    graph, use eigenvector centrality to rank rows by isolation, greedily select
    most compatible (isolated) rows."""
    
    # Phase 1: Generate diverse candidate pool
    candidates = _generate_candidate_rows(k, max_val, pool_size, rng)
    if len(candidates) < n_rows:
        # Generate more with different randomization
        for _ in range(10):
            more = _generate_candidate_rows(k, max_val, pool_size // 2, rng)
            candidates.extend(more)
        if len(candidates) < n_rows:
            return None, float('inf')
    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    if len(unique_candidates) < n_rows:
        return None, float('inf')
    
    # Precompute difference sets for each candidate
    diff_sets = []
    for c in unique_candidates:
        diff_sets.append(_row_diffs(c))
    
    n_cands = len(unique_candidates)
    
    # Build compatibility matrix: A[i][j] = 1 if row i and row j have disjoint diffs
    # For memory efficiency, use sparse representation via bitsets or direct computation
    # Since we need eigenvector centrality, build the adjacency matrix
    
    # For large n_cands, use a sampling approach for the adjacency
    sample_size = min(n_cands, 200)
    indices = list(range(n_cands))
    if n_cands > sample_size:
        indices = rng.sample(indices, sample_size)
    
    # Build adjacency for sampled subset
    n_sample = len(indices)
    adj = np.zeros((n_sample, n_sample), dtype=np.float64)
    
    for i in range(n_sample):
        if time.time() > deadline:
            break
        di = diff_sets[indices[i]]
        for j in range(i + 1, n_sample):
            dj = diff_sets[indices[j]]
            if not (di & dj):
                adj[i][j] = 1.0
                adj[j][i] = 1.0
    
    # Compute eigenvector centrality
    if n_sample > 1:
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(adj)
            # Largest eigenvalue's eigenvector gives centrality
            max_idx = np.argmax(eigenvalues)
            centrality = np.abs(eigenvectors[:, max_idx])
            # Normalize
            max_c = centrality.max()
            if max_c > 0:
                centrality = centrality / max_c
        except np.linalg.LinAlgError:
            centrality = np.ones(n_sample) / n_sample
    else:
        centrality = np.array([1.0])
    
    # Map centrality back to original indices
    cand_centrality = np.zeros(n_cands)
    for i, idx in enumerate(indices):
        cand_centrality[idx] = centrality[i]
    
    # Score each candidate: prefer low scope (max entry) and high centrality (isolation)
    scores = []
    for i, c in enumerate(unique_candidates):
        scope_i = max(c)
        # Weight: minimize scope, maximize centrality (isolation means fewer conflicts)
        score = scope_i * 1.0 - cand_centrality[i] * 10.0
        scores.append((score, scope_i, i))
    
    # Sort by score (lower is better)
    scores.sort()
    
    # Greedy selection: pick rows that are mutually compatible
    selected = []
    selected_diffs = set()
    
    for score, scope_i, idx in scores:
        if len(selected) == n_rows:
            break
        if time.time() > deadline:
            break
        
        di = diff_sets[idx]
        if not (di & selected_diffs):
            selected.append(unique_candidates[idx])
            selected_diffs |= di
    
    if len(selected) < n_rows:
        return None, float('inf')
    
    return selected, _scope(selected)


def _greedy_construct(n_rows, k, randomize=False, seed_offset=0):
    """Greedy construction with optional randomization."""
    rng = random.Random(42 + seed_offset)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if randomize:
                max_step = max(1, int((k + 1) * 2.5))
                found = False
                for _attempt in range(50):
                    step = rng.randint(1, max_step)
                    x = row[-1] + step
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        found = True
                        break
                if not found:
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                        x += 1
            else:
                x += 1
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
            
            row.append(x)
            used |= new_diffs
        rows.append(row)
    
    return rows


def _simulated_annealing(rows, k, deadline, rng):
    """Simulated annealing to reduce the maximum entry by coordinated row modifications."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    used = _all_diffs(rows)
    
    current_scope = _scope(rows)
    T = 20.0  # initial temperature
    cooling = 0.9995
    min_T = 0.1
    
    while time.time() < deadline and T > min_T:
        # Pick a random row and position to modify
        ri = rng.randint(0, len(rows) - 1)
        row = rows[ri]
        pos = rng.randint(1, len(row) - 1)
        
        old_val = row[pos]
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos + 1 < len(row) else old_val + 15
        
        if low >= high:
            T *= cooling
            continue
        
        # Sample new value
        new_val = rng.randint(low, high)
        if new_val == old_val:
            T *= cooling
            continue
        
        # Compute diff changes
        old_diffs = set()
        for i in range(pos):
            old_diffs.add(old_val - row[i])
        for i in range(pos + 1, len(row)):
            old_diffs.add(row[i] - old_val)
        
        new_diffs = set()
        for i in range(pos):
            new_diffs.add(new_val - row[i])
        for i in range(pos + 1, len(row)):
            new_diffs.add(row[i] - new_val)
        
        conflict = new_diffs & (used - old_diffs)
        if conflict:
            T *= cooling
            continue
        
        # Accept if improves or with probability exp(-delta/T)
        new_scope = _scope([row[:pos] + [new_val] + row[pos + 1:]] + rows[:ri] + rows[ri+1:])
        delta = new_scope - current_scope
        
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            # Accept move
            row[pos] = new_val
            used -= old_diffs
            used |= new_diffs
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        
        T *= cooling
    
    return best_rows


def _backtrack_row(k, used, max_val, rng, time_limit=None, deadline=None):
    """Try to build a single row of k+1 elements starting at 0, with all differences
    not in 'used', and all elements <= max_val. Returns row or None."""
    if deadline is not None and time.time() > deadline:
        return None
    
    row = [0]
    
    def backtrack(idx):
        if deadline is not None and time.time() > deadline:
            return False
        if idx == k + 1:
            return True
        
        low = row[-1] + 1
        high = max_val
        
        # Use MRV-inspired ordering: try values that create fewer conflicts first
        # Sample a subset of candidates for speed
        if high - low > 30:
            # Sample 30 random candidates plus some near the lower end
            candidates = set(rng.sample(range(low, high + 1), min(30, high - low + 1)))
            candidates.update(range(low, min(low + 10, high + 1)))
            candidates = list(candidates)
            rng.shuffle(candidates)
        else:
            candidates = list(range(low, high + 1))
            rng.shuffle(candidates)
        
        for val in candidates:
            new_diffs = {val - a for a in row}
            if new_diffs & used:
                continue
            row.append(val)
            used.update(new_diffs)
            if backtrack(idx + 1):
                return True
            used.difference_update(new_diffs)
            row.pop()
        
        return False
    
    if backtrack(1):
        return row
    return None


def _construct_with_backtracking(n_rows, k, max_val, rng, deadline):
    """Construct n_rows rows using backtracking, each with elements <= max_val."""
    used = set()
    rows = []
    
    for r in range(n_rows):
        if time.time() > deadline:
            return None
        row = _backtrack_row(k, used, max_val, rng, deadline=deadline)
        if row is None:
            return None
        rows.append(row)
        used |= _row_diffs(row)
    
    return rows


def _local_search_improve(rows, k, deadline, rng=None):
    """Local search: try to reduce the maximum entry by modifying individual elements.
    Works at the difference level: for each element position, try alternative values."""
    if rng is None:
        rng = random.Random()
    
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    used = _all_diffs(rows)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_scope = _scope(rows)
        
        # Try to reduce elements at or near the current maximum
        for ri in range(len(rows)):
            if time.time() > deadline:
                break
            row = rows[ri]
            for pos in range(len(row) - 1, 0, -1):  # from last to first
                if time.time() > deadline:
                    break
                old_val = row[pos]
                if old_val < current_scope - 2:
                    continue
                
                # Try reducing this value
                low = row[pos - 1] + 1
                high = old_val - 1
                if pos + 1 < len(row):
                    high = min(high, row[pos + 1] - 1)
                
                if low > high:
                    continue
                
                # Sample candidate values, preferring smaller ones
                n_candidates = min(20, high - low + 1)
                candidates = rng.sample(range(low, high + 1), n_candidates)
                candidates.sort()  # try smallest first for max reduction
                
                for new_val in candidates:
                    if time.time() > deadline:
                        break
                    if new_val >= old_val:
                        continue
                    
                    # Compute old diffs involving pos
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(old_val - row[i])
                    for i in range(pos + 1, len(row)):
                        old_diffs.add(row[i] - old_val)
                    
                    # Compute new diffs involving pos
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for i in range(pos + 1, len(row)):
                        new_diffs.add(row[i] - new_val)
                    
                    # Check for conflicts
                    conflict = new_diffs & (used - old_diffs)
                    if conflict:
                        continue
                    
                    # Accept this change
                    row[pos] = new_val
                    used -= old_diffs
                    used |= new_diffs
                    improved = True
                    break
    
    return rows


def _difference_swap_mutation(rows, k, used, num_swaps=None, rng=None):
    """Perform difference-swap moves to reduce the max entry."""
    if rng is None:
        rng = random.Random()
    if num_swaps is None:
        num_swaps = rng.randint(1, 3)
    
    rows = [list(r) for r in rows]
    used = set(used)
    n_rows = len(rows)
    
    for _ in range(num_swaps):
        r_idx = rng.randint(0, n_rows - 1)
        row = rows[r_idx]
        pos = rng.randint(1, len(row) - 1)
        
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos + 1 < len(row) else old_val + 20
        
        if low >= high:
            continue
        
        best_new_val = None
        best_score = float('inf')
        
        candidates = set()
        for _ in range(30):
            candidates.add(rng.randint(low, high))
        current_max = max(row)
        if pos == len(row) - 1:
            for dv in range(1, 8):
                candidates.add(max(low, current_max - dv))
        
        for new_val in candidates:
            if new_val == old_val or new_val < low or new_val > high:
                continue
            
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for i in range(pos + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for i in range(pos + 1, len(row)):
                new_diffs.add(row[i] - new_val)
            
            conflict = new_diffs & (used - old_diffs)
            if conflict:
                continue
            
            new_row_max = max(row[:pos] + [new_val] + row[pos + 1:])
            old_row_max = max(row)
            score = new_row_max - old_row_max
            if score < best_score:
                best_score = score
                best_new_val = new_val
        
        if best_new_val is not None:
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for i in range(pos + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(best_new_val - row[i])
            for i in range(pos + 1, len(row)):
                new_diffs.add(row[i] - best_new_val)
            
            row[pos] = best_new_val
            used -= old_diffs
            used |= new_diffs
    
    return rows, used


def _randomized_backtracking(n_rows, k, deadline, rng, initial_max=120):
    """Repeatedly try to construct a valid set using randomized backtracking,
    with decreasing max_val bounds."""
    best_rows = None
    best_scope = float('inf')
    
    # Try various max_val bounds
    for max_val in range(initial_max, 104, -1):
        if time.time() > deadline:
            break
        
        # Multiple attempts at this max_val
        attempts = 3
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            rows = _construct_with_backtracking(n_rows, k, max_val, rng, deadline)
            if rows is not None:
                scope = _scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
                    break  # Found a valid set at this max_val, try lower
    
    return best_rows, best_scope


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    rng = random.Random(42)
    
    # Phase 1: Spectral clustering row selection (primary structural change)
    best_rows = None
    best_scope = float('inf')
    
    # Try spectral selection with various max_val bounds
    for max_val in [115, 112, 110, 108, 106, 105]:
        if time.time() > deadline - 50.0:
            break
        spec_rows, spec_scope = _spectral_row_selection(
            k, n_rows, max_val, deadline - 50.0, rng, pool_size=800
        )
        if spec_rows is not None and spec_scope < best_scope:
            best_scope = spec_scope
            best_rows = spec_rows
            if spec_scope <= 105:
                break  # Hit counting bound
    
    # If spectral didn't find anything, fall back to greedy
    if best_rows is None:
        for seed_off in range(15):
            rows = _greedy_construct(n_rows, k, randomize=True, seed_offset=seed_off)
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
    
    # Phase 2: Randomized backtracking with decreasing bounds
    if time.time() < deadline - 40.0 and best_rows is not None:
        bt_rows, bt_scope = _randomized_backtracking(
            n_rows, k, deadline - 40.0, rng, initial_max=best_scope
        )
        if bt_rows is not None and bt_scope < best_scope:
            best_rows = bt_rows
            best_scope = bt_scope
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 15.0 and best_rows is not None:
        sa_rows = _simulated_annealing(best_rows, k, deadline - 15.0, rng)
        sa_scope = _scope(sa_rows)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope
    
    # Phase 4: Local search to improve
    if time.time() < deadline - 5.0 and best_rows is not None:
        improved = _local_search_improve(best_rows, k, deadline - 5.0, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    # Phase 5: Final refinement with difference-swap mutations
    if time.time() < deadline - 2.0 and best_rows is not None:
        used = _all_diffs(best_rows)
        for _ in range(15):
            if time.time() > deadline - 2.0:
                break
            refined, used = _difference_swap_mutation(
                best_rows, k, used, num_swaps=rng.randint(2, 5), rng=rng
            )
            new_scope = _scope(refined)
            if new_scope < best_scope:
                best_rows = refined
                best_scope = new_scope
    
    return best_rows

# EVOLVE-BLOCK-END
