import time
import random
import math
import numpy as np
from functools import lru_cache

_DEADLINE = 0.0
_BEST = None

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _fast_greedy(k, n_rows, rng, spread=False):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

def _get_row_diffs(row):
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs

def _sa_vectorized(rows, k, deadline, rng, T_init=10.0, cooling=0.9997, min_T=0.05):
    """
    SA with vectorized cost computation and row-pair swap moves.
    Cost = scope + W * collisions, where collisions = sum(max(0, c-1) for c in diff_counts).
    """
    n_rows = len(rows)
    mat = np.array(rows, dtype=np.int32)  # shape (n_rows, k+1)
    current_scope = int(mat.max())
    
    # Precompute pair indices for difference extraction
    # For a row of length k+1, there are (k+1)*k/2 pairs
    n_pairs = (k + 1) * k // 2
    # Build index arrays for all (i, j) pairs with i < j within a row
    row_idx = []
    col_idx = []
    for i in range(k + 1):
        for j in range(i + 1, k + 1):
            row_idx.append(i)
            col_idx.append(j)
    row_idx = np.array(row_idx, dtype=np.int32)
    col_idx = np.array(col_idx, dtype=np.int32)
    
    def get_all_diffs(m):
        """Return 1D array of all differences across all rows."""
        diffs_per_row = m[:, col_idx] - m[:, row_idx]  # (n_rows, n_pairs)
        return diffs_per_row.flatten()
    
    def compute_cost(m):
        diffs = get_all_diffs(m)
        if len(diffs) == 0:
            return int(m.max()), 0
        unique, counts = np.unique(diffs, return_counts=True)
        collisions = int(np.sum(np.maximum(0, counts - 1)))
        scope = int(m.max())
        return scope, collisions
    
    current_scope, current_collisions = compute_cost(mat)
    W = 15.0
    current_cost = current_scope + W * current_collisions
    best_cost = current_cost
    best_mat = mat.copy()
    best_scope = current_scope
    
    T = T_init
    n_iters = 0
    stagnant = 0
    MAX_STAGNANT = 3000
    
    while time.time() < deadline:
        n_iters += 1
        r = rng.random()
        
        if r < 0.25:
            # --- Row rebuild: replace one row via greedy with conflict awareness ---
            i = rng.randint(0, n_rows - 1)
            old_row = mat[i].copy()
            
            # Build a set of currently-used differences (excluding row i)
            other_mat = np.delete(mat, i, axis=0)
            other_diffs = get_all_diffs(other_mat)
            used_diffs = set(int(d) for d in other_diffs)
            
            # Greedily build new row
            new_row = [0]
            x = 0
            ok = True
            for j in range(1, k + 1):
                found = False
                # Try nearby values first, then expand
                for attempt in range(50):
                    if attempt == 0:
                        start = x + 1
                        end = x + 1 + max(1, (k - j + 1) * 3)
                    else:
                        start = x + 1
                        end = x + 1 + attempt * 5
                    for t in range(start, min(end, current_scope + 10)):
                        conflict = False
                        for a in new_row:
                            d = t - a
                            if d <= 0:
                                continue
                            if d in used_diffs:
                                conflict = True
                                break
                        if not conflict:
                            new_row.append(t)
                            for a in new_row[:-1]:
                                used_diffs.add(t - a)
                            x = t
                            found = True
                            break
                    if found:
                        break
                if not found:
                    ok = False
                    break
            
            if not ok or len(new_row) != k + 1:
                continue
            
            new_row_arr = np.array(new_row, dtype=np.int32)
            if new_row_arr.max() > current_scope + 5:
                continue
            
            mat[i] = new_row_arr
            new_scope, new_collisions = compute_cost(mat)
            new_cost = new_scope + W * new_collisions
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                current_collisions = new_collisions
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_mat = mat.copy()
                    best_scope = new_scope
                    stagnant = 0
            else:
                mat[i] = old_row
                
        elif r < 0.45:
            # --- Row shift ---
            i = rng.randint(0, n_rows - 1)
            shift = rng.randint(-3, 3)
            if shift == 0:
                continue
            old_row = mat[i].copy()
            new_row = old_row.copy()
            new_row[1:] += shift
            if new_row[1:].min() <= 0 or new_row[1:].max() > current_scope + 10:
                continue
            if np.any(np.diff(new_row) <= 0):
                continue
            mat[i] = new_row
            new_scope, new_collisions = compute_cost(mat)
            new_cost = new_scope + W * new_collisions
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                current_collisions = new_collisions
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_mat = mat.copy()
                    best_scope = new_scope
                    stagnant = 0
            else:
                mat[i] = old_row
                
        elif r < 0.70:
            # --- Single-element move ---
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            old_val = int(mat[i, j])
            prev_val = int(mat[i, j - 1])
            next_val = int(mat[i, j + 1]) if j < k else current_scope + 100
            
            lo = prev_val + 1
            hi = next_val - 1
            if lo > hi:
                continue
            
            if old_val == current_scope:
                new_val = rng.randint(lo, min(hi, old_val - 1))
            else:
                span = hi - lo
                delta = int(rng.gauss(0, max(1, span * 0.25)))
                new_val = old_val + delta
            
            if new_val == old_val or new_val < lo or new_val > hi:
                continue
            
            mat[i, j] = new_val
            new_scope, new_collisions = compute_cost(mat)
            new_cost = new_scope + W * new_collisions
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                current_collisions = new_collisions
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_mat = mat.copy()
                    best_scope = new_scope
                    stagnant = 0
            else:
                mat[i, j] = old_val
                
        else:
            # --- Pairwise element move ---
            i = rng.randint(0, n_rows - 1)
            j1, j2 = rng.sample(range(1, k + 1), 2)
            if j1 > j2:
                j1, j2 = j2, j1
            
            old_v1 = int(mat[i, j1])
            old_v2 = int(mat[i, j2])
            
            prev1 = int(mat[i, j1 - 1])
            next1 = int(mat[i, j1 + 1]) if j1 < k else current_scope + 100
            prev2 = int(mat[i, j2 - 1])
            next2 = int(mat[i, j2 + 1]) if j2 < k else current_scope + 100
            
            lo1 = prev1 + 1
            hi1 = next1 - 1
            lo2 = prev2 + 1
            hi2 = next2 - 1
            
            if lo1 > hi1 or lo2 > hi2:
                continue
            
            new_v1 = rng.randint(lo1, hi1)
            new_v2 = rng.randint(lo2, hi2)
            
            if new_v1 == old_v1 and new_v2 == old_v2:
                continue
            
            mat[i, j1] = new_v1
            mat[i, j2] = new_v2
            if np.any(np.diff(mat[i]) <= 0):
                mat[i, j1] = old_v1
                mat[i, j2] = old_v2
                continue
            
            new_scope, new_collisions = compute_cost(mat)
            new_cost = new_scope + W * new_collisions
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                current_collisions = new_collisions
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_mat = mat.copy()
                    best_scope = new_scope
                    stagnant = 0
            else:
                mat[i, j1] = old_v1
                mat[i, j2] = old_v2
        
        T = max(T * cooling, min_T)
        
        if n_iters % 500 == 0 and stagnant > MAX_STAGNANT:
            mat = best_mat.copy()
            current_scope = best_scope
            current_scope, current_collisions = compute_cost(mat)
            current_cost = current_scope + W * current_collisions
            for _ in range(rng.randint(1, 3)):
                i = rng.randint(0, n_rows - 1)
                j = rng.randint(1, k)
                prev_val = int(mat[i, j - 1])
                next_val = int(mat[i, j + 1]) if j < k else current_scope + 100
                lo = prev_val + 1
                hi = next_val - 1
                if lo <= hi:
                    mat[i, j] = rng.randint(lo, hi)
                    if np.any(np.diff(mat[i]) <= 0):
                        mat[i, j] = prev_val + 1
            current_scope, current_collisions = compute_cost(mat)
            current_cost = current_scope + W * current_collisions
            T = T_init * 0.5
            stagnant = 0
    
    return best_scope, [list(map(int, r)) for r in best_mat]

def _dfs_row_minimize(row, k, used_diffs, max_scope, deadline):
    best_row = None
    best_max = max_scope + 1
    used_set = set(used_diffs)
    
    def dfs(pos, current_row, current_x):
        nonlocal best_row, best_max
        if time.time() > deadline:
            return
        if pos == k + 1:
            if current_x < best_max:
                best_max = current_x
                best_row = list(current_row)
            return
        if current_x + (k + 1 - pos) >= best_max:
            return
        for t in range(current_x + 1, best_max):
            ok = True
            for a in current_row:
                d = t - a
                if d <= 0:
                    continue
                if d in used_set:
                    ok = False
                    break
            if ok:
                current_row.append(t)
                for a in current_row[:-1]:
                    used_set.add(t - a)
                dfs(pos + 1, current_row, t)
                for a in current_row[:-1]:
                    used_set.discard(t - a)
                current_row.pop()
            if time.time() > deadline:
                return
    
    dfs(1, [0], 0)
    return best_row

def _knapsack_packing(n_rows, k, deadline, rng):
    """
    Structural change: Multi-dimensional knapsack / set packing.
    We generate a pool of candidate rows (items) with small scopes.
    Each item is defined by its difference set (bitset) and scope.
    We use DP/B&B to select n_rows items with disjoint difference vectors.
    """
    # 1. Generate a diverse pool of candidate rows
    # We want rows with scope up to some limit M.
    # Let's estimate M based on greedy results.
    greedy_rows = _fast_greedy(k, n_rows, rng, spread=False)
    M_initial = max(r[-1] for r in greedy_rows)
    
    # Pool of candidates: list of (scope, diff_bitset, row)
    # diff_bitset is a Python int where bit d is set if difference d is present.
    candidates = []
    seen_diffs = set() # To avoid duplicate rows in pool? Maybe not necessary but good for diversity.
    
    # Generate candidates with scope up to M_initial + 20
    M_max = M_initial + 20
    
    # We need to be careful with memory and time.
    # Number of possible rows is huge. We will sample them.
    # Strategy: Randomized DFS to generate valid rows with small scope.
    
    def generate_random_row(max_scope, rng):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Try to pick next element
            found = False
            # Randomize the search order to get diverse rows
            # But we need to ensure we don't exceed max_scope
            # We'll try values from x+1 to max_scope
            # To make it faster, we can use a bitset for used diffs in this row
            row_diffs = 0
            for a in row:
                pass # Precompute not needed for single row gen if we just check conflicts
            
            # Actually, let's use the same bitset logic as greedy but with randomization
            # We'll try to build a row avoiding a global set of diffs? No, this is just generation.
            
            # Simple randomized greedy:
            # At each step, pick a random valid next element.
            # Valid means: strictly increasing, and differences with previous elements are unique within the row.
            
            # To speed up, we can precompute potential conflicts?
            # For k=5, row length is 6. Pairs are 15.
            
            # Let's just do a random walk.
            # To ensure we stay within max_scope, we backtrack if we get stuck.
            
            # This generation phase might be slow if done naively.
            # Let's use a heuristic: 
            # Start with 0.
            # Next element: pick from {1, ..., max_scope//k} roughly?
            
            # Alternative: Use the existing _fast_greedy but with random perturbations.
            # Let's generate 1000 candidates using random greedy.
            
            pass # Placeholder

    # Better generation: 
    # We will generate candidates by modifying the greedy row with random swaps/moves, 
    # or by using a randomized DFS that picks random valid next steps.
    
    def gen_row_random(max_scope, rng):
        row = [0]
        x = 0
        row_diffs = 0 # Bitset of diffs within this row
        
        for _ in range(k):
            # Find all valid next values
            valid_vals = []
            # We can't scan up to max_scope every time if max_scope is large (e.g. 100).
            # But k is small.
            # Let's scan from x+1 to max_scope.
            # Optimization: if x is close to max_scope, few options.
            
            # To make it faster, we can limit the search range based on remaining elements.
            # Min gap is 1. So we need at least (k - len(row)) more elements.
            # So x + (k - len(row)) <= max_scope
            
            for t in range(x + 1, max_scope + 1):
                # Check if t can be added
                new_diffs = 0
                for a in row:
                    d = t - a
                    # d is always positive since t > x >= a
                    # Check if d is already in row_diffs
                    if (row_diffs >> d) & 1:
                        break
                    new_diffs |= (1 << d)
                else:
                    # No conflict
                    valid_vals.append(t)
            
            if not valid_vals:
                return None
            
            # Pick a random valid value
            # Bias towards smaller values to keep scope small?
            # Or uniform? Uniform might find diverse structures.
            # Let's pick uniformly from valid_vals.
            t = rng.choice(valid_vals)
            
            row.append(t)
            x = t
            # Update row_diffs
            for a in row[:-1]:
                d = t - a
                row_diffs |= (1 << d)
        
        return row

    # Generate pool
    N_CANDIDATES = 5000
    candidates = []
    for _ in range(N_CANDIDATES):
        if time.time() > deadline - 10: break
        row = gen_row_random(M_max, rng)
        if row:
            # Compute bitset
            bitset = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    bitset |= (1 << d)
            scope = row[-1]
            candidates.append((scope, bitset, row))
    
    if not candidates:
        return None, None

    # Sort by scope
    candidates.sort(key=lambda x: x[0])
    
    # DP / B&B to select n_rows items with disjoint bitsets.
    # State: (num_selected, used_bitset) -> min_max_scope
    # This state space is too large for pure DP if bitset is large (100+ bits).
    # However, the number of items is limited.
    # We can use a recursive search with pruning.
    
    # Pruning:
    # 1. If current_max_scope > best_found, prune.
    # 2. If it's impossible to select n_rows - num_selected more items from the remaining candidates 
    #    that are disjoint from used_bitset, prune.
    #    This is a set packing feasibility check.
    
    # To make this efficient, we should limit the pool size or use a greedy selection first.
    
    # Let's try a greedy selection first to get a good upper bound.
    # Sort candidates by scope.
    # Greedily pick the smallest scope candidate that doesn't conflict.
    
    def greedy_select(cands, n_needed):
        used = 0
        selected = []
        max_scope = 0
        for scope, bitset, row in cands:
            if (used & bitset) == 0:
                selected.append(row)
                used |= bitset
                max_scope = max(max_scope, scope)
                if len(selected) == n_needed:
                    return max_scope, selected
        return None, None
    
    best_scope, best_rows = greedy_select(candidates, n_rows)
    
    if best_scope is None:
        return None, None
    
    # Now, use B&B to improve.
    # We can try to replace rows with smaller scope rows or find a different combination.
    # Given time constraints, a simple local search on the set of selected rows might be better.
    # Or, we can use the candidates to do a randomized restart of the greedy selection.
    
    # Let's do a few randomized greedy passes.
    for _ in range(100):
        if time.time() > deadline - 5: break
        # Shuffle candidates slightly? Or just pick random order?
        # If we pick random order, we might miss small scope ones.
        # Let's stick to sorted order but add some noise to the score.
        # Or just re-run greedy on a subset?
        
        # Actually, the greedy on sorted list is deterministic.
        # To explore, we can randomly remove some candidates from the top and re-greedy?
        # Or, we can use the SA on the selected rows to refine.
        
        # Let's just return the best found by greedy and let the main SA refine it.
        # This "Knapsack" approach mainly serves as a strong initializer.
        
        pass
    
    return best_scope, best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 108.0

    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])

    rng = random.Random(42)

    # 1. Try the new Knapsack/Packing strategy
    knap_scope, knap_rows = _knapsack_packing(n_rows, k, _DEADLINE, rng)
    if knap_rows:
        consider(knap_rows)

    # 2. Generate diverse initial candidates (as before)
    candidates = []
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    for _ in range(3):
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                max_step = rng.randint(1, 10)
                for t in range(x + 1, x + 1 + max_step * 4):
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    if (used & new) == 0:
                        row.append(t)
                        used |= new
                        x = t
                        found = True
                        break
                if not found:
                    t = x + 1
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    while used & new:
                        t += 1
                        new = 0
                        for a in row:
                            new |= (1 << (t - a))
                    row.append(t)
                    used |= new
                    x = t
            rows.append(row)
        candidates.append(rows)
    
    # Add knapsack result if it's better than some candidates
    if knap_rows:
        candidates.append(knap_rows)

    candidates.sort(key=_scope)

    # Run vectorized SA on top candidates
    for idx, cand in enumerate(candidates[:3]):
        if time.time() > _DEADLINE:
            break
        remaining = _DEADLINE - time.time()
        if idx == 0:
            sa_deadline = time.time() + remaining * 0.35
        elif idx == 1:
            sa_deadline = time.time() + remaining * 0.35
        else:
            sa_deadline = _DEADLINE
        s, rows = _sa_vectorized(cand, k, sa_deadline, rng)
        consider(rows)

    # DFS refinement on best result
    if _BEST is not None and time.time() < _DEADLINE - 5:
        best_rows = [list(r) for r in _BEST[1]]
        all_diffs = set()
        for r in best_rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    all_diffs.add(r[j] - r[i])
        current_scope = _scope(best_rows)
        improved = True
        while improved and time.time() < _DEADLINE - 5:
            improved = False
            for i in range(n_rows):
                if time.time() > _DEADLINE - 5:
                    break
                old_row = list(best_rows[i])
                row_diffs = set()
                for a in range(len(old_row)):
                    for b in range(a + 1, len(old_row)):
                        row_diffs.add(old_row[b] - old_row[a])
                other_diffs = all_diffs - row_diffs
                new_row = _dfs_row_minimize(old_row, k, other_diffs, current_scope, _DEADLINE - 5)
                if new_row is not None and new_row[-1] < old_row[-1]:
                    best_rows[i] = new_row
                    all_diffs = other_diffs | set(
                        new_row[b] - new_row[a]
                        for a in range(len(new_row))
                        for b in range(a + 1, len(new_row))
                    )
                    current_scope = min(current_scope, _scope(best_rows))
                    improved = True
                else:
                    all_diffs = other_diffs | row_diffs
                    best_rows[i] = old_row
        consider(best_rows)

    # Final polish SA
    if _BEST is not None and time.time() < _DEADLINE - 3:
        remaining = _DEADLINE - time.time()
        if remaining > 3:
            s, rows = _sa_vectorized(_BEST[1], k, _DEADLINE - 1, rng, T_init=3.0, cooling=0.9999, min_T=0.01)
            consider(rows)

    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
