"""Difference triangle set construction with randomized greedy and local search."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, subject to all positive differences
    within each row being distinct across all rows."""
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # For the specific task (7,5): n_rows=7, k=5 -> 6 entries per row
    # Total differences per row: k*(k+1)/2 = 15
    # Total differences needed: n_rows * 15 = 105
    # Counting bound: smallest m such that m*(m+1)/2 >= 105 -> m=15 (but that's for one row)
    # Actually the counting bound on the largest entry m: we need 105 distinct differences
    # from {1, ..., m}, so m*(m+1)/2 >= 105 -> m >= 14.something, so m >= 15? 
    # Wait, the problem says counting bound is 105 for the largest entry. Let me re-read.
    # "the simple counting bound is 105" - this means m >= 105? That seems high.
    # Actually re-reading: "the largest entry as small as possible. The best known set has
    # largest entry 112; the simple counting bound is 105."
    # So the counting bound says m >= 105, and best known is 112.
    
    # Let's use a randomized greedy approach with multiple restarts and local search.
    
    best_rows = None
    best_scope = float('inf')
    
    def greedy_construct(n_rows, k, rng):
        """Greedy construction with randomization."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def randomized_greedy(n_rows, k, rng):
        """Randomized greedy: try different orderings and perturbations."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            # Add some randomization in step size
            while len(row) < k + 1:
                # Try a few random step sizes
                candidates = []
                for step in range(1, 5):
                    x_test = x + step
                    new_diffs = {x_test - a for a in row}
                    if not new_diffs & used:
                        candidates.append((x_test, new_diffs))
                if not candidates:
                    # Fallback: increment by 1
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                    continue
                # Pick randomly from valid candidates, weighted toward smaller
                # Use exponential weighting
                weights = [1.0 / (c[0] - x + 1) for c in candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                chosen = candidates[0]
                for c, w in zip(candidates, weights):
                    cum += w
                    if r <= cum:
                        chosen = c
                        break
                x = chosen[0]
                row.append(x)
                used |= chosen[1]
            rows.append(row)
        return rows
    
    def local_search(rows, n_rows, k, rng, max_iters=500):
        """Try to reduce the largest entry by modifying rows."""
        rows = [list(r) for r in rows]
        best = max(max(r) for r in rows)
        improved = True
        iterations = 0
        while improved and iterations < max_iters:
            improved = False
            iterations += 1
            # Try to reduce the largest element in each row
            for i in range(n_rows):
                row = rows[i]
                max_idx = len(row) - 1
                # Try to reduce the last element
                for new_val in range(row[max_idx] - 1, 0, -1):
                    # Check if this new value creates valid differences
                    # Remove old differences involving the last element
                    old_diffs = {row[max_idx] - a for a in row[:-1]}
                    # Also need to check all pairwise differences in the row
                    # Actually, we need to rebuild the difference set for this row
                    new_row = row[:max_idx] + [new_val]
                    new_diffs = set()
                    valid = True
                    for a in range(len(new_row)):
                        for b in range(a + 1, len(new_row)):
                            d = new_row[b] - new_row[a]
                            if d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                    if not valid:
                        continue
                    # Check against all other rows
                    conflict = False
                    for j in range(n_rows):
                        if j == i:
                            continue
                        other_row = rows[j]
                        for a in range(len(other_row)):
                            for b in range(a + 1, len(other_row)):
                                d = other_row[b] - other_row[a]
                                if d in new_diffs:
                                    conflict = True
                                    break
                            if conflict:
                                break
                        if conflict:
                            break
                    if not conflict:
                        rows[i] = new_row
                        new_max = max(max(r) for r in rows)
                        if new_max < best:
                            best = new_max
                            improved = True
                        break
        return rows, best
    
    def compute_scope(rows):
        """Compute the largest entry (scope)."""
        return max(max(r) for r in rows)
    
    def validate(rows, n_rows, k):
        """Validate that all differences are distinct across all rows."""
        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True
    
    # First, get a baseline solution
    rng = random.Random(42)
    baseline = greedy_construct(n_rows, k, rng)
    if validate(baseline, n_rows, k):
        best_rows = baseline
        best_scope = compute_scope(baseline)
    
    # Try randomized greedy with multiple seeds
    seed = 0
    while time.time() < deadline - 10:
        rng = random.Random(seed)
        rows = randomized_greedy(n_rows, k, rng)
        if validate(rows, n_rows, k):
            scope = compute_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
        seed += 1
    
    # Local search from best solution
    if best_rows is not None and time.time() < deadline - 5:
        rng = random.Random(123)
        improved_rows, improved_scope = local_search(best_rows, n_rows, k, rng, max_iters=200)
        if validate(improved_rows, n_rows, k) and improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # More local search iterations if time permits
    if best_rows is not None and time.time() < deadline - 2:
        rng = random.Random(456)
        improved_rows, improved_scope = local_search(best_rows, n_rows, k, rng, max_iters=300)
        if validate(improved_rows, n_rows, k) and improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    if best_rows is None:
        # Fallback: use the simple greedy
        best_rows = greedy_construct(n_rows, k, random.Random(0))
    
    return best_rows

# EVOLVE-BLOCK-END
