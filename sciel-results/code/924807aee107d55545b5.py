import time
import random
import math
import bisect
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    num_diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * num_diffs_per_row

    # Lower bound on scope
    lb_scope = int(math.ceil((1 + math.sqrt(1 + 8 * total_diffs)) / 2))

    best_scope = float('inf')
    best_rows = None

    # Phase 1: Generate initial solutions with best-fit greedy
    seed = 0
    max_initial_seeds = 100  # Increased seeds if greedy is faster
    while time.time() < deadline - 60 and seed < max_initial_seeds:
        rows = _greedy_construct(n_rows, k, seed)
        if rows is None:
            seed += 1
            continue
        scope = _scope(rows)

        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]

        seed += 1

    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k, 0)
        if best_rows is None:
            return [[0, i] for i in range(k + 1)]
        best_scope = _scope(best_rows)

    # Phase 2: Simulated annealing to improve
    if time.time() < deadline - 15:
        improved = _simulated_annealing(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

    # Phase 3: Final polish pass targeting the maximum element
    if time.time() < deadline - 3 and best_rows is not None:
        polished = _polish_max(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            pol_scope = _scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished

    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _all_diffs(rows):
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
    return diffs

def _greedy_construct(n_rows, k, seed):
    rng = random.Random(seed)
    used = set()
    used_bool = [False] * 256  # Fast lookup, max scope likely < 200
    rows = []

    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            limit = 250
            
            best_score = float('inf')
            best_x = None
            candidates = []

            for x in range(last_x + 1, limit + 1):
                valid = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if used_bool[d]:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                    
                # Score: Prefer smaller x, and "dense" diffs
                # Simple score: x + sum of gaps to used?
                # Let's use a simple heuristic: minimize x primarily, 
                # then minimize the max gap introduced?
                # Or just minimize x? 
                # The previous score was fragmentation based.
                # Let's try: score = x * 10 + len([d for d in new_diffs if not (d-1 in used or d+1 in used)])
                frag = 0
                for d in new_diffs:
                    if d > 0 and d < len(used_bool) and not (used_bool[d-1] or (d+1 < len(used_bool) and used_bool[d+1])):
                        frag += 1
                
                score = x + 0.1 * frag
                if score < best_score:
                    best_score = score
                    best_x = x
                    candidates = [(score, x)]
                elif abs(score - best_score) < 1e-9:
                    candidates.append((score, x))

            if best_x is None:
                # Expand search
                for x in range(limit + 1, limit * 3 + 1):
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d >= len(used_bool):
                            # Extend used_bool if needed, or just check set
                            if d in used:
                                valid = False
                                break
                            new_diffs.append(d)
                        else:
                            if used_bool[d]:
                                valid = False
                                break
                            new_diffs.append(d)
                    
                    if not valid:
                        continue
                        
                    frag = 0
                    for d in new_diffs:
                         # Check neighbors in set for large d
                         if not ((d-1) in used or (d+1) in used):
                             frag += 1
                             
                    score = x + 0.1 * frag
                    if score < best_score:
                        best_score = score
                        best_x = x
                        candidates = [(score, x)]
                    elif abs(score - best_score) < 1e-9:
                        candidates.append((score, x))

            if best_x is None:
                return None

            # Pick best candidate to ensure tightness, maybe randomize slightly
            # If we always pick best, we might get stuck in a local optimum quickly.
            # But for initial construction, greedy best is good.
            # Let's pick the best one.
            chosen_x = best_x
            
            row.append(chosen_x)
            for a in row[:-1]:
                d = chosen_x - a
                used.add(d)
                if d < len(used_bool):
                    used_bool[d] = True

        rows.append(row)

    return rows

def _simulated_annealing(initial_rows, n_rows, k, start_time, deadline):
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]

    T_start = 50.0
    T_end = 0.1
    total_steps = 20000
    cool_rate = (T_end / T_start) ** (1.0 / total_steps)

    T = T_start
    steps_since_improve = 0
    max_no_improve = 3000

    # Precompute used diffs for other rows to speed up candidate generation?
    # This is tricky because one row changes.
    # We will compute it inside _generate_candidate_row but optimize that function.

    for step in range(total_steps):
        if time.time() > deadline - 5:
            break

        if rng.random() < 0.4:
            row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        else:
            row_idx = rng.randint(0, n_rows - 1)

        new_row = _generate_candidate_row(rows, row_idx, k, rng)

        if new_row is None:
            continue

        new_rows = rows[:]
        new_rows[row_idx] = new_row
        new_scope = _scope(new_rows)

        all_diffs = _all_diffs(new_rows)
        expected_diffs = n_rows * (k + 1) * k // 2
        if len(all_diffs) != expected_diffs:
            continue

        delta = new_scope - current_scope
        if delta <= 0:
            rows = new_rows
            current_scope = new_scope
            steps_since_improve = 0
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            if T > 1e-9:
                prob = math.exp(-delta / T)
            else:
                prob = 0.0
            if rng.random() < prob:
                rows = new_rows
                current_scope = new_scope
            steps_since_improve += 1

        T *= cool_rate

        if steps_since_improve > max_no_improve:
            pert_idx = rng.randint(0, n_rows - 1)
            pert_row = _generate_candidate_row(rows, pert_idx, k, rng)
            if pert_row is not None:
                test_rows = rows[:]
                test_rows[pert_idx] = pert_row
                if _is_valid(test_rows, n_rows, k):
                    rows = test_rows
                    current_scope = _scope(rows)
            steps_since_improve = 0
            T = max(T, 1.0)

    return best_rows

def _is_valid(rows, n_rows, k):
    all_diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return len(all_diffs) == n_rows * (k + 1) * k // 2

def _generate_candidate_row(rows, exclude_idx, k, rng):
    used_diffs = set()
    used_bool = [False] * 256
    for i, row in enumerate(rows):
        if i == exclude_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                d = row[b] - row[a]
                used_diffs.add(d)
                if d < len(used_bool):
                    used_bool[d] = True

    row = [0]
    for _ in range(k):
        last_x = row[-1]
        limit = 250

        best_score = float('inf')
        best_x = None

        for x in range(last_x + 1, limit + 1):
            valid = True
            new_diffs = []
            for a in row:
                d = x - a
                if d < len(used_bool):
                    if used_bool[d]:
                        valid = False
                        break
                else:
                    if d in used_diffs:
                        valid = False
                        break
                new_diffs.append(d)
            
            if not valid:
                continue

            # Score: minimize x, then fragmentation
            frag = 0
            for d in new_diffs:
                if d < len(used_bool):
                    if not (used_bool[d-1] if d > 0 else False or used_bool[d+1] if d+1 < len(used_bool) else False):
                        frag += 1
                else:
                    if not ((d-1) in used_diffs or (d+1) in used_diffs):
                        frag += 1

            score = x + 0.1 * frag
            if score < best_score:
                best_score = score
                best_x = x

        if best_x is None:
            # Expand
            for x in range(limit + 1, limit * 3 + 1):
                valid = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                frag = 0
                for d in new_diffs:
                    if not ((d-1) in used_diffs or (d+1) in used_diffs):
                        frag += 1

                score = x + 0.1 * frag
                if score < best_score:
                    best_score = score
                    best_x = x

        if best_x is None:
            return None

        row.append(best_x)
        for a in row[:-1]:
            d = best_x - a
            used_diffs.add(d)
            if d < len(used_bool):
                used_bool[d] = True

    return row

def _polish_max(rows, n_rows, k, start_time, deadline):
    rng = random.Random(123)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)

    for iteration in range(150):
        if time.time() > deadline - 3:
            break

        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]

        improved_row = _reduce_max_row(rows, max_idx, k, rng, current_max)

        if improved_row is not None and improved_row[-1] < current_max:
            new_rows = rows[:]
            new_rows[max_idx] = improved_row
            if _is_valid(new_rows, n_rows, k):
                new_scope = _scope(new_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in new_rows]
                    rows = new_rows

    return best_rows if best_scope < _scope([list(r) for r in rows]) else None

def _reduce_max_row(rows, row_idx, k, rng, current_max):
    used_diffs = set()
    used_bool = [False] * 256
    for i, row in enumerate(rows):
        if i == row_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                d = row[b] - row[a]
                used_diffs.add(d)
                if d < len(used_bool):
                    used_bool[d] = True

    for target_max in range(current_max - 1, max(current_max - 10, 0), -1):
        row = [0]
        success = True
        for _ in range(k):
            last_x = row[-1]
            best_score = float('inf')
            best_x = None

            for x in range(last_x + 1, target_max + 1):
                valid = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d < len(used_bool):
                        if used_bool[d]:
                            valid = False
                            break
                    else:
                        if d in used_diffs:
                            valid = False
                            break
                    new_diffs.append(d)
                
                if not valid:
                    continue

                frag = 0
                for d in new_diffs:
                    if d < len(used_bool):
                        if not (used_bool[d-1] if d > 0 else False or used_bool[d+1] if d+1 < len(used_bool) else False):
                            frag += 1
                    else:
                        if not ((d-1) in used_diffs or (d+1) in used_diffs):
                            frag += 1

                score = x + 0.1 * frag
                if score < best_score:
                    best_score = score
                    best_x = x

            if best_x is None:
                success = False
                break

            row.append(best_x)
            for a in row[:-1]:
                d = best_x - a
                used_diffs.add(d)
                if d < len(used_bool):
                    used_bool[d] = True

        if success and len(row) == k + 1:
            return row

    return None
