import random
import time
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for 120s limit

    best_rows = None
    best_score = float('inf')

    # Precompute expected number of differences per row
    expected_diffs = (k + 1) * k // 2

    def row_diffs(row):
        """Return set of all positive differences in a row."""
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

    def is_valid_row(row):
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        diffs = row_diffs(row)
        return len(diffs) == expected_diffs

    def validate_solution(rows):
        """Check if a solution is valid (all diffs distinct across rows)."""
        all_diffs = set()
        for r in rows:
            if not is_valid_row(r):
                return False
            d_set = row_diffs(r)
            if all_diffs & d_set:
                return False
            all_diffs.update(d_set)
        return True

    def greedy_construct(used, cap=200, rng=None):
        """Greedy construction of a single row given used differences."""
        if rng is None:
            rng = random
        row = [0]
        for _ in range(k):
            last = row[-1]
            # Find valid candidates
            valids = []
            for x in range(last + 1, cap + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used) and len(new_diffs) == len(row):
                    valids.append(x)
                    if len(valids) > 50:
                        break
            if not valids:
                return None
            
            # Weighted random choice favoring smaller values
            if len(valids) > 1:
                consider = valids[:20]
                weights = [1.0 / (v + 1) ** 1.5 for v in consider]
                total_w = sum(weights)
                r = rng.uniform(0, total_w)
                acc = 0
                chosen = consider[0]
                for v, w in zip(consider, weights):
                    acc += w
                    if r <= acc:
                        chosen = v
                        break
            else:
                chosen = valids[0]
            
            row.append(chosen)
            used.update({chosen - a for a in row[:-1]})
        
        return row

    def build_full_set(rng, cap_hint=None):
        """Build a full set of rows using randomized greedy."""
        used = set()
        rows = []
        for _ in range(n_rows):
            cap = cap_hint if cap_hint else 150
            if best_score < float('inf'):
                cap = min(cap, best_score)
            
            row = greedy_construct(used, cap, rng)
            if row is None:
                return None
            rows.append(row)
        
        return rows

    def local_search(rows, max_time=None):
        """Local search to improve the solution."""
        if max_time is None:
            max_time = time_limit - (time.time() - start_time) - 5
        
        current_max = max(max(r) for r in rows)
        row_diff_sets = [row_diffs(r) for r in rows]
        
        improved = True
        while improved and time.time() - start_time < time_limit - 2:
            improved = False
            row_order = list(range(n_rows))
            random.shuffle(row_order)

            for i in row_order:
                if time.time() - start_time >= time_limit - 2:
                    break
                
                row = rows[i]
                
                # Try to reduce each element
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() - start_time >= time_limit - 2:
                        break
                    
                    current_val = row[pos]
                    min_val = row[pos - 1] + 1
                    max_val = current_val - 1

                    if max_val < min_val:
                        continue

                    # Compute other diffs
                    other_diffs = set()
                    for r_idx in range(n_rows):
                        if r_idx != i:
                            other_diffs.update(row_diff_sets[r_idx])

                    found = False
                    for v in range(min_val, max_val + 1):
                        if v <= row[pos - 1]:
                            continue
                        if pos < len(row) - 1 and v >= row[pos + 1]:
                            continue

                        candidate_row = list(row)
                        candidate_row[pos] = v
                        cand_diffs = row_diffs(candidate_row)

                        if len(cand_diffs) != expected_diffs:
                            continue
                        if cand_diffs & other_diffs:
                            continue

                        rows[i] = candidate_row
                        row_diff_sets[i] = cand_diffs
                        new_max = max(max(r) for r in rows)
                        if new_max < current_max:
                            current_max = new_max
                            improved = True
                        found = True
                        break
                    if found:
                        break

                # Try rebuilding the row
                if not improved and time.time() - start_time < time_limit - 5:
                    other_diffs = set()
                    for r_idx in range(n_rows):
                        if r_idx != i:
                            other_diffs.update(row_diff_sets[r_idx])
                    
                    # Rebuild with DFS
                    new_row = rebuild_row_dfs(other_diffs, current_max)
                    if new_row and max(new_row) < max(rows[i]):
                        rows[i] = new_row
                        row_diff_sets[i] = row_diffs(new_row)
                        current_max = max(max(r) for r in rows)
                        improved = True
                        break

        return rows, current_max

    def rebuild_row_dfs(other_diffs, max_scope):
        """Rebuild a row using DFS to minimize max element."""
        best_row_found = None
        best_max_found = max_scope + 1
        nodes_visited = 0
        max_nodes = 3000
        
        def dfs(row, used):
            nonlocal best_row_found, best_max_found, nodes_visited
            if nodes_visited > max_nodes:
                return
            nodes_visited += 1
            
            if len(row) == k + 1:
                m = max(row)
                if m < best_max_found:
                    best_max_found = m
                    best_row_found = row[:]
                return
            
            last = row[-1]
            if last >= best_max_found:
                return
                
            for x in range(last + 1, best_max_found):
                new_diffs = {x - a for a in row}
                if new_diffs & used or new_diffs & other_diffs:
                    continue
                if len(new_diffs) != len(row):
                    continue
                
                row.append(x)
                used.update(new_diffs)
                
                dfs(row, used)
                
                row.pop()
                used.difference_update(new_diffs)
        
        dfs([0], set())
        return best_row_found

    def simulated_annealing(rows, max_iters=20000):
        """Simulated annealing to escape local minima."""
        current_rows = [list(r) for r in rows]
        current_max = max(max(r) for r in current_rows)
        row_diff_sets = [row_diffs(r) for r in current_rows]
        
        best_rows_sa = [list(r) for r in current_rows]
        best_max_sa = current_max
        
        T = 8.0
        cooling = 0.9998
        
        for it in range(max_iters):
            if time.time() - start_time >= time_limit - 3:
                break
            if T < 0.01:
                break
                
            i = random.randint(0, n_rows - 1)
            pos = random.randint(1, len(current_rows[i]) - 1)
            
            other_diffs = set()
            for r_idx in range(n_rows):
                if r_idx != i:
                    other_diffs.update(row_diff_sets[r_idx])
            
            current_val = current_rows[i][pos]
            min_val = current_rows[i][pos - 1] + 1
            max_val = min(current_val + 3, current_max + 3)
            
            if max_val <= min_val:
                continue
                
            v = random.randint(min_val, max_val)
            if v == current_val:
                continue
            if v <= current_rows[i][pos - 1]:
                continue
            if pos < len(current_rows[i]) - 1 and v >= current_rows[i][pos + 1]:
                continue
            
            candidate_row = list(current_rows[i])
            candidate_row[pos] = v
            cand_diffs = row_diffs(candidate_row)
            
            if len(cand_diffs) != expected_diffs:
                continue
            if cand_diffs & other_diffs:
                continue
            
            old_row = current_rows[i]
            current_rows[i] = candidate_row
            row_diff_sets[i] = cand_diffs
            actual_new_max = max(max(r) for r in current_rows)
            
            delta = actual_new_max - current_max
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                current_max = actual_new_max
                if actual_new_max < best_max_sa:
                    best_max_sa = actual_new_max
                    best_rows_sa = [list(r) for r in current_rows]
            else:
                current_rows[i] = old_row
                row_diff_sets[i] = row_diffs(old_row)
            
            T *= cooling
        
        return best_rows_sa, best_max_sa

    # Main loop
    attempts = 0
    while time.time() - start_time < time_limit:
        rows = build_full_set(random, cap_hint=120 if best_score == float('inf') else best_score)
        if rows is not None:
            score = max(max(r) for r in rows)
            if score < best_score:
                best_score = score
                best_rows = [list(r) for r in rows]
                
                # Local search
                if time.time() - start_time < time_limit - 10:
                    improved_rows, improved_score = local_search(best_rows)
                    if validate_solution(improved_rows):
                        if improved_score < best_score:
                            best_score = improved_score
                            best_rows = improved_rows
                
                # Simulated annealing
                if time.time() - start_time < time_limit - 15:
                    sa_rows, sa_score = simulated_annealing(best_rows)
                    if validate_solution(sa_rows):
                        if sa_score < best_score:
                            best_score = sa_score
                            best_rows = sa_rows
                
                if best_score <= 105:
                    break

        attempts += 1
        if attempts % 30 == 0:
            if best_rows is not None and time.time() - start_time < time_limit - 20:
                # Try to improve with a different order
                shuffled = list(best_rows)
                random.shuffle(shuffled)
                improved_rows, improved_score = local_search(shuffled)
                if validate_solution(improved_rows):
                    if improved_score < best_score:
                        best_score = improved_score
                        best_rows = improved_rows
                        if time.time() - start_time < time_limit - 25:
                            sa_rows, sa_score = simulated_annealing(best_rows)
                            if validate_solution(sa_rows):
                                if sa_score < best_score:
                                    best_score = sa_score
                                    best_rows = sa_rows

    if best_rows is None:
        # Fallback: deterministic construction
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        break
                    x += 1
                row.append(x)
                used.update({x - a for a in row[:-1]})
            rows.append(row)
        best_rows = rows

    return best_rows
