import time
import random
import math
import sys

def construct(n_rows, k):
    start = time.time()
    deadline = start + 145  # Safety margin before 150s limit
    
    # Total possible differences: max difference is bounded
    # For k+1 elements starting at 0, max diff <= cap
    # We'll use a reasonable cap estimate
    max_cap = 200  # Upper bound for differences
    
    def row_diffs(row):
        """Return set of all positive differences in row"""
        out = set()
        for i in range(len(row)):
            a = row[i]
            for j in range(i + 1, len(row)):
                out.add(row[j] - a)
        return out
    
    def row_diffs_list(row):
        """Return list of all positive differences in row (for bitset)"""
        out = []
        for i in range(len(row)):
            a = row[i]
            for j in range(i + 1, len(row)):
                out.append(row[j] - a)
        return out
    
    def greedy_row(used_set, k, cap=None, randomize=False, rng=None):
        """Greedy construction of a single row avoiding used differences"""
        if rng is None:
            rng = random
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if cap is not None and x > cap:
                return None
            new = {x - a for a in row}
            if not (new & used_set):
                row.append(x)
                used_set |= new
        return row
    
    def dfs_row(row, used_set, cap, row_deadline, randomize=False, rng=None):
        """DFS to complete a row with pruning"""
        if time.time() > row_deadline or time.time() > deadline:
            return None
        if len(row) == k + 1:
            return row[:]
        
        last = row[-1]
        needed = (k + 1) - len(row)
        # Upper bound: we need 'needed' more elements, each at least 1 apart
        upper = cap - (needed - 1)
        if upper < last + 1:
            return None
        
        candidates = list(range(last + 1, upper + 1))
        if randomize and rng is not None:
            rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() > row_deadline or time.time() > deadline:
                return None
            new = {x - a for a in row}
            if new & used_set:
                continue
            res = dfs_row(row + [x], used_set | new, cap, row_deadline, randomize, rng)
            if res is not None:
                return res
        return None
    
    def validate_solution(rows):
        """Validate that all rows are valid and differences are distinct"""
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i+1]:
                    return False
            diffs = row_diffs(row)
            if diffs & all_diffs:
                return False
            all_diffs |= diffs
        return True
    
    def compute_difficulty(rows, used_set):
        """Compute difficulty score for each row based on available extensions"""
        difficulties = []
        for i, row in enumerate(rows):
            # Count how many new differences this row uses
            row_diffs_set = row_diffs(row)
            # Difficulty: how constrained is this row?
            # More differences used = harder to replace
            difficulties.append(len(row_diffs_set))
        return difficulties
    
    # Phase 1: Greedy Baseline with difficulty ordering
    used = set()
    rows = []
    
    # First pass: greedy construction
    for _ in range(n_rows):
        row = greedy_row(set(used), k)
        if row is None:
            row = greedy_row(set(used), k, randomize=True)
        if row is None:
            return [[0] * (k+1) for _ in range(n_rows)]
        rows.append(row)
        used |= row_diffs(row)
    
    current_max = max(r[-1] for r in rows)
    best_rows = rows[:]
    best_max = current_max
    
    # Phase 2: Multi-start Simulated Annealing with difficulty-ordered replacement
    rng = random.Random(42)
    
    def generate_candidate(i, cap, randomize=True, time_budget=0.3):
        """Generate a candidate row for position i"""
        used_without_i = set(used)
        used_without_i.difference_update(row_diffs(rows[i]))
        
        row_deadline = min(deadline - 5, time.time() + time_budget)
        cand = dfs_row([0], used_without_i, cap, row_deadline, randomize=randomize, rng=rng)
        return cand
    
    def try_improve(rows_copy, used_copy, current_max_copy, time_budget=60):
        """Run SA on a copy and return best found"""
        rows_local = rows_copy[:]
        used_local = set(used_copy)
        current_max_local = current_max_copy
        best_local_max = current_max_local
        best_local_rows = rows_local[:]
        
        T = 12.0
        cooling = 0.998
        no_improve = 0
        
        sa_start = time.time()
        sa_deadline = sa_start + time_budget
        
        while time.time() < sa_deadline - 2:
            if T < 0.05:
                break
            
            i = rng.randint(0, n_rows - 1)
            
            # Choose cap: usually current max, sometimes slightly larger for exploration
            cap = current_max_local
            if rng.random() < 0.15:
                cap = current_max_local + rng.randint(1, 8)
            
            cand = generate_candidate(i, cap, randomize=True, time_budget=0.25)
            
            if cand is not None:
                new_last = cand[-1]
                max_others = max((rows_local[j][-1] for j in range(n_rows) if j != i), default=0)
                new_global_max = max(max_others, new_last)
                delta = new_global_max - current_max_local
                
                if delta < 0:
                    accept = True
                else:
                    try:
                        prob = math.exp(-delta / T)
                    except:
                        prob = 0.0
                    accept = rng.random() < prob
                
                if accept:
                    used_local.difference_update(row_diffs(rows_local[i]))
                    rows_local[i] = cand
                    used_local |= row_diffs(cand)
                    current_max_local = new_global_max
                    
                    if current_max_local < best_local_max:
                        best_local_max = current_max_local
                        best_local_rows = rows_local[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    if no_improve > 300:
                        no_improve = 0
                        # Perturb: reset 1-2 rows
                        for _ in range(rng.randint(1, 2)):
                            j = rng.randint(0, n_rows - 1)
                            used_local.difference_update(row_diffs(rows_local[j]))
                            new_row = greedy_row(set(used_local), k, randomize=True, rng=rng)
                            if new_row is not None:
                                rows_local[j] = new_row
                                used_local |= row_diffs(new_row)
                                current_max_local = max(r[-1] for r in rows_local)
            
            T *= cooling
        
        return best_local_rows, best_local_max
    
    # Run multiple SA starts with different seeds
    remaining_time = deadline - 20 - (time.time() - start)
    n_starts = max(1, min(4, int(remaining_time / 30)))
    
    for s in range(n_starts):
        if time.time() > deadline - 20:
            break
        
        seed = 42 + s * 1000
        rng_local = random.Random(seed)
        
        # Create a perturbed starting point
        rows_perturbed = best_rows[:]
        used_perturbed = set()
        for r in rows_perturbed:
            used_perturbed |= row_diffs(r)
        
        # Perturb 1-2 rows
        for _ in range(rng_local.randint(1, 2)):
            idx = rng_local.randint(0, n_rows - 1)
            used_perturbed.difference_update(row_diffs(rows_perturbed[idx]))
            new_row = greedy_row(set(used_perturbed), k, randomize=True, rng=rng_local)
            if new_row is not None:
                rows_perturbed[idx] = new_row
                used_perturbed |= row_diffs(new_row)
        
        current_max_perturbed = max(r[-1] for r in rows_perturbed)
        
        # Run SA on this start
        time_for_sa = max(5, min(30, (deadline - 20 - (time.time() - start)) / max(1, n_starts - s)))
        rows_result, max_result = try_improve(rows_perturbed, used_perturbed, current_max_perturbed, time_budget=time_for_sa)
        
        if max_result < best_max:
            best_max = max_result
            best_rows = rows_result[:]
    
    # Phase 3: Final Polish - try to shrink each row in best_rows
    final_rows = best_rows[:]
    final_used = set()
    for r in final_rows:
        final_used |= row_diffs(r)
    
    for i in range(n_rows):
        if time.time() > deadline - 5:
            break
        cur_last = final_rows[i][-1]
        if cur_last <= 0:
            continue
        
        used_without = set(final_used)
        used_without.difference_update(row_diffs(final_rows[i]))
        
        found = None
        # Try to find smaller rows with decreasing caps
        for cap in range(cur_last - 1, max(0, cur_last - 15), -1):
            if time.time() > deadline - 5:
                break
            row_deadline = min(deadline - 5, time.time() + 0.5)
            cand = dfs_row([0], used_without, cap, row_deadline, randomize=False, rng=rng)
            if cand is not None:
                found = cand
                break
        
        if found is not None:
            final_used.difference_update(row_diffs(final_rows[i]))
            final_rows[i] = found
            final_used |= row_diffs(found)
    
    # Final validation
    if not validate_solution(final_rows):
        if validate_solution(best_rows):
            return best_rows
        # Last resort: return greedy baseline
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(set(used), k)
            rows.append(row)
            used |= row_diffs(row)
        return rows
    
    return final_rows
