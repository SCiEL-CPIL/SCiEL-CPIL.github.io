import time
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for 120s limit
    
    # Helper to calculate differences of a row
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper to check if a row is internally valid
    def is_valid_row(row):
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        diffs = get_diffs(row)
        expected_diffs = len(row) * (len(row) - 1) // 2
        return len(diffs) == expected_diffs

    # Generate all valid rows up to a given scope
    def generate_valid_rows(scope):
        valid_rows = []
        # We need to generate sequences of length k+1 starting with 0
        # The remaining k elements must be chosen from 1..scope
        # such that the row is strictly increasing and differences are unique
        
        # Backtracking
        def backtrack(current_row, used_diffs):
            if len(current_row) == k + 1:
                if is_valid_row(current_row):
                    valid_rows.append(list(current_row))
                return
            
            last_val = current_row[-1]
            start_val = last_val + 1
            # Optimization: The next value cannot be too large if we need to fit more elements
            # But since we are bounding by 'scope', we just iterate up to scope
            
            # Pruning: If we have already used 'scope' and still need more elements, 
            # but the max possible value is scope, we can't extend if current_row is too short?
            # Actually, just iterate.
            
            for next_val in range(start_val, scope + 1):
                new_diffs = {next_val - v for v in current_row}
                if new_diffs & used_diffs:
                    continue
                # Also check internal uniqueness of new_diffs itself? 
                # No, get_diffs handles the whole row. But we can prune early.
                # The new differences must be unique among themselves and with used_diffs.
                if len(new_diffs) != len(current_row):
                    continue
                
                backtrack(current_row + [next_val], used_diffs | new_diffs)
        
        backtrack([0], set())
        return valid_rows

    # Branch and Bound for Maximum Clique with Min Max Element
    def solve_clique(candidates, target_size):
        # candidates: list of (row, diff_set, max_val)
        # We want a subset of size target_size such that all diff_sets are disjoint
        # And we want to minimize the max_val of the chosen rows.
        
        if not candidates:
            return None, float('inf')
        
        # Sort by max_val to allow early termination
        candidates.sort(key=lambda x: x[2])
        
        n = len(candidates)
        if n < target_size:
            return None, float('inf')
        
        # Precompute compatibility bitmasks or just use sets for small n
        # Since n_rows is small (7) and k is small (5), the number of candidates might be large,
        # but the clique size is small.
        
        best_clique = None
        best_max_val = float('inf')
        
        # Recursion
        def dfs(start_idx, current_clique, current_diffs, current_max):
            nonlocal best_clique, best_max_val
            
            if len(current_clique) == target_size:
                if current_max < best_max_val:
                    best_max_val = current_max
                    best_clique = list(current_clique)
                return
            
            # Pruning: If we can't reach target_size with remaining nodes
            remaining = n - start_idx
            if len(current_clique) + remaining < target_size:
                return
            
            # Pruning: If current_max is already >= best_max_val, we might still find a better one?
            # No, if we are minimizing max, and current_max is already bad, 
            # adding more nodes will only keep or increase max.
            # So if current_max >= best_max_val, we can prune.
            if current_max >= best_max_val:
                return
            
            # Iterate through candidates
            for i in range(start_idx, n):
                row, diffs, max_val = candidates[i]
                
                # Check compatibility
                if diffs & current_diffs:
                    continue
                
                new_max = max(current_max, max_val)
                if new_max >= best_max_val:
                    # Since candidates are sorted by max_val, all subsequent j > i will have max_val >= max_val
                    # So new_max will be >= new_max >= best_max_val.
                    # We can break here if we are strictly minimizing and sorted.
                    # However, a later node might have a smaller max_val? No, sorted ascending.
                    # So if max_val >= best_max_val, we can break.
                    # But wait, if current_max is large, new_max is large.
                    # If max_val < best_max_val, we proceed.
                    # If max_val >= best_max_val, then new_max >= best_max_val.
                    # So we can break.
                    break
                
                current_clique.append(row)
                dfs(i + 1, current_clique, current_diffs | diffs, new_max)
                current_clique.pop()
                
                # Time check
                if time.time() - start_time > time_limit - 2:
                    raise TimeoutError()
        
        dfs(0, [], set(), 0)
        return best_clique, best_max_val

    # Heuristic Local Search (Fallback / Polish)
    def local_search(rows):
        current_max = max(max(r) for r in rows)
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(n_rows):
                row = rows[i]
                for pos in range(len(row) - 1, 0, -1):
                    current_val = row[pos]
                    min_val = row[pos-1] + 1
                    max_val = current_val - 1
                    if max_val < min_val:
                        continue
                    other_diffs = set()
                    for r_idx, r in enumerate(rows):
                        if r_idx != i:
                            other_diffs.update(get_diffs(r))
                    for v in range(min_val, max_val + 1):
                        candidate_row = list(row)
                        candidate_row[pos] = v
                        if not is_valid_row(candidate_row):
                            continue
                        cand_diffs = get_diffs(candidate_row)
                        if cand_diffs & other_diffs:
                            continue
                        rows[i] = candidate_row
                        new_max = max(max(r) for r in rows)
                        if new_max < current_max:
                            current_max = new_max
                            improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
        return rows, current_max

    # Main Logic
    best_rows = None
    best_score = float('inf')
    
    # Phase 1: Exact Search with increasing scope
    # The counting bound is 105. Best known is 112.
    # We try scopes from 105 upwards.
    
    # To avoid generating too many rows, we cap the scope for the exact search.
    # If scope is too high, the number of valid rows explodes.
    # For k=5, length 6.
    # Let's try exact search for scopes 105 to 120.
    
    for scope in range(105, 125):
        if time.time() - start_time > time_limit - 10:
            break
            
        try:
            candidates = generate_valid_rows(scope)
            # Filter candidates to only keep those that are "promising" or all of them?
            # For small scopes, the number of candidates is manageable.
            # For scope 112, how many valid rows?
            # Let's just run the solver.
            
            clique, score = solve_clique(candidates, n_rows)
            if clique is not None:
                # Validate
                all_diffs = set()
                valid = True
                for r in clique:
                    d = get_diffs(r)
                    if all_diffs & d:
                        valid = False
                        break
                    all_diffs.update(d)
                
                if valid:
                    best_rows = clique
                    best_score = score
                    # We found a solution with this scope. 
                    # Since we are increasing scope, this is the minimum scope for this candidate set.
                    # But wait, generate_valid_rows(scope) only generates rows with max element <= scope.
                    # So if we find a clique, its max element is <= scope.
                    # We want the minimum max element.
                    # The solver minimizes the max element of the chosen rows.
                    # So 'score' is the actual max element.
                    # If score < best_score, update.
                    # Actually, if we find a clique at scope S, the max element is <= S.
                    # We should break if we find a solution?
                    # No, a solution at scope 105 is better than at 106.
                    # So if we find one at 105, we can return immediately?
                    # Yes, because we are iterating scopes in increasing order.
                    # If a solution exists with max element <= 105, we will find it in the scope=105 iteration.
                    # The solver finds the clique with the minimum max element among those with max element <= scope.
                    # So if we find a clique at scope=105, its max element is the minimal possible for rows restricted to <=105.
                    # Is it the global minimum?
                    # If the global minimum is 104, we would have found it at scope=104.
                    # Since we start at 105, we assume the lower bound is 105.
                    # If we find a solution, it is the best possible for the current scope.
                    # Since we go up, the first solution found is the global minimum (assuming our lower bound is correct).
                    
                    # However, the counting bound is 105. It might be that no solution exists at 105.
                    # So we continue to 106, etc.
                    
                    # Once we find a valid clique, it is the optimal for the current scope.
                    # Since we iterate scopes 105, 106, ..., the first valid clique found has the smallest max element.
                    # So we can return immediately?
                    # Yes, provided we trust the solver.
                    
                    # Let's do a quick local search polish just in case, but the clique is already optimal for its set.
                    # Actually, the clique search is exact for the given candidate set.
                    # So no local search needed for the exact part.
                    
                    return clique
                    
        except TimeoutError:
            break
        except Exception:
            # If exact search fails or errors, fall back to heuristic
            pass

    # Phase 2: Heuristic Fallback
    # If exact search didn't find anything (or timed out), use randomized greedy + local search.
    
    def randomized_greedy(max_scope_hint=None):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                valids = []
                last = row[-1]
                cap = max_scope_hint if max_scope_hint else 200
                if best_score < float('inf'):
                    cap = min(cap, best_score)
                
                for x in range(last + 1, cap + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        valids.append(x)
                
                if not valids:
                    return None
                
                if len(valids) > 1:
                    weights = [1.0 / (v + 1) for v in valids[:20]]
                    total_w = sum(weights)
                    r = random.uniform(0, total_w)
                    acc = 0
                    chosen = valids[0]
                    for v, w in zip(valids[:20], weights):
                        acc += w
                        if r <= acc:
                            chosen = v
                            break
                else:
                    chosen = valids[0]
                
                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
            rows.append(row)
        return rows

    attempts = 0
    while time.time() - start_time < time_limit - 5:
        rows = randomized_greedy()
        if rows is not None:
            score = max(max(r) for r in rows)
            if score < best_score:
                best_score = score
                best_rows = [list(r) for r in rows]
                # Local search
                if time.time() - start_time < time_limit - 10:
                    improved_rows, improved_score = local_search(best_rows)
                    if improved_score < best_score:
                        all_diffs = set()
                        valid = True
                        for r in improved_rows:
                            d_set = get_diffs(r)
                            if all_diffs & d_set:
                                valid = False
                                break
                            all_diffs.update(d_set)
                        if valid:
                            best_score = improved_score
                            best_rows = improved_rows
        attempts += 1
        if attempts % 50 == 0 and best_rows:
            # Shuffle and polish
            shuffled = list(best_rows)
            random.shuffle(shuffled)
            improved_rows, improved_score = local_search(shuffled)
            if improved_score < best_score:
                all_diffs = set()
                valid = True
                for r in improved_rows:
                    d_set = get_diffs(r)
                    if all_diffs & d_set:
                        valid = False
                        break
                    all_diffs.update(d_set)
                if valid:
                    best_score = improved_score
                    best_rows = improved_rows

    if best_rows is None:
        # Deterministic greedy fallback
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
