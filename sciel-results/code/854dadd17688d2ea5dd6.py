import random
import time
import sys
from itertools import combinations

def construct(n_rows, k):
    """
    Constructs a difference triangle set.
    n_rows: number of rows
    k: parameter such that each row has k+1 elements.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin against 150s kill

    k_plus_1 = k + 1  # Number of elements per row
    num_diffs_per_row = k_plus_1 * (k_plus_1 - 1) // 2

    def get_score(rows):
        if not rows:
            return float('inf')
        return max(max(row) for row in rows)

    def is_valid(rows):
        """Checks if the rows form a valid difference triangle set."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diff = row[j] - row[i]
                    if diff in used:
                        return False
                    used.add(diff)
        return True

    def greedy_solution(seed=0):
        """Generate a solution using randomized greedy with adaptive candidate selection."""
        random.seed(seed)
        used = set()
        rows = []

        for row_idx in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k_plus_1:
                candidates = []
                x = current_max + 1
                limit = x + 1000
                while x <= limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                        if len(candidates) >= 20:
                            break
                    x += 1

                if not candidates:
                    x = current_max + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                            break
                        x += 1

                if len(candidates) <= 3:
                    chosen = random.choice(candidates)
                else:
                    weights = [1.0 / (1.0 + c * 0.02) for c in candidates[:15]]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    chosen = candidates[0]
                    for c, w in zip(candidates[:15], weights):
                        cumulative += w
                        if r <= cumulative:
                            chosen = c
                            break

                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
                current_max = chosen

            rows.append(row)

        return rows

    def local_search(rows, max_time=5.0):
        """Try to improve the solution by reducing the maximum entry."""
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_score = get_score(rows)

        start = time.time()
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])

        improved = True
        stall_count = 0

        while improved and time.time() - start < max_time:
            improved = False

            # Element Moves
            elements = []
            for r in range(n_rows):
                for i in range(len(rows[r])):
                    if i > 0:
                        elements.append((rows[r][i], r, i))
            elements.sort(reverse=True)

            for val, max_r, max_i in elements[:10]:
                if time.time() - start > max_time - 0.1:
                    break

                row = rows[max_r]
                current_val = row[max_i]

                if current_val <= 0:
                    continue

                min_val = row[max_i - 1] + 1
                max_allowed = current_val - 1
                if max_i < len(row) - 1:
                    max_allowed = min(max_allowed, row[max_i + 1] - 1)

                if max_allowed < min_val:
                    continue

                old_diffs = set()
                for a in row:
                    if a != current_val:
                        d = abs(a - current_val)
                        old_diffs.add(d)
                used -= old_diffs

                new_val_found = None
                for v in range(min_val, current_val):
                    valid = True
                    new_diffs = set()
                    for a in row:
                        if a != current_val:
                            d = abs(a - v)
                            if d in used:
                                valid = False
                                break
                            new_diffs.add(d)
                    if valid:
                        new_val_found = v
                        break

                if new_val_found is not None:
                    row[max_i] = new_val_found
                    used |= new_diffs
                    improved = True
                    score = get_score(rows)
                    if score < best_score:
                        best_score = score
                        best_rows = [list(r) for r in rows]
                    break
                else:
                    used |= old_diffs

            if not improved:
                stall_count += 1
                if stall_count > 10:
                    break

        return best_rows

    def clique_search(M, time_limit):
        """
        Search for a clique of size n_rows in the conflict graph.
        Vertices: valid rows (k+1 elements from {0,...,M} starting at 0)
        Edge: two rows conflict if they share any difference.
        We want a set of n_rows mutually non-conflicting rows.
        
        Uses branch and bound with bitset representation.
        """
        # Generate candidate rows
        # A row is a tuple of k+1 increasing integers starting at 0
        # We limit the number of candidates to keep it tractable
        
        candidates = []
        candidate_diffs = []  # set of differences for each candidate
        
        # Generate rows with max element <= M
        # Use a heuristic: generate rows greedily with some randomness
        # But we need a bounded set of candidates
        
        # For M=134, k=7, we need 8 elements from {0,...,134}
        # Number of such subsets is C(134, 7) which is huge
        # So we need to be selective
        
        # Strategy: generate candidate rows using a modified greedy with limited randomness
        # and keep only those with "good" properties (small max, diverse differences)
        
        random.seed(42)
        
        # Generate a pool of candidate rows
        num_candidates = 5000
        seen_rows = set()
        
        for _ in range(num_candidates * 3):
            if len(candidates) >= num_candidates:
                break
            if time.time() > time_limit:
                break
            
            # Generate a random valid row with max <= M
            row = [0]
            current_max = 0
            internal_diffs = set()
            valid = True
            
            for _ in range(k_plus_1 - 1):
                # Choose next element
                x = random.randint(current_max + 1, M)
                new_diffs = {x - a for a in row}
                
                if new_diffs & internal_diffs:
                    valid = False
                    break
                
                internal_diffs.update(new_diffs)
                row.append(x)
                current_max = x
            
            if valid and len(row) == k_plus_1:
                row_tuple = tuple(row)
                if row_tuple not in seen_rows:
                    seen_rows.add(row_tuple)
                    candidates.append(row_tuple)
                    candidate_diffs.append(internal_diffs)
        
        if not candidates:
            return None
        
        # Build conflict graph using bitsets
        # Two candidates conflict if they share any difference
        n_cand = len(candidates)
        
        # For efficiency, use a different approach:
        # Instead of building a full conflict graph, use a backtracking search
        # that checks conflicts on the fly
        
        best_clique = []
        best_max = float('inf')
        
        def backtrack(selected, available, used_diffs, depth):
            nonlocal best_clique, best_max
            
            if time.time() > time_limit:
                return
            
            if depth == n_rows:
                # Found a valid set
                max_entry = max(max(row) for row in selected)
                if max_entry < best_max:
                    best_max = max_entry
                    best_clique = list(selected)
                return
            
            # Pruning: if the minimum possible max is already >= best_max, stop
            # (This is a weak bound, but helpful)
            
            # Try adding candidates from available
            # Heuristic: try candidates with smaller max first
            for idx in list(available):
                if time.time() > time_limit:
                    return
                
                row = candidates[idx]
                diffs = candidate_diffs[idx]
                
                # Check conflict
                if diffs & used_diffs:
                    continue
                
                # Check if this row's max already exceeds best_max
                if max(row) >= best_max:
                    continue
                
                # Add to selection
                selected.append(row)
                new_used = used_diffs | diffs
                new_available = set(available)
                new_available.remove(idx)
                
                # Filter available to those that don't conflict with new_used
                # This is expensive, so we do it lazily
                backtrack(selected, new_available, new_used, depth + 1)
                
                selected.pop()
        
        # Initial available set
        available = set(range(n_cand))
        backtrack([], available, set(), 0)
        
        if best_clique:
            return [list(row) for row in best_clique]
        return None

    # Main search
    best_rows = None
    best_score = float('inf')
    
    # First, get an initial solution with greedy
    for seed in range(10):
        elapsed = time.time() - start_time
        if time.time() - start_time > time_limit - 30:
            break
        
        rows = greedy_solution(seed)
        score = get_score(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Try local search
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10:
        polished = local_search(best_rows, max_time=min(remaining * 0.4, 20.0))
        score = get_score(polished)
        if score < best_score:
            best_score = score
            best_rows = polished
    
    # Try clique search with a bounded M
    remaining = time_limit - (time.time() - start_time)
    if remaining > 20 and best_score > 112:
        # Try to find a better solution using clique search
        # Start with M = best_score - 1 and try to find a clique
        M = best_score - 1
        clique_result = clique_search(M, time_limit=time_limit - (time.time() - start_time) - 5)
        if clique_result is not None:
            score = get_score(clique_result)
            if score < best_score:
                best_score = score
                best_rows = clique_result
    
    return best_rows
