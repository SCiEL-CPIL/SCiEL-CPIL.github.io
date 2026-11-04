import time
import random
from collections import defaultdict

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave buffer for the evaluator
    
    best_result = None
    best_scope = float('inf')
    
    # Heuristic parameters
    SMALL_LIMIT = 40
    
    def get_heuristic_score(new_diffs):
        return sum(1 for d in new_diffs if d <= SMALL_LIMIT)

    def build_row(used_diffs, max_val):
        """
        Recursively builds a single row within max_val.
        Returns a list of integers if successful, None otherwise.
        """
        row = [0]
        
        def backtrack(pos, current_row):
            if pos == k:
                return current_row[:]
            
            last_val = current_row[-1]
            candidates = []
            
            for x in range(last_val + 1, max_val + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                score = get_heuristic_score(new_diffs)
                candidates.append((score, x, new_diffs))
                
                # Limit candidates to keep it fast
                if len(candidates) >= 15:
                    break
            
            if not candidates:
                return None
            
            # Group candidates by score
            score_groups = defaultdict(list)
            for c in candidates:
                score_groups[c[0]].append(c)
                
            # Sort scores
            sorted_scores = sorted(score_groups.keys())
            
            for sc in sorted_scores:
                # Shuffle candidates with this score to add randomness
                random.shuffle(score_groups[sc])
                for score, x, new_diffs in score_groups[sc]:
                    new_used = used_diffs | new_diffs
                    result = backtrack(pos + 1, current_row + [x])
                    if result is not None:
                        return result
            
            return None

        return backtrack(0, row)

    max_restarts = 3000
    
    for _ in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        used_diffs = set()
        rows = []
        success = True
        
        # Target scope for this attempt
        if best_scope < float('inf'):
            target_scope = best_scope - 1
        else:
            target_scope = 120 # Start with a reasonable guess
            
        for r in range(n_rows):
            if time.time() - start_time > time_limit:
                success = False
                break
                
            row = build_row(used_diffs, target_scope)
            
            if row is None:
                success = False
                break
            
            rows.append(row)
            
            # Update used_diffs
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
            
        if success and len(rows) == n_rows:
            current_scope = max(r[-1] for r in rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_result = [r[:] for r in rows]
                if best_scope <= 105:
                    break

    if best_result is None:
        # Fallback greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
        
    return best_result
