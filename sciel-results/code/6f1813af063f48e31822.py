import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Greedy baseline
    def greedy(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    best_rows = greedy(n_rows, k)
    best_scope = max(max(r) for r in best_rows)
    
    # Local search: minimize max entry by perturbing rows
    # Represent each row as a list of k+1 sorted integers starting at 0
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def perturb(rows, scope_limit):
        """Create a perturbed version with entries <= scope_limit"""
        new_rows = []
        for row in rows:
            new_row = [0]
            for val in row[1:]:
                # Randomly adjust
                if random.random() < 0.3:
                    delta = random.randint(-3, 3)
                    new_val = val + delta
                else:
                    new_val = val
                # Ensure increasing and within bounds
                if new_val <= new_row[-1]:
                    new_val = new_row[-1] + 1
                if new_val > scope_limit:
                    new_val = scope_limit
                if new_val > new_row[-1]:
                    new_row.append(new_val)
            # Ensure we have k+1 entries
            while len(new_row) < k + 1:
                new_row.append(new_row[-1] + 1)
            new_rows.append(new_row[:k+1])
        return new_rows
    
    def local_search(rows, scope_limit, max_iters=5000):
        """Simulated annealing to find valid config within scope_limit"""
        current = [list(r) for r in rows]
        current_valid = is_valid(current)
        current_scope = get_scope(current)
        
        best = [list(r) for r in current]
        best_scope = current_scope
        best_valid = current_valid
        
        temp = 10.0
        cooling = 0.995
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Perturb
            candidate = perturb(current, scope_limit)
            candidate_scope = get_scope(candidate)
            candidate_valid = is_valid(candidate)
            
            # Score: prefer valid, then lower scope
            if current_valid:
                current_score = candidate_scope - current_scope
            else:
                current_score = 1000 - current_scope
            
            if candidate_valid:
                candidate_score = candidate_scope - scope_limit
            else:
                candidate_score = 1000 - candidate_scope
            
            delta = candidate_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 0.1)):
                current = candidate
                current_valid = candidate_valid
                current_scope = candidate_scope
                
                if candidate_valid and candidate_scope < best_scope:
                    best = [list(r) for r in candidate]
                    best_scope = candidate_scope
                    best_valid = True
            
            temp *= cooling
        
        return best, best_scope, best_valid
    
    # Try decreasing scope limits
    for scope in range(best_scope, max(best_scope - 30, 100), -1):
        if time.time() > deadline - 10:
            break
        
        # Multiple restarts
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            # Start from greedy or best found
            start_rows = [list(r) for r in best_rows]
            # Perturb initial
            for r in start_rows:
                for i in range(1, len(r)):
                    if random.random() < 0.2:
                        r[i] += random.randint(-2, 2)
                    r[i] = max(r[i], r[i-1] + 1)
                    r[i] = min(r[i], scope)
            
            result, result_scope, result_valid = local_search(start_rows, scope, max_iters=3000)
            
            if result_valid and result_scope < best_scope:
                best_rows = result
                best_scope = result_scope
    
    # Final validation
    if is_valid(best_rows):
        return best_rows
    
    return best_rows
