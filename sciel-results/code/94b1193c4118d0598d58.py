import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    def get_max_scope(rows):
        return max(max(r) for r in rows)
    
    # Helper to get involved diffs
    def get_involved_diffs(row, pos, val):
        diffs = set()
        for i in range(pos):
            diffs.add(val - row[i])
        for j in range(pos + 1, len(row)):
            diffs.add(row[j] - val)
        return diffs
    
    # Generate greedy solution
    def generate_greedy():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows, used_diffs
    
    # Try multiple greedy orderings and keep the best
    best_rows = None
    best_scope = float('inf')
    best_used_diffs = None
    
    # Try different row orderings
    for attempt in range(3):
        if time.time() > deadline - 60:
            break
        rows, used_diffs = generate_greedy()
        scope = get_max_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
            best_used_diffs = used_diffs.copy()
    
    if best_rows is None:
        best_rows, best_used_diffs = generate_greedy()
        best_scope = get_max_scope(best_rows)
    
    rows = best_rows
    used_diffs = best_used_diffs
    current_scope = best_scope
    
    # Phase 1: Aggressive Hill Climbing with multiple passes
    def hill_climb_pass(rows, used_diffs, current_scope, max_iterations=1000):
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline - 10:
            improved = False
            iterations += 1
            for r_idx in range(n_rows):
                if time.time() > deadline - 10:
                    break
                row = rows[r_idx]
                for pos in range(1, k + 1):
                    old_val = row[pos]
                    low = row[pos - 1] + 1
                    high = row[pos + 1] - 1 if pos < k else old_val
                    
                    # Try to reduce value
                    for new_val in range(low, old_val):
                        old_involved = get_involved_diffs(row, pos, old_val)
                        used_diffs.difference_update(old_involved)
                        
                        new_involved = get_involved_diffs(row, pos, new_val)
                        
                        if len(new_involved) != len(old_involved):
                            used_diffs |= old_involved
                            continue
                        
                        if new_involved & used_diffs:
                            used_diffs |= old_involved
                            continue
                        
                        # Valid
                        used_diffs |= new_involved
                        row[pos] = new_val
                        
                        new_scope = get_max_scope(rows)
                        if new_scope < current_scope:
                            current_scope = new_scope
                            improved = True
                            # Continue scanning to find more improvements
                        else:
                            # Revert
                            used_diffs.difference_update(new_involved)
                            used_diffs |= old_involved
                            row[pos] = old_val
                    if improved:
                        break
                if improved:
                    break
        return rows, used_diffs, current_scope
    
    # Run multiple hill climbing passes
    for _ in range(5):
        if time.time() > deadline - 15:
            break
        rows, used_diffs, current_scope = hill_climb_pass(rows, used_diffs, current_scope)
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [r[:] for r in rows]
    
    # Phase 2: Simulated Annealing with restarts
    T = 5.0
    cooling = 0.995
    sa_restarts = 3
    
    for restart in range(sa_restarts):
        if time.time() > deadline - 5:
            break
        
        if restart > 0:
            # Restart from best known solution with some perturbation
            rows = [r[:] for r in best_rows]
            used_diffs = set()
            for r in rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        used_diffs.add(r[j] - r[i])
            current_scope = get_max_scope(rows)
            
            # Perturb: randomly move a few positions
            for _ in range(3):
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                row = rows[r_idx]
                old_val = row[pos]
                low = row[pos - 1] + 1
                high = row[pos + 1] - 1 if pos < k else old_val + 20
                
                if low > high:
                    continue
                    
                # Try random values in range
                candidates = []
                for v in range(low, high + 1):
                    if v == old_val:
                        continue
                    old_involved = get_involved_diffs(row, pos, old_val)
                    new_involved = get_involved_diffs(row, pos, v)
                    if len(new_involved) == len(old_involved):
                        # Check if valid
                        temp_used = used_diffs - old_involved
                        if not (new_involved & temp_used):
                            candidates.append(v)
                
                if candidates:
                    new_val = random.choice(candidates)
                    old_involved = get_involved_diffs(row, pos, old_val)
                    new_involved = get_involved_diffs(row, pos, new_val)
                    used_diffs.difference_update(old_involved)
                    used_diffs |= new_involved
                    row[pos] = new_val
        
        T = 5.0
        steps = 0
        max_steps = 5000
        
        while steps < max_steps and time.time() < deadline - 2:
            steps += 1
            
            # Pick a random row
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            
            # Pick a random position
            pos = random.randint(1, k)
            old_val = row[pos]
            
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < k else old_val + 30
            
            # Bias: if this is the max, try smaller. Else, try nearby.
            if old_val == max(row):
                if low < old_val:
                    # Strong bias toward reducing
                    if random.random() < 0.7:
                        new_val = random.randint(low, max(low, old_val - 15))
                    else:
                        new_val = random.randint(low, old_val - 1)
                else:
                    continue
            else:
                # Random neighborhood with wider range
                search_low = max(low, old_val - 15)
                search_high = min(high, old_val + 15)
                if search_low > search_high:
                    continue
                new_val = random.randint(search_low, search_high)
                
            if new_val == old_val:
                continue
                
            old_involved = get_involved_diffs(row, pos, old_val)
            used_diffs.difference_update(old_involved)
            
            new_involved = get_involved_diffs(row, pos, new_val)
            
            if len(new_involved) != len(old_involved):
                used_diffs |= old_involved
                continue
            
            if new_involved & used_diffs:
                used_diffs |= old_involved
                continue
            
            # Valid move
            used_diffs |= new_involved
            row[pos] = new_val
            
            new_scope = get_max_scope(rows)
            
            if new_scope < current_scope:
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            else:
                delta = new_scope - current_scope
                # Accept with probability based on temperature
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    current_scope = new_scope
                else:
                    # Revert
                    used_diffs.difference_update(new_involved)
                    used_diffs |= old_involved
                    row[pos] = old_val
            
            T *= cooling
            if T < 0.05:
                T = 0.05
    
    # Final polish: quick hill climb from best solution
    if time.time() < deadline - 2:
        final_rows = [r[:] for r in best_rows]
        final_used = set()
        for r in final_rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    final_used.add(r[j] - r[i])
        final_scope = get_max_scope(final_rows)
        
        # Quick local search
        for _ in range(200):
            if time.time() > deadline - 1:
                break
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            row = final_rows[r_idx]
            old_val = row[pos]
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < k else old_val
            
            for new_val in range(low, old_val):
                old_involved = get_involved_diffs(row, pos, old_val)
                final_used.difference_update(old_involved)
                new_involved = get_involved_diffs(row, pos, new_val)
                
                if len(new_involved) != len(old_involved):
                    final_used |= old_involved
                    continue
                if new_involved & final_used:
                    final_used |= old_involved
                    continue
                
                final_used |= new_involved
                row[pos] = new_val
                new_scope = get_max_scope(final_rows)
                
                if new_scope < final_scope:
                    final_scope = new_scope
                    break
                else:
                    final_used.difference_update(new_involved)
                    final_used |= old_involved
                    row[pos] = old_val
        
        if final_scope < best_scope:
            best_scope = final_scope
            best_rows = final_rows
    
    return best_rows
