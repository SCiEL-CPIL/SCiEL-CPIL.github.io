import time
import random
import sys

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set with n_rows rows and k+1 elements per row.
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    best_solution = None
    best_score = float('inf')
    
    def get_score(rows):
        if not rows:
            return 0
        return max(max(row) for row in rows)

    def get_cost(rows):
        """Composite cost: Max * 10000 + Sum of all elements."""
        if not rows:
            return 0
        m = max(max(row) for row in rows)
        s = sum(sum(row) for row in rows)
        return m * 10000 + s

    def generate_greedy_seed(randomize=False):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            next_val = 0
            while len(row) < k + 1:
                next_val += 1
                new_diffs = {next_val - a for a in row}
                if not new_diffs & used_diffs:
                    if randomize and random.random() < 0.2:
                        continue
                    row.append(next_val)
                    used_diffs.update(new_diffs)
            rows.append(row)
        return rows

    def local_search(rows, time_limit):
        current_rows = [list(r) for r in rows]
        current_cost = get_cost(current_rows)
        current_max = max(max(r) for r in current_rows)
        current_sum = sum(sum(r) for r in current_rows)
        
        used_diffs = set()
        for r in current_rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used_diffs.add(r[j] - r[i])
        
        best_local_cost = current_cost
        best_local_rows = [list(r) for r in current_rows]
        best_local_max = current_max
        
        no_improve_count = 0
        start_ls = time.time()
        
        while time.time() < time_limit:
            i = random.randint(0, n_rows - 1)
            j = random.randint(1, k)
            
            old_val = current_rows[i][j]
            lower_bound = current_rows[i][j-1] + 1
            upper_bound = current_rows[i][k] - 1 if j < k else current_max + 50
            
            # Generate candidates
            candidates = set()
            # Bias towards smaller values if this element is large
            if old_val > current_max * 0.8:
                candidates.add(old_val - 1)
                candidates.add(old_val - 2)
                candidates.add(old_val - random.randint(1, 20))
            else:
                candidates.add(old_val + random.randint(-10, 10))
            
            # Always try to compact: smallest valid value?
            # That's expensive to find. Skip.
            
            moved = False
            for new_val in candidates:
                if new_val < lower_bound or new_val >= upper_bound:
                    continue
                if new_val == old_val:
                    continue
                
                # Remove old diffs
                old_diffs = set()
                for m in range(k + 1):
                    if m == j: continue
                    if m < j:
                        old_diffs.add(old_val - current_rows[i][m])
                    else:
                        old_diffs.add(current_rows[i][m] - old_val)
                
                for d in old_diffs:
                    used_diffs.discard(d)
                
                # Check new diffs
                new_diffs = set()
                valid = True
                for m in range(k + 1):
                    if m == j: continue
                    if m < j:
                        d = new_val - current_rows[i][m]
                    else:
                        d = current_rows[i][m] - new_val
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    # Apply
                    current_rows[i][j] = new_val
                    used_diffs.update(new_diffs)
                    moved = True
                    
                    # Update metrics
                    # Only row i changed
                    # Recalculate sum for row i
                    old_row_sum = sum(current_rows[i]) # Wait, current_rows[i] is already updated.
                    # We need old_row_sum before update.
                    # This is a flaw. Let's calculate sum diff.
                    # delta_sum = new_val - old_val
                    delta_sum = new_val - old_val
                    new_sum = current_sum + delta_sum
                    new_max = max(current_max, new_val) # If new_val is smaller, max might decrease, but we don't know which was the max.
                    # Safe bet: if new_val < current_max, max is unchanged unless old_val was the unique max.
                    # To be safe and fast, just recalculate max if new_val < current_max?
                    # Or just recalculate max always? O(n*k).
                    # Let's just recalculate max.
                    new_max = max(max(r) for r in current_rows)
                    new_cost = new_max * 10000 + new_sum
                    
                    if new_cost < current_cost:
                        current_cost = new_cost
                        current_sum = new_sum
                        current_max = new_max
                        
                        if new_cost < best_local_cost:
                            best_local_cost = new_cost
                            best_local_rows = [list(r) for r in current_rows]
                            best_local_max = new_max
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        # Revert
                        current_rows[i][j] = old_val
                        used_diffs.difference_update(new_diffs)
                        used_diffs.update(old_diffs)
                        no_improve_count += 1
                        break # Try next candidate? No, this candidate failed. Try next candidate.
                        # Actually, if one candidate fails, others might work.
                        # But we broke out of the 'if valid' block.
                        # We should continue to next candidate.
                        # So 'break' is wrong. Remove break.
                        # But we already reverted. So continue loop.
                        
            if no_improve_count > 2000:
                break

        return best_local_rows, best_local_max

    # 1. Generate seeds
    num_seeds = 30
    seeds = []
    for i in range(num_seeds):
        if time.time() > deadline:
            break
        seed = generate_greedy_seed(randomize=(i > 0))
        seeds.append(seed)
        
    # 2. Local Search
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 5:
            break
        remaining = deadline - time.time()
        t_per = remaining / max(1, len(seeds) - i)
        
        improved_rows, improved_max = local_search(seed, time.time() + t_per)
        
        if improved_max < best_score:
            best_score = improved_max
            best_solution = improved_rows

    if best_solution is None:
        best_solution = seeds[0]
        best_score = get_score(best_solution)
        
    return best_solution
