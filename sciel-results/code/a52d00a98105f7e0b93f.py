import time
import random
import copy
import math
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0
    
    def greedy_construct():
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

    best_rows = greedy_construct()
    best_scope = max(max(row) for row in best_rows)
    
    current_rows = copy.deepcopy(best_rows)
    
    # Initialize conflict state
    def calculate_initial_state(rows):
        diff_counter = Counter()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diff_counter[row[j] - row[i]] += 1
        conflicts = sum(c - 1 for c in diff_counter.values() if c > 1)
        scope = max(max(row) for row in rows)
        return diff_counter, conflicts, scope

    diff_counter, curr_conflicts, curr_scope = calculate_initial_state(current_rows)
    
    def get_scope(rows):
        return max(max(row) for row in rows)
    
    def try_shrink_all():
        """Try to shrink every element by 1, repeatedly, until no more improvement."""
        nonlocal best_scope, best_rows, curr_scope, curr_conflicts
        progress = True
        while progress and time.time() < deadline:
            progress = False
            for r in range(n_rows):
                for j in range(k, 0, -1):
                    old_val = current_rows[r][j]
                    new_val = old_val - 1
                    
                    if new_val <= current_rows[r][j-1]:
                        continue
                    
                    # Remove old differences
                    old_diffs = []
                    for i in range(j):
                        d = old_val - current_rows[r][i]
                        old_diffs.append(d)
                        diff_counter[d] -= 1
                        if diff_counter[d] == 0:
                            del diff_counter[d]
                    
                    for i in range(j+1, len(current_rows[r])):
                        d = current_rows[r][i] - old_val
                        old_diffs.append(d)
                        diff_counter[d] -= 1
                        if diff_counter[d] == 0:
                            del diff_counter[d]
                    
                    # Apply new value
                    current_rows[r][j] = new_val
                    
                    # Add new differences
                    for i in range(j):
                        d = new_val - current_rows[r][i]
                        diff_counter[d] += 1
                    
                    for i in range(j+1, len(current_rows[r])):
                        d = current_rows[r][i] - new_val
                        diff_counter[d] += 1
                    
                    new_conflicts = sum(c - 1 for c in diff_counter.values() if c > 1)
                    
                    if new_conflicts == 0:
                        new_scope = max(max(row) for row in current_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = copy.deepcopy(current_rows)
                        curr_scope = new_scope
                        curr_conflicts = new_conflicts
                        progress = True
                    else:
                        # Revert
                        current_rows[r][j] = old_val
                        for d in old_diffs:
                            diff_counter[d] += 1
                        for i in range(j):
                            d = new_val - current_rows[r][i]
                            diff_counter[d] -= 1
                            if diff_counter[d] == 0:
                                del diff_counter[d]
                        for i in range(j+1, len(current_rows[r])):
                            d = current_rows[r][i] - new_val
                            diff_counter[d] -= 1
                            if diff_counter[d] == 0:
                                del diff_counter[d]
    
    # Initial shrink pass
    try_shrink_all()
    
    # Local search with restarts
    no_improve_count = 0
    restart_threshold = 5000
    max_restarts = 20
    restarts = 0
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Greedy shrink attempts
        for _ in range(50):
            r = random.randint(0, n_rows - 1)
            j = random.randint(1, k)
            
            old_val = current_rows[r][j]
            new_val = old_val - 1
            
            if new_val <= current_rows[r][j-1]:
                continue
            
            # Remove old differences
            old_diffs = []
            for i in range(j):
                d = old_val - current_rows[r][i]
                old_diffs.append(d)
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            
            for i in range(j+1, len(current_rows[r])):
                d = current_rows[r][i] - old_val
                old_diffs.append(d)
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            
            # Apply new value
            current_rows[r][j] = new_val
            
            # Add new differences
            for i in range(j):
                d = new_val - current_rows[r][i]
                diff_counter[d] += 1
            
            for i in range(j+1, len(current_rows[r])):
                d = current_rows[r][i] - new_val
                diff_counter[d] += 1
            
            new_conflicts = sum(c - 1 for c in diff_counter.values() if c > 1)
            
            if new_conflicts == 0:
                new_scope = max(max(row) for row in current_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = copy.deepcopy(current_rows)
                    improved = True
                    no_improve_count = 0
                curr_scope = new_scope
                curr_conflicts = new_conflicts
                break
            else:
                # Revert
                current_rows[r][j] = old_val
                for d in old_diffs:
                    diff_counter[d] += 1
                for i in range(j):
                    d = new_val - current_rows[r][i]
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                for i in range(j+1, len(current_rows[r])):
                    d = current_rows[r][i] - new_val
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
        
        if improved:
            # After improvement, do a full shrink pass
            try_shrink_all()
            continue
        
        # Phase 2: SA Perturbation with adaptive temperature
        T = max(1.0, 50.0 * (1.0 - no_improve_count / restart_threshold))
        
        r = random.randint(0, n_rows - 1)
        j = random.randint(1, k)
        
        old_val = current_rows[r][j]
        
        min_val = current_rows[r][j-1] + 1
        max_val = max(old_val + 5, best_scope)
        if j < k:
            max_val = min(max_val, current_rows[r][j+1] - 1)
            
        if min_val > max_val:
            no_improve_count += 1
            continue
            
        # Bias towards smaller values
        if random.random() < 0.7:
            new_val = random.randint(min_val, min(old_val, max_val))
        else:
            new_val = random.randint(min_val, max_val)
        
        if new_val == old_val:
            no_improve_count += 1
            continue
            
        # Remove old diffs
        old_diffs = []
        for i in range(j):
            d = old_val - current_rows[r][i]
            old_diffs.append(d)
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
        for i in range(j+1, len(current_rows[r])):
            d = current_rows[r][i] - old_val
            old_diffs.append(d)
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
        
        # Apply new
        current_rows[r][j] = new_val
        
        # Add new diffs
        for i in range(j):
            d = new_val - current_rows[r][i]
            diff_counter[d] += 1
        for i in range(j+1, len(current_rows[r])):
            d = current_rows[r][i] - new_val
            diff_counter[d] += 1
            
        new_scope = max(max(row) for row in current_rows)
        new_conflicts = sum(c - 1 for c in diff_counter.values() if c > 1)
        
        delta_scope = new_scope - curr_scope
        delta_conflicts = new_conflicts - curr_conflicts
        delta_energy = delta_scope + 20 * delta_conflicts
        
        accept = False
        if delta_energy < 0:
            accept = True
        else:
            if T > 0:
                accept = random.random() < math.exp(-delta_energy / T)
        
        if accept:
            curr_scope = new_scope
            curr_conflicts = new_conflicts
            if new_conflicts == 0 and new_scope < best_scope:
                best_scope = new_scope
                best_rows = copy.deepcopy(current_rows)
                improved = True
                no_improve_count = 0
        else:
            # Revert
            current_rows[r][j] = old_val
            for d in old_diffs:
                diff_counter[d] += 1
            for i in range(j):
                d = new_val - current_rows[r][i]
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for i in range(j+1, len(current_rows[r])):
                d = current_rows[r][i] - new_val
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
        
        if not improved:
            no_improve_count += 1
        
        # Restart if stuck
        if no_improve_count > restart_threshold:
            restarts += 1
            if restarts >= max_restarts:
                break
            
            # Restart with greedy + shrink
            current_rows = greedy_construct()
            diff_counter, curr_conflicts, curr_scope = calculate_initial_state(current_rows)
            try_shrink_all()
            no_improve_count = 0

    return best_rows
