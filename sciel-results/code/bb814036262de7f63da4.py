import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = 120.0
    
    # Helper to calculate scope and conflicts
    def get_scope_conflicts(rows):
        used = set()
        conflicts = 0
        max_val = 0
        for r in rows:
            max_val = max(max_val, max(r))
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        conflicts += 1
                    else:
                        used.add(d)
        return max_val, conflicts

    # 1. Greedy Initialization
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        break
                    x += 1
                row.append(x)
                used_diffs |= new_diffs
            rows.append(row)
        return rows

    current_rows = greedy_construct()
    best_rows = [r[:] for r in current_rows]
    best_scope = max(max(r) for r in current_rows)
    
    # 2. Iterated Local Search (ILS) with Simulated Annealing
    # We use SA to escape local minima, and if stuck, we perturb and restart SA.
    
    def run_sa(initial_rows, time_limit):
        rows = [r[:] for r in initial_rows]
        scope, conflicts = get_scope_conflicts(rows)
        cost = scope + 1000 * conflicts
        
        T = 20.0
        T_min = 0.5
        cooling = 0.9995
        no_improve_count = 0
        best_local_scope = scope
        best_local_rows = [r[:] for r in rows]
        
        while time.time() < time_limit:
            if T < T_min:
                break
            
            r_idx = random.randint(0, n_rows - 1)
            j_idx = random.randint(1, k)
            
            # Bias towards reducing scope: larger negative deltas are slightly more likely?
            # Or just uniform small delta.
            delta = random.randint(-5, 5)
            
            old_val = rows[r_idx][j_idx]
            new_val = old_val + delta
            
            if new_val <= 0:
                T *= cooling
                continue
            
            # Check bounds
            if j_idx > 0 and new_val <= rows[r_idx][j_idx-1]:
                T *= cooling
                continue
            if j_idx < k and new_val >= rows[r_idx][j_idx+1]:
                T *= cooling
                continue
            
            # Calculate new cost
            cand_rows = [r[:] for r in rows]
            cand_rows[r_idx][j_idx] = new_val
            cand_scope, cand_conflicts = get_scope_conflicts(cand_rows)
            cand_cost = cand_scope + 1000 * cand_conflicts
            
            delta_cost = cand_cost - cost
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                rows = cand_rows
                cost = cand_cost
                scope = cand_scope
                conflicts = cand_conflicts
                
                if conflicts == 0 and scope < best_local_scope:
                    best_local_scope = scope
                    best_local_rows = [r[:] for r in rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                # If we've been stuck for a while, break to allow ILS perturbation
                if no_improve_count > 1000:
                    break
                
            T *= cooling
            
        return best_local_rows, best_local_scope

    # ILS Loop
    # Start with greedy solution
    current_rows = greedy_construct()
    best_scope = max(max(r) for r in current_rows)
    best_rows = [r[:] for r in current_rows]
    
    # Run SA to improve
    current_rows, current_scope = run_sa(current_rows, time.time() + 10)
    if current_scope < best_scope:
        best_scope = current_scope
        best_rows = [r[:] for r in current_rows]
        
    # Iteration
    stagnation_count = 0
    while time.time() < deadline - 5.0:
        # Perturb the current best solution
        # Pick a random row and a random element, change it by a larger amount
        perturbed_rows = [r[:] for r in best_rows]
        
        # Strong perturbation: change 2-5 elements
        num_changes = random.randint(2, 5)
        for _ in range(num_changes):
            r_idx = random.randint(0, n_rows - 1)
            j_idx = random.randint(1, k)
            # Large delta to jump out of local minimum
            delta = random.randint(-20, 20)
            
            old_val = perturbed_rows[r_idx][j_idx]
            new_val = old_val + delta
            
            if new_val <= 0:
                continue
            if j_idx > 0 and new_val <= perturbed_rows[r_idx][j_idx-1]:
                continue
            if j_idx < k and new_val >= perturbed_rows[r_idx][j_idx+1]:
                continue
                
            perturbed_rows[r_idx][j_idx] = new_val
            
        # Run SA on perturbed solution
        # Allocate remaining time minus a safety margin for next iterations
        # Simple heuristic: give 5-10 seconds to each SA run
        sa_time_limit = time.time() + random.uniform(3, 8)
        improved_rows, improved_scope = run_sa(perturbed_rows, sa_time_limit)
        
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = [r[:] for r in improved_rows]
            stagnation_count = 0
        else:
            stagnation_count += 1
            
        # If stagnation is high, we could try a different perturbation strategy
        # but for now, just continue.
        
    return best_rows
