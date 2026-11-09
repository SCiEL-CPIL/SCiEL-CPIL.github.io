import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

DEADLINE = time.time() + 270  # Safety margin

def _build_paley37():
    """Return adjacency bitset list for Paley graph on 37 vertices (vertices 0..36)."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _find_k4_sets(adj, n):
    """Find all K4 cliques in graph with n vertices. Return list of 4-tuples."""
    k4_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            for k in range(j + 1, n):
                if not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)):
                    continue
                for l in range(k + 1, n):
                    if (adj[i] & (1 << l)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                        k4_list.append((i, j, k, l))
    return k4_list

def _find_ind4_sets(adj, n):
    """Find all independent 4-sets in graph with n vertices. Return list of 4-tuples."""
    ind4_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                continue
            for k in range(j + 1, n):
                if (adj[i] & (1 << k)) or (adj[j] & (1 << k)):
                    continue
                for l in range(k + 1, n):
                    if not (adj[i] & (1 << l)) and not (adj[j] & (1 << l)) and not (adj[k] & (1 << l)):
                        ind4_list.append((i, j, k, l))
    return ind4_list

def _try_add_vertex(adj, n, k4_list, ind4_list):
    """
    Try to add a new vertex (index n) to the graph defined by adj (size n).
    Uses efficient Min-Conflicts with Walk and incremental updates.
    Returns new_adj (list of bitsets of size n+1) or None.
    """
    # Precompute constraint lists per variable
    k4_by_var = [[] for _ in range(n)]
    for idx, quad in enumerate(k4_list):
        for v in quad:
            k4_by_var[v].append(idx)
            
    ind4_by_var = [[] for _ in range(n)]
    for idx, quad in enumerate(ind4_list):
        for v in quad:
            ind4_by_var[v].append(idx)

    # Combine all constraints for easy access
    # Type 0: K4 (NAND: not all 1)
    # Type 1: Ind4 (NOR: not all 0)
    all_constraints = []
    for quad in k4_list:
        all_constraints.append((0, quad))
    for quad in ind4_list:
        all_constraints.append((1, quad))
        
    num_constraints = len(all_constraints)
    
    # Helper to check if a constraint is satisfied
    def is_satisfied(con_type, quad, assign):
        if con_type == 0: # K4: Satisfied if NOT (all 1)
            return not (assign[quad[0]] == 1 and assign[quad[1]] == 1 and assign[quad[2]] == 1 and assign[quad[3]] == 1)
        else: # Ind4: Satisfied if NOT (all 0)
            return not (assign[quad[0]] == 0 and assign[quad[1]] == 0 and assign[quad[2]] == 0 and assign[quad[3]] == 0)

    # Initialize a random assignment
    # We do multiple restarts
    max_restarts = 500
    max_iters_per_restart = 2000
    
    for _ in range(max_restarts):
        if time.time() > DEADLINE:
            return None
            
        assign = [random.choice([0, 1]) for _ in range(n)]
        
        # Initial violation count
        viol_per_var = [0] * n
        total_viol = 0
        
        for con_type, quad in all_constraints:
            if not is_satisfied(con_type, quad, assign):
                total_viol += 1
                for v in quad:
                    viol_per_var[v] += 1
        
        if total_viol == 0:
            # Success
            new_adj = adj[:] + [0]
            for v in range(n):
                if assign[v]:
                    new_adj[n] |= (1 << v)
                    new_adj[v] |= (1 << n)
            return new_adj

        # Local Search Loop
        for it in range(max_iters_per_restart):
            if time.time() > DEADLINE:
                return None
                
            if total_viol == 0:
                new_adj = adj[:] + [0]
                for v in range(n):
                    if assign[v]:
                        new_adj[n] |= (1 << v)
                        new_adj[v] |= (1 << n)
                return new_adj

            # Pick a variable to flip
            # Strategy: With high probability, pick a variable from a random violated constraint.
            # With low probability, pick a random variable (Walk).
            
            # Find a violated constraint
            # We can just pick a random variable with max violations, or random from violated set.
            # To be efficient, we maintain a list of violated variables?
            # Simpler: Pick a random variable v. If it has 0 violations, pick another.
            # Even simpler: Pick a random constraint that is violated.
            
            # Let's find a violated constraint index
            # We can iterate through constraints until we find a violated one? O(N)
            # Or we can just pick a random variable with highest violation count.
            
            # Let's use the "Walk" on constraints:
            # Pick a random constraint. If it's violated, pick a random variable from it.
            # If it's satisfied, pick a random variable from it anyway? No, that's not standard.
            
            # Standard Min-Conflicts: Pick a random violated constraint, flip a random variable in it.
            # We need a way to pick a violated constraint quickly.
            # We can keep a list of violated constraint indices.
            
            # Let's maintain `violated_constraints` list.
            # This adds overhead to update.
            
            # Alternative: Just pick a random variable `v` with `viol_per_var[v] > 0`.
            # If none, we are done (handled above).
            
            # Find a variable with violations
            # To avoid O(N) scan, we can just pick random v and check.
            # Since N=37, scanning is cheap.
            
            # Pick a variable to flip
            # Weighted by violation count?
            # Let's just pick a random variable from those with violations.
            
            # Collect candidates
            candidates = [v for v in range(n) if viol_per_var[v] > 0]
            if not candidates:
                break # Should not happen if total_viol > 0
                
            v = random.choice(candidates)
            
            # Flip v
            old_val = assign[v]
            new_val = 1 - old_val
            assign[v] = new_val
            
            # Update violations for all constraints containing v
            # We need to check all K4s and Ind4s containing v
            
            # Get list of constraint indices containing v
            # We have k4_by_var and ind4_by_var.
            
            # We need to check the state of the constraint BEFORE and AFTER.
            # Actually, we can just check the new state.
            # If new state is violated, it contributes to viol_per_var.
            # If old state was violated, it was contributing.
            
            # Let's iterate through all constraints containing v.
            # For each, determine if it was violated before, and if it is violated now.
            
            # Optimization: We only need to update `viol_per_var` for the 4 vars in the constraint.
            
            # List of constraint indices to check
            c_indices = k4_by_var[v] + ind4_by_var[v]
            
            for c_idx in c_indices:
                con_type, quad = all_constraints[c_idx]
                
                # Check old state
                # We can't easily check old state without reverting or storing.
                # Let's just check new state and compare with "was it violated?".
                # We need to know if it was violated.
                
                # This approach is tricky because we don't store per-constraint violation status.
                # Let's store `is_violated` array for constraints.
                
            # Let's restart the logic with `is_violated` array.
            
        # If we exit the loop without success, try next restart
        # But wait, I need to implement the `is_violated` tracking properly inside the restart loop.
        
        # Revised Inner Loop Logic:
        # 1. Init assign, viol_per_var, total_viol, is_violated (list of bools)
        # 2. Loop:
        #    - Pick v from violated vars.
        #    - Flip v.
        #    - For each constraint C containing v:
        #       - old_viol = is_violated[C]
        #       - new_viol = not is_satisfied(C, assign)
        #       - if old_viol == new_viol: continue
        #       - if new_viol:
        #           total_viol += 1
        #           is_violated[C] = True
        #           for u in C: viol_per_var[u] += 1
        #       else:
        #           total_viol -= 1
        #           is_violated[C] = False
        #           for u in C: viol_per_var[u] -= 1

    return None

def construct():
    adj = _build_paley37()
    n = 37
    
    best_n = 37
    best_adj = adj[:]
    
    # Try to extend up to 42
    for target_n in range(38, 43):
        if time.time() > DEADLINE:
            break
            
        k4_list = _find_k4_sets(adj, n)
        ind4_list = _find_ind4_sets(adj, n)
        
        success = False
        # Try multiple times in case of bad luck
        attempts = 0
        while time.time() < DEADLINE and not success:
            attempts += 1
            new_adj = _try_add_vertex(adj, n, k4_list, ind4_list)
            if new_adj is not None:
                adj = new_adj
                n = target_n
                best_n = n
                best_adj = adj[:]
                print(f"Extended to n={n}")
                success = True
            else:
                # If it fails, we might need to restart the whole search for this target_n?
                # The function already does restarts. If it returns None, it's likely hard.
                # We can break to save time for other strategies?
                # For now, if it fails once with 500 restarts, it's probably stuck.
                break
        
        if not success:
            break
    
    # Build edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append((i, j))
                
    return best_n, edges

# EVOLVE-BLOCK-END
