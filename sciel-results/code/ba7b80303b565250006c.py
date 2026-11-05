import random
import time
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline: 320 seconds to allow for overhead and safety margin against 330s kill
    deadline = time.time() + 320
    
    # Pre-process clauses for faster satisfaction check
    # Store as list of tuples (var_index, is_positive)
    # 1-based literals -> 0-based indices
    processed_clauses = []
    for c in clauses:
        pc = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            pc.append((var_idx, is_pos))
        processed_clauses.append(pc)
        
    m = len(processed_clauses)
    
    # Heuristic: Initialize with random assignment, but maybe slightly biased? 
    # For planted SAT, uniform random is a good start.
    
    def solve_single_run(rng, max_steps, time_limit):
        """
        Runs a single WalkSAT-style search.
        Returns (assignment, success_flag).
        """
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Track clause satisfaction status to avoid O(m) scan per step?
        # For 50 vars, clauses are likely ~250-400 (c/n ratio ~5).
        # O(m) check is ~400 operations. 100k steps = 40M ops. Fast in Python if optimized.
        # However, maintaining counters is O(1) update per flip (checking neighbors).
        
        # Let's use a lightweight approach first, but with clause weights for bias.
        # Actually, simple WalkSAT is robust. Let's add "restarts" logic outside.
        
        # To speed up satisfaction check, we can pre-calculate literals per variable.
        # But let's stick to the simple O(m) check first, it's often fast enough for n=50.
        # If it's too slow, we can optimize.
        
        def is_satisfied(clause_idx):
            for var_idx, is_pos in processed_clauses[clause_idx]:
                if assignment[var_idx] == is_pos:
                    return True
            return False
            
        # Main loop
        for step in range(max_steps):
            # Check for deadline
            if time.time() > time_limit:
                return assignment, False
                
            # Find unsatisfied clauses
            unsat_indices = []
            for i in range(m):
                if not is_satisfied(i):
                    unsat_indices.append(i)
                    
            if not unsat_indices:
                return assignment, True
                
            # Select an unsatisfied clause
            # Simple: random choice. 
            # Better: Weighted by number of literals in clause? 
            # Or just random.
            clause_idx = rng.choice(unsat_indices)
            clause = processed_clauses[clause_idx]
            
            # Select a literal to flip
            # Standard WalkSAT:
            # 1. With probability p, pick a literal from the clause that makes the clause satisfied (if any exist in unsat clause? No, if clause is unsat, no literal satisfies it. 
            #    Wait, WalkSAT flips a literal from the *unsatisfied* clause. 
            #    If we flip a literal in an unsatisfied clause, we might satisfy it.
            #    The "greedy" part is: prefer literals that don't break other satisfied clauses.
            
            # Let's implement the standard greedy heuristic:
            # For each literal in the unsatisfied clause, calculate the cost of flipping it.
            # Cost = number of satisfied clauses that would become unsatisfied.
            # Pick the literal with min cost. Tie-break randomly.
            
            min_cost = float('inf')
            best_var = -1
            best_is_pos = False # target value to set
            candidates = []
            
            for var_idx, is_pos in clause:
                # If we flip this variable, what is the cost?
                # We need to check all clauses containing var_idx.
                # This requires a mapping: var -> list of clauses it appears in.
                # Let's precompute that outside the inner loop? 
                # For n=50, we can just do it on the fly if we have the mapping.
                
                cost = 0
                # Iterate through all clauses? That's O(m) per literal, O(m*3) per step. 
                # 3 * 400 = 1200 ops per step. 100k steps = 120M ops. Might be slow in Python.
                
                # Optimization: Precompute var_to_clauses mapping.
                # But we are inside solve_single_run. Let's pass the mapping.
                
                # Since I haven't built the mapping in this scope, let's build it once outside solve_single_run?
                # Or build it here? Building it every restart is overhead.
                
                # Let's refactor: Build var_to_clauses once at the start of solve().
                
                pass # Placeholder logic, need to integrate mapping
            
            # Due to the complexity of implementing the full weighted WalkSAT correctly with precomputed maps inside this single function structure without significant refactoring, 
            # and given n=50 is small, the simple O(m) scan per step is likely sufficient if we use many steps.
            # Let's stick to a simpler, robust variant: 
            # 1. Random clause.
            # 2. Random literal from that clause.
            # 3. Flip.
            # This is the seed program. It scored 1.0.
            
            # To improve, we can use "Restarts" and "Different Initial Assignments".
            # The seed program used a fixed budget of 20000 steps.
            # If we allow more steps via restarts, we can find harder instances.
            
            # Let's implement a simpler improvement: 
            # Instead of a single long run, do multiple shorter runs with different seeds/assignments.
            
            # For now, let's just flip a random literal in the chosen clause.
            var_idx, is_pos = rng.choice(clause)
            assignment[var_idx] = not assignment[var_idx]
            
        return assignment, False

    # Precompute variable to clauses mapping for future optimization if needed, 
    # but for now let's rely on the robustness of multiple restarts.
    
    # Strategy:
    # 1. Determine number of restarts based on time.
    # 2. Each restart uses a random assignment.
    # 3. Use a geometric distribution for the number of steps in each restart (restart heuristic).
    
    # Parameters
    # Typical successful runs for n=50, m~250 might take 10k-100k steps.
    # Let's try a base of 5000 steps, doubling on failure, capped by time.
    
    # To avoid the O(m) scan being the bottleneck, we can actually check satisfaction more efficiently?
    # No, for n=50, 400 clauses, Python is fast enough for 100k steps.
    # 100k * 400 = 40M simple ops. ~1-2 seconds.
    # So we can do ~100 restarts of 100k steps? No, time limit is 320s.
    # So we can do roughly 100-200 such runs if they take 2s each.
    
    # However, if a run is unsat or hard, it will hit the step limit or time limit.
    
    # Let's implement the restart logic in the main solve function.
    
    best_assignment = [False] * n
    rng_master = random.Random()
    
    # Use a time-based budget
    start_time = time.time()
    
    while time.time() < deadline - 5: # 5s safety margin
        # Decide how long this restart should run
        # Exponential restarts are common.
        # Let's pick a random max_steps for this run.
        # Maybe start small, increase?
        # Or just fixed chunks of time?
        
        # Let's use a fixed step limit per restart, but increase it if we fail?
        # Simpler: Use a random step limit from a distribution.
        # Or just run for a fixed time slice?
        # Running for a fixed time slice (e.g., 1 second) is robust.
        
        # Let's run for 1 second or 100k steps, whichever comes first.
        max_steps = 100000
        run_deadline = time.time() + 1.0
        
        # New random seed for this run
        run_rng = random.Random(rng_master.randint(0, 2**31))
        
        # Initialize assignment
        assignment = [run_rng.random() < 0.5 for _ in range(n)]
        
        # Local search
        for _ in range(max_steps):
            if time.time() > run_deadline:
                break
            if time.time() > deadline - 5:
                break
                
            # Check if solved
            is_solved = True
            for c in processed_clauses:
                sat = False
                for var_idx, is_pos in c:
                    if assignment[var_idx] == is_pos:
                        sat = True
                        break
                if not sat:
                    is_solved = False
                    # Break early if we just want to check, but we need to pick a clause to flip.
                    # Let's collect unsat clauses for better heuristics?
                    # For now, just find one unsat clause.
                    break
            
            if is_solved:
                return assignment
            else:
                # If we broke out of the check loop, we have an unsat clause `c` (if we kept it)
                # But the loop above breaks on the first unsat clause.
                # Let's refine the loop to explicitly find an unsat clause.
                
                # Reset is_solved logic for the flip step
                unsat_clause = None
                for c in processed_clauses:
                    sat = False
                    for var_idx, is_pos in c:
                        if assignment[var_idx] == is_pos:
                            sat = True
                            break
                    if not sat:
                        unsat_clause = c
                        break
                
                if unsat_clause is None:
                    # This shouldn't happen if is_solved was False, but just in case
                    continue
                else:
                    # Pick a literal to flip from the unsat clause
                    var_idx, is_pos = run_rng.choice(unsat_clause)
                    assignment[var_idx] = not assignment[var_idx]
        
        # If we exit the step loop without solving, continue to next restart
        # We could update best_assignment if this one was "closer" to solution, 
        # but without a good metric (e.g., number of unsat clauses), we can't easily.
        # Let's just keep the last assignment as a fallback? 
        # Actually, the task says "Return a satisfying assignment... score is fraction solved".
        # So we only care about success.
        # If we fail all restarts, we return the last generated assignment (which will be unsat, scoring 0 for that instance).
        # But since we return early on success, the last assignment only matters if we fail completely.
        
    return best_assignment # Fallback, though unlikely to be reached if we keep restarting until deadline
