"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys
import heapq
from collections import defaultdict

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Safety margin: 300s limit, leave 10s for cleanup/return
    deadline = time.time() + 290
    
    # 1. Preprocessing and Data Structures
    # Convert to 0-based indices internally for speed, but keep 1-based for output logic if needed
    # We will work with 0-based variable indices and sign +1/-1
    # lit > 0 means var is True, lit < 0 means var is False
    # In 0-based: var = abs(lit)-1
    
    # Build initial clauses as lists of (var, sign)
    # sign: 1 if lit > 0, -1 if lit < 0
    raw_clauses = []
    for c in clauses:
        cl = []
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            cl.append((v, s))
        raw_clauses.append(cl)
        
    m = len(raw_clauses)
    
    # Quick check: if any clause is empty -> UNSAT (but task says satisfiable)
    # If any clause is size 1, we can fix it.
    
    # --- Helper: Check if assignment satisfies all clauses ---
    def is_satisfied(assign):
        for c in raw_clauses:
            sat = False
            for v, s in c:
                # s=1 means we need assign[v]==True (1)
                # s=-1 means we need assign[v]==False (0)
                if (assign[v] == 1 and s == 1) or (assign[v] == 0 and s == -1):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def count_unsat(assign):
        cnt = 0
        for c in raw_clauses:
            sat = False
            for v, s in c:
                if (assign[v] == 1 and s == 1) or (assign[v] == 0 and s == -1):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    # --- Preprocessing: Unit Propagation and Pure Literal Elimination ---
    # We'll do a simplified version to get a starting point or reduce the problem
    # For DPLL, we handle unit propagation dynamically.
    # For WalkSAT, a good starting assignment is key.
    
    # Let's try a simple deterministic heuristic for initial assignment:
    # Count occurrences of each literal. Assign to satisfy the more frequent one?
    # Or just random.
    
    # --- Strategy 1: DPLL with Watched Literals ---
    # This is the structural change.
    
    def dpll_solver(deadline_dpll):
        """
        Recursive DPLL with Watched Literals.
        Returns (solved, assignment)
        """
        # Data structures for DPLL
        # We maintain a stack of decisions
        # assignment: list of -1 (unassigned), 0 (False), 1 (True)
        assignment = [-1] * n
        
        # Clause representation for Watched Literals
        # Each clause is a list of literals.
        # We watch two literals per clause.
        # To simplify implementation in Python (which is slow), we will use a simplified DPLL
        # that doesn't fully implement watched literals in the most efficient way (arrays of indices)
        # but rather a standard recursive DPLL with activity-based variable ordering.
        # Given n=50, standard DPLL might be okay if the instance is easy.
        # However, standard DPLL without watched literals is O(m*n) per step.
        
        # Let's implement a "MiniDPLL" that uses a list of clauses and filters them.
        # This is O(m) per step. With n=50, depth max 50. 50*50*m steps.
        # If m is ~100-200, this is 250k-500k operations per path.
        # Number of paths could be exponential.
        
        # To make it robust, we will use a heuristic:
        # 1. Unit propagation
        # 2. Pure literal elimination (optional, can be skipped for simplicity in recursion)
        # 3. Split on variable with highest degree or VSIDS
        
        # VSIDS Implementation
        # var_activity[v]
        # clause_activity[c]
        
        var_activity = [0.0] * n
        clause_activity = [0.0] * m
        decay_var = 0.95
        decay_clause = 0.95
        
        # We need to track which clauses contain which variables for efficient increment
        var_to_clauses = defaultdict(list)
        for i, c in enumerate(raw_clauses):
            for v, s in c:
                var_to_clauses[v].append(i)
        
        # To avoid deep recursion issues in Python, we might hit recursion limit.
        # sys.setrecursionlimit(10000)
        sys.setrecursionlimit(10000)
        
        # We will use a non-recursive stack-based approach or careful recursion.
        # Given the time limit, we will implement a simplified recursive DPLL.
        
        def propagate(assignment, clauses_state):
            """
            Performs unit propagation.
            clauses_state: list of clauses, where each clause is a list of (v, s).
            We modify clauses_state in place (remove satisfied, shrink unit).
            Returns:
            -1 if contradiction
            0 if success
            """
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for c in clauses_state:
                    # Check if clause is satisfied
                    sat = False
                    # Check if clause is unit
                    unit_lit = None
                    unassigned_count = 0
                    for v, s in c:
                        val = assignment[v]
                        if val == 1 and s == 1:
                            sat = True
                            break
                        elif val == 0 and s == -1:
                            sat = True
                            break
                        elif val == -1:
                            unassigned_count += 1
                            if unassigned_count == 1:
                                unit_lit = (v, s)
                    
                    if sat:
                        continue # Remove satisfied clause
                    
                    if unassigned_count == 0:
                        return -1 # Contradiction (empty clause)
                    
                    if unassigned_count == 1:
                        # Unit clause
                        v, s = unit_lit
                        if assignment[v] == -1:
                            assignment[v] = 1 if s == 1 else 0
                            changed = True
                        elif (assignment[v] == 1 and s == -1) or (assignment[v] == 0 and s == 1):
                            return -1 # Conflict
                        # Keep the clause? No, it's satisfied by the assignment.
                        # Actually, if we assign the unit literal, the clause is satisfied.
                        # So we can remove it.
                        continue
                    
                    new_clauses.append(c)
                
                clauses_state.clear()
                clauses_state.extend(new_clauses)
            
            return 0

        def pick_variable(clauses_state):
            """
            Pick variable using VSIDS-like heuristic.
            """
            # Decay activities
            for i in range(n):
                var_activity[i] *= decay_var
            for i in range(m):
                clause_activity[i] *= decay_clause
            
            # Increment activity of variables in unsatisfied clauses?
            # Standard VSIDS increments on conflict. Here we are just branching.
            # Let's use a simpler heuristic: Degree of variable in remaining clauses.
            # Or random if activities are similar.
            
            # Count occurrences in remaining clauses
            degree = [0] * n
            for c in clauses_state:
                seen = set()
                for v, s in c:
                    if assignment[v] == -1: # Only count unassigned
                        if v not in seen:
                            degree[v] += 1
                            seen.add(v)
            
            # Pick variable with max degree among unassigned
            max_deg = -1
            best_v = -1
            for v in range(n):
                if assignment[v] == -1 and degree[v] > max_deg:
                    max_deg = degree[v]
                    best_v = v
            
            if best_v == -1:
                return -1 # All assigned
            
            return best_v

        def dpll_rec(assignment, clauses_state, depth):
            if time.time() > deadline_dpll:
                return None # Timeout
            
            # Check if all assigned
            # If all assigned and no contradictions (implied by clauses_state being empty or satisfied), we are done.
            # But we need to ensure all original clauses are satisfied.
            # If clauses_state is empty, all remaining clauses were satisfied by propagation.
            # So if we reach here with no contradictions, and all vars assigned, it's a solution.
            
            # Check if any unassigned variables
            unassigned = [v for v in range(n) if assignment[v] == -1]
            if not unassigned:
                # Verify solution
                if is_satisfied(assignment):
                    return assignment[:]
                return None # Should not happen if propagation is correct
            
            # Unit Propagation
            # We pass a copy of clauses_state to propagate? 
            # Propagate modifies it. We need to backtrack.
            # Deep copying lists of lists is expensive.
            # For n=50, m~100, it might be okay for a few levels, but bad for deep search.
            
            # Optimization: Only copy if we branch.
            # First, try propagation on the current state.
            # If propagation leads to contradiction, we need to backtrack.
            
            # Let's do a simplified approach:
            # 1. Copy clauses_state
            # 2. Propagate
            # 3. If contradiction, return None
            # 4. If solved, return solution
            # 5. Else, pick var, branch True, then branch False.
            
            # To avoid O(m) copy every time, we could use trail-based backtracking, but that's complex.
            # Given Python speed, let's try to limit depth or use a simpler heuristic.
            
            # Actually, let's just use WalkSAT. DPLL in Python is notoriously slow for 50 vars unless the instance is very easy.
            # The "Structural Change" might be better served by a **Genetic Algorithm** or **Simulated Annealing** which are iterative and don't have recursion/copy overhead.
            
            # Let's pivot the DPLL idea to a **Local Search with Clause Weighting** which is structurally distinct from the basic WalkSAT in the seed (which uses static weights or simple bias).
            # We will implement **Minisat-style Local Search**:
            # 1. Maintain clause weights.
            # 2. When a clause is unsatisfied, increase its weight.
            # 3. Variable flip selection is weighted by the sum of weights of unsatisfied clauses containing that variable.
            # 4. Restarts with clause weight decay.
            
            return None # Abort DPLL path

        # Since DPLL in Python is risky, let's stick to the **Clause Weighting Local Search** as the structural improvement.
        # It is a different optimization landscape (energy landscape with dynamic weights) compared to the seed's static/biased WalkSAT.
        
        return None

    # --- Strategy 2: Clause Weighting Local Search (The Structural Change) ---
    
    # Initial assignment
    # Use a heuristic: assign variables to satisfy the most frequent literals
    lit_counts = defaultdict(int)
    for c in raw_clauses:
        for v, s in c:
            lit_counts[(v, s)] += 1
    
    init_assign = [0] * n
    for v in range(n):
        # Compare (v, 1) vs (v, -1)
        if lit_counts[(v, 1)] > lit_counts[(v, -1)]:
            init_assign[v] = 1
        else:
            init_assign[v] = 0
    
    # Check if solved
    if is_satisfied(init_assign):
        return [bool(x) for x in init_assign]
    
    # Data structures for Local Search
    # clause_weights[ci]
    clause_weights = [1.0] * m
    
    # Precompute variable to clause index
    var_to_clauses = defaultdict(list)
    for i, c in enumerate(raw_clauses):
        for v, s in c:
            var_to_clauses[v].append(i)
    
    def get_unsat_clauses(assign):
        unsat = []
        for i, c in enumerate(raw_clauses):
            sat = False
            for v, s in c:
                if (assign[v] == 1 and s == 1) or (assign[v] == 0 and s == -1):
                    sat = True
                    break
            if not sat:
                unsat.append(i)
        return unsat
    
    def local_search(deadline_ls):
        """
        Clause Weighting Local Search.
        """
        cur_assign = init_assign[:]
        cur_unsat = get_unsat_clauses(cur_assign)
        best_assign = cur_assign[:]
        best_unsat_count = len(cur_unsat)
        
        # Clause weights are global, but we decay them on restart
        # We will run multiple "phases"
        
        rng = random.Random(42)
        
        phase = 0
        # Adaptive restart strategy
        # Start with short runs, increase if no improvement
        
        while time.time() < deadline_ls:
            phase += 1
            # Decay clause weights
            decay = 0.5
            for i in range(m):
                clause_weights[i] *= decay
            
            # Reset weights for unsatisfied clauses in current best?
            # Or just let them grow during the search.
            
            # Run a local search phase for a certain number of steps
            # Steps proportional to n * 10?
            max_steps = 1000 + phase * 500
            
            # Reset clause weights to 1 for unsatisfied clauses of current state?
            # No, keep the weights, they guide the search.
            
            for step in range(max_steps):
                if time.time() > deadline_ls:
                    break
                
                if not cur_unsat:
                    return cur_assign
                
                # Pick an unsatisfied clause
                # Weighted selection based on clause_weights
                total_w = sum(clause_weights[i] for i in cur_unsat)
                r = rng.random() * total_w
                cum = 0
                target_ci = cur_unsat[-1]
                for ci in cur_unsat:
                    cum += clause_weights[ci]
                    if r <= cum:
                        target_ci = ci
                        break
                
                # Increment weight of this clause
                clause_weights[target_ci] += 1.0
                
                # Pick a variable from this clause to flip
                # Weight variables by the sum of weights of OTHER unsatisfied clauses they appear in?
                # Or just random?
                # Standard WalkSAT: pick random var from clause.
                # Improved: pick var that appears in fewest other unsatisfied clauses (to minimize damage)
                
                clause = raw_clauses[target_ci]
                # Calculate score for each var in clause
                # Score = number of unsatisfied clauses (excluding current) that this var is in
                # Lower score is better
                scores = []
                for v, s in clause:
                    # Count unsat clauses containing v
                    cnt = 0
                    for aci in var_to_clauses[v]:
                        if aci != target_ci and aci in cur_unsat: # cur_unsat is a list, membership check is O(N)
                            # Optimization: use a set for cur_unsat
                            pass
                    # This is too slow. 
                    # Let's use a simpler heuristic: Random flip with bias towards satisfying literal.
                    scores.append(rng.random())
                
                # Pick var with min score (or random if we use random scores)
                # Let's stick to: 50% chance to flip to satisfy, 50% random flip.
                v, s = rng.choice(clause)
                
                # Decide new value
                if rng.random() < 0.5:
                    # Flip to satisfy
                    new_val = 1 if s == 1 else 0
                else:
                    # Flip to opposite
                    new_val = 0 if cur_assign[v] == 1 else 1
                
                old_val = cur_assign[v]
                if old_val == new_val:
                    continue
                
                cur_assign[v] = new_val
                
                # Update unsatisfied clauses
                # This is the expensive part.
                # We need to re-evaluate all clauses containing v.
                new_unsat = []
                for i in range(m):
                    c = raw_clauses[i]
                    sat = False
                    for vv, ss in c:
                        if (cur_assign[vv] == 1 and ss == 1) or (cur_assign[vv] == 0 and ss == -1):
                            sat = True
                            break
                    if not sat:
                        new_unsat.append(i)
                
                cur_unsat = new_unsat
                cur_unsat_count = len(cur_unsat)
                
                if cur_unsat_count < best_unsat_count:
                    best_unsat_count = cur_unsat_count
                    best_assign = cur_assign[:]
                    
                if cur_unsat_count == 0:
                    return cur_assign
                    
            # End of phase
            # If we didn't improve, maybe restart from best_assign
            if best_unsat_count < len(cur_unsat):
                cur_assign = best_assign[:]
                cur_unsat = get_unsat_clauses(cur_assign)
            
            # Decay weights again at end of phase?
            for i in range(m):
                clause_weights[i] *= 0.5

        return best_assign

    # Run Local Search
    # Allocate 280s to Local Search
    result = local_search(time.time() + 280)
    
    if result is not None:
        # Convert to boolean list
        return [bool(x) for x in result]
    
    # Fallback: Return best effort or random
    return [bool(x) for x in init_assign]

# EVOLVE-BLOCK-END
