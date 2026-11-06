"""
Planted 3-SAT solver using a hybrid approach:
1. Lightweight DPLL (CDCL-style) with unit propagation and pure literals.
2. Robust WalkSAT with bitmask-accelerated state tracking, structural initialization,
   and a final greedy hill-climbing polish.
"""
import time
import random
import sys
from collections import defaultdict
from collections import defaultdict

DEADLINE_FRACTION = 0.95

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    t_start = time.time()
    deadline = t_start + 330 * DEADLINE_FRACTION
    
    # --- DPLL Solver ---
    def dpll_solver():
        """
        Lightweight DPLL solver with unit propagation and pure literal elimination.
        Uses a simple recursive backtracking with heuristic variable selection.
        """
        # Data structures for DPLL
        # clause_lit_counts: dict mapping (clause_idx, var) -> list of signs? 
        # Simpler: 
        # active_clauses: list of sets of literals (0-based, signed)
        # var_occurrences: dict var -> list of (clause_idx, sign)
        
        # We need to handle unit propagation efficiently.
        # For n=50, m~100-200, a simple list-based approach might be fast enough.
        
        # Let's use a simpler representation for DPLL to avoid complex updates.
        # Variables: 0..n-1
        # Assignment: dict var -> bool or None
        
        # Precompute literal occurrences for DPLL
        # occ[v] = list of (clause_idx, sign)
        dpll_occ = [[] for _ in range(n)]
        dpll_clauses = []
        
        for ci, c in enumerate(clauses):
            c_vars = []
            for l in c:
                if l > 0:
                    v = l - 1
                    sign = 1
                else:
                    v = -l - 1
                    sign = -1
                c_vars.append((v, sign))
                dpll_occ[v].append((ci, sign))
            dpll_clauses.append(c_vars)
            
        def is_satisfied(lit, assign):
            v, sign = lit
            if v in assign:
                return assign[v] == (sign == 1)
            return None
            
        def is_false(lit, assign):
            v, sign = lit
            if v in assign:
                return assign[v] != (sign == 1)
            return None
            
        def unit_propagate(assign, active_clauses):
            """
            Perform unit propagation.
            Returns (success, new_assign, new_active_clauses, conflict_var)
            success: False if conflict
            """
            changed = True
            while changed:
                changed = False
                new_active = []
                for ci, lits in active_clauses:
                    # Filter out false literals
                    remaining = []
                    unit = None
                    conflict = False
                    for lit in lits:
                        s = is_satisfied(lit, assign)
                        if s is True:
                            # Clause satisfied, skip
                            remaining = None
                            break
                        f = is_false(lit, assign)
                        if f is True:
                            # Literal false, skip
                            continue
                        # Literal unassigned
                        remaining.append(lit)
                        if len(remaining) == 1:
                            unit = lit
                    if remaining is None:
                        continue
                    if not remaining:
                        return False, assign, None, None
                    if len(remaining) == 1:
                        v, sign = remaining[0]
                        if v in assign:
                            if assign[v] != (sign == 1):
                                return False, assign, None, v
                            else:
                                continue
                        # Assign
                        assign[v] = (sign == 1)
                        changed = True
                        # Don't add to new_active yet, it's satisfied
                    else:
                        new_active.append((ci, remaining))
                        
                if not changed:
                    return True, assign, new_active, None
                # Loop again
                active_clauses = new_active
                
            return True, assign, new_active, None

        # Main DPLL loop
        # We use a stack for backtracking
        # State: (assign, active_clauses, depth)
        
        # Initial active clauses
        init_active = []
        for ci, lits in enumerate(dpll_clauses):
            init_active.append((ci, lits))
            
        assign = {}
        active = init_active
        
        # Pure literal elimination
        def pure_literal_elimination(assign, active):
            # Find pure literals
            # If a variable appears only with one sign in all active clauses, assign it
            pos_vars = set()
            neg_vars = set()
            for ci, lits in active:
                for v, sign in lits:
                    if v in assign:
                        continue
                    if sign == 1:
                        pos_vars.add(v)
                    else:
                        neg_vars.add(v)
                        
            pure = []
            for v in pos_vars:
                if v not in neg_vars:
                    pure.append((v, True))
            for v in neg_vars:
                if v not in pos_vars:
                    pure.append((v, False))
                    
            for v, val in pure:
                assign[v] = val
                
            # Re-propagate
            if pure:
                return unit_propagate(assign, active)
            return True, assign, active, None

        # Heuristic variable selection
        def select_var(assign, active):
            # Pick variable with max occurrences in active clauses
            counts = defaultdict(int)
            for ci, lits in active:
                for v, sign in lits:
                    if v not in assign:
                        counts[v] += 1
            if not counts:
                return None
            # Tie-breaker: literal frequency bias
            max_count = max(counts.values())
            candidates = [v for v, c in counts.items() if c == max_count]
            # Use pos_count/neg_count from outer scope
            best_v = candidates[0]
            best_bias = -1
            for v in candidates:
                bias = pos_count[v] - neg_count[v]
                # If bias is high, prefer True? 
                # Just pick one with highest abs bias
                if abs(bias) > abs(best_bias):
                    best_bias = bias
                    best_v = v
            return best_v

        # DPLL recursive (implemented iteratively with stack to avoid recursion limit)
        # Stack items: (assign_copy, active_copy, next_branch_var, next_branch_val)
        # This is memory heavy. For n=50, it's okay.
        
        # Simpler: Recursive with sys.setrecursionlimit
        sys.setrecursionlimit(10000)
        
        def dpll_rec(assign, active, depth):
            if time.time() > deadline:
                return None, False
            
            # Pure literal elimination
            ok, assign, active, _ = pure_literal_elimination(assign, active)
            if not ok:
                return None, False
            
            # Check if all satisfied
            if not active:
                return assign, True
            
            # Unit propagation
            ok, assign, active, _ = unit_propagate(assign, active)
            if not ok:
                return None, False
            
            if not active:
                return assign, True
            
            v = select_var(assign, active)
            if v is None:
                # No unassigned variables? Should be satisfied if active is empty.
                # If active is not empty, we have a problem (shouldn't happen with UP)
                return None, False
            
            # Try True
            assign[v] = True
            res, success = dpll_rec(assign, active, depth + 1)
            if success:
                return res, True
            
            # Try False
            assign[v] = False
            res, success = dpll_rec(assign, active, depth + 1)
            if success:
                return res, True
            
            # Backtrack
            del assign[v]
            return None, False

        res, success = dpll_rec(assign, init_active, 0)
        if success:
            # Convert to list of booleans
            sol = [False] * n
            for v, val in res.items():
                sol[v] = val
            # Check if all assigned
            for v in range(n):
                if v not in res:
                    # Unassigned variables can be anything, let's say False
                    sol[v] = False
            return sol
        return None
    
    # --- DPLL Solver ---
    def dpll_solver():
        """
        Lightweight DPLL solver with unit propagation and pure literal elimination.
        Uses a simple recursive backtracking with heuristic variable selection.
        """
        # Data structures for DPLL
        # clause_lit_counts: dict mapping (clause_idx, var) -> list of signs? 
        # Simpler: 
        # active_clauses: list of sets of literals (0-based, signed)
        # var_occurrences: dict var -> list of (clause_idx, sign)
        
        # We need to handle unit propagation efficiently.
        # For n=50, m~100-200, a simple list-based approach might be fast enough.
        
        # Let's use a simpler representation for DPLL to avoid complex updates.
        # Variables: 0..n-1
        # Assignment: dict var -> bool or None
        
        # Precompute literal occurrences for DPLL
        # occ[v] = list of (clause_idx, sign)
        dpll_occ = [[] for _ in range(n)]
        dpll_clauses = []
        
        for ci, c in enumerate(clauses):
            c_vars = []
            for l in c:
                if l > 0:
                    v = l - 1
                    sign = 1
                else:
                    v = -l - 1
                    sign = -1
                c_vars.append((v, sign))
                dpll_occ[v].append((ci, sign))
            dpll_clauses.append(c_vars)
            
        def is_satisfied(lit, assign):
            v, sign = lit
            if v in assign:
                return assign[v] == (sign == 1)
            return None
            
        def is_false(lit, assign):
            v, sign = lit
            if v in assign:
                return assign[v] != (sign == 1)
            return None
            
        def unit_propagate(assign, active_clauses):
            """
            Perform unit propagation.
            Returns (success, new_assign, new_active_clauses, conflict_var)
            success: False if conflict
            """
            changed = True
            while changed:
                changed = False
                new_active = []
                for ci, lits in active_clauses:
                    # Filter out false literals
                    remaining = []
                    unit = None
                    conflict = False
                    for lit in lits:
                        s = is_satisfied(lit, assign)
                        if s is True:
                            # Clause satisfied, skip
                            remaining = None
                            break
                        f = is_false(lit, assign)
                        if f is True:
                            # Literal false, skip
                            continue
                        # Literal unassigned
                        remaining.append(lit)
                        if len(remaining) == 1:
                            unit = lit
                    if remaining is None:
                        continue
                    if not remaining:
                        return False, assign, None, None
                    if len(remaining) == 1:
                        v, sign = remaining[0]
                        if v in assign:
                            if assign[v] != (sign == 1):
                                return False, assign, None, v
                            else:
                                continue
                        # Assign
                        assign[v] = (sign == 1)
                        changed = True
                        # Don't add to new_active yet, it's satisfied
                    else:
                        new_active.append((ci, remaining))
                        
                if not changed:
                    return True, assign, new_active, None
                # Loop again
                active_clauses = new_active
                
            return True, assign, new_active, None

        # Main DPLL loop
        # We use a stack for backtracking
        # State: (assign, active_clauses, depth)
        
        # Initial active clauses
        init_active = []
        for ci, lits in enumerate(dpll_clauses):
            init_active.append((ci, lits))
            
        assign = {}
        active = init_active
        
        # Pure literal elimination
        def pure_literal_elimination(assign, active):
            # Find pure literals
            # If a variable appears only with one sign in all active clauses, assign it
            pos_vars = set()
            neg_vars = set()
            for ci, lits in active:
                for v, sign in lits:
                    if v in assign:
                        continue
                    if sign == 1:
                        pos_vars.add(v)
                    else:
                        neg_vars.add(v)
                        
            pure = []
            for v in pos_vars:
                if v not in neg_vars:
                    pure.append((v, True))
            for v in neg_vars:
                if v not in pos_vars:
                    pure.append((v, False))
                    
            for v, val in pure:
                assign[v] = val
                
            # Re-propagate
            if pure:
                return unit_propagate(assign, active)
            return True, assign, active, None

        # Heuristic variable selection
        def select_var(assign, active):
            # Pick variable with max occurrences in active clauses
            counts = defaultdict(int)
            for ci, lits in active:
                for v, sign in lits:
                    if v not in assign:
                        counts[v] += 1
            if not counts:
                return None
            # Tie-breaker: literal frequency bias
            max_count = max(counts.values())
            candidates = [v for v, c in counts.items() if c == max_count]
            # Use pos_count/neg_count from outer scope
            best_v = candidates[0]
            best_bias = -1
            for v in candidates:
                bias = pos_count[v] - neg_count[v]
                # If bias is high, prefer True? 
                # Just pick one with highest abs bias
                if abs(bias) > abs(best_bias):
                    best_bias = bias
                    best_v = v
            return best_v

        # DPLL recursive (implemented iteratively with stack to avoid recursion limit)
        # Stack items: (assign_copy, active_copy, next_branch_var, next_branch_val)
        # This is memory heavy. For n=50, it's okay.
        
        # Simpler: Recursive with sys.setrecursionlimit
        sys.setrecursionlimit(10000)
        
        def dpll_rec(assign, active, depth):
            if time.time() > deadline:
                return None, False
            
            # Pure literal elimination
            ok, assign, active, _ = pure_literal_elimination(assign, active)
            if not ok:
                return None, False
            
            # Check if all satisfied
            if not active:
                return assign, True
            
            # Unit propagation
            ok, assign, active, _ = unit_propagate(assign, active)
            if not ok:
                return None, False
            
            if not active:
                return assign, True
            
            v = select_var(assign, active)
            if v is None:
                # No unassigned variables? Should be satisfied if active is empty.
                # If active is not empty, we have a problem (shouldn't happen with UP)
                return None, False
            
            # Try True
            assign[v] = True
            res, success = dpll_rec(assign, active, depth + 1)
            if success:
                return res, True
            
            # Try False
            assign[v] = False
            res, success = dpll_rec(assign, active, depth + 1)
            if success:
                return res, True
            
            # Backtrack
            del assign[v]
            return None, False

        res, success = dpll_rec(assign, init_active, 0)
        if success:
            # Convert to list of booleans
            sol = [False] * n
            for v, val in res.items():
                sol[v] = val
            # Check if all assigned
            for v in range(n):
                if v not in res:
                    # Unassigned variables can be anything, let's say False
                    sol[v] = False
            return sol
        return None

    # --- Preprocessing ---
    # Convert to 0-based variable indices internally for faster access
    # Input: 1-based signed literals (1..n, -1..-n)
    # Internal: 0-based variables (0..n-1), literals stored as (var, sign) where sign is True for +, False for -
    
    clause_vars = []
    pos_count = [0] * n
    neg_count = [0] * n
    
    for c in clauses:
        c_vars = []
        for l in c:
            if l > 0:
                v = l - 1
                sign = True
                pos_count[v] += 1
            else:
                v = -l - 1
                sign = False
                neg_count[v] += 1
            c_vars.append((v, sign))
        clause_vars.append(c_vars)
        
    # Build variable-to-clause incidence lists
    # var_clauses[v] = list of (clause_idx, sign)
    var_clauses = [[] for _ in range(n)]
    for ci, c_vars in enumerate(clause_vars):
        for v, sign in c_vars:
            var_clauses[v].append((ci, sign))

    # --- State Management ---
    # assignment: list of booleans (True/False)
    # support: list of ints, support[ci] is the number of satisfied literals in clause ci
    
    def make_support(assignment):
        support = [0] * m
        for ci, c_vars in enumerate(clause_vars):
            s = 0
            for v, sign in c_vars:
                if assignment[v] == sign:
                    s += 1
            support[ci] = s
        return support

    def apply_flip(v, assignment, support, cur_unsat):
        """
        Flip variable v. Returns new cur_unsat.
        """
        old_val = assignment[v]
        new_val = not old_val
        assignment[v] = new_val
        
        delta_unsat = 0
        for ci, sign in var_clauses[v]:
            old_support = support[ci]
            if new_val == sign:
                new_support = old_support + 1
            else:
                new_support = old_support - 1
            
            support[ci] = new_support
            
            if old_support == 0 and new_support == 1:
                delta_unsat -= 1
            elif old_support == 1 and new_support == 0:
                delta_unsat += 1
        
        return cur_unsat + delta_unsat

    # --- Initialization Strategies ---
    rng = random.Random(42)
    
    def init_structural():
        """Initialize based on literal frequency bias."""
        assignment = []
        for v in range(n):
            if pos_count[v] > neg_count[v]:
                # Bias towards True if +v appears more
                p = 0.7
            elif neg_count[v] > pos_count[v]:
                # Bias towards False if -v appears more
                p = 0.3
            else:
                p = 0.5
            assignment.append(rng.random() < p)
        return assignment

    def init_random():
        return [rng.random() < 0.5 for _ in range(n)]

    # --- Search Strategies ---
    
    def walk_sat(assignment, support, cur_unsat, max_steps, noise_prob=0.25):
        """
        Standard WalkSAT.
        If noise, pick random var from random unsat clause.
        If greedy, pick var that maximizes gain (satisfies more unsat, breaks fewer sat).
        For speed, we use a hybrid: 
        1. Pick a random unsatisfied clause.
        2. With probability `noise_prob`, pick a random variable in it.
        3. Otherwise, pick the variable in it that has the best "gain" (local check within the clause is not enough, 
           but checking global gain is expensive. 
           A common fast approximation: pick the variable that appears in the most unsatisfied clauses? 
           Or just random. 
           Let's stick to random for speed, but maybe add a simple heuristic: 
           pick the variable that currently satisfies the fewest clauses? 
           Actually, standard WalkSAT random is often sufficient. 
           Let's add a "greedy" option that calculates gain for the 3 vars in the clause only.
           Gain = (unsat clauses satisfied) - (sat clauses broken).
           This is O(occurrences of the 3 vars).
        """
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            if cur_unsat == 0:
                return assignment, cur_unsat, True
            
            # Pick random unsatisfied clause
            # Optimization: Maintain a list of unsat clauses? 
            # For now, random sampling. If m is small, it's fine.
            # To be faster, we could keep a list, but updating it is tricky.
            # Let's try random sampling first.
            while True:
                ci = rng.randint(0, m - 1)
                if support[ci] == 0:
                    break
            
            c_vars = clause_vars[ci]
            
            if rng.random() < noise_prob:
                # Random flip
                v, sign = c_vars[rng.randint(0, 2)]
            else:
                # Greedy flip among the 3 literals in the clause
                best_v = -1
                best_gain = -100
                
                for v, sign in c_vars:
                    gain = 0
                    # Check impact on all clauses containing v
                    for c_idx, s in var_clauses[v]:
                        if assignment[v] == s:
                            # Currently satisfied by v
                            # Flipping will make it unsatisfied if support becomes 0
                            if support[c_idx] == 1:
                                gain -= 1
                        else:
                            # Currently unsatisfied by v
                            # Flipping will satisfy it if support becomes 1
                            if support[c_idx] == 0:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                
                v = best_v

            cur_unsat = apply_flip(v, assignment, support, cur_unsat)

        return assignment, cur_unsat, (cur_unsat == 0)

    def greedy_hill_climb(assignment, support, cur_unsat, max_time=2.0):
        """
        Greedy hill climbing to polish the solution.
        """
        t_start = time.time()
        improved = True
        while improved and (time.time() - t_start) < max_time and time.time() < deadline:
            improved = False
            if cur_unsat == 0:
                return assignment, cur_unsat, True
            
            best_gain = 0
            best_v = -1
            
            for v in range(n):
                gain = 0
                for ci, sign in var_clauses[v]:
                    if assignment[v] == sign:
                        if support[ci] == 1:
                            gain -= 1
                    else:
                        if support[ci] == 0:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            
            if best_v != -1:
                cur_unsat = apply_flip(best_v, assignment, support, cur_unsat)
                improved = True
        
        return assignment, cur_unsat, (cur_unsat == 0)

    # --- Main Loop ---
    
    # Try DPLL first
    dpll_sol = dpll_solver()
    if dpll_sol:
        return dpll_sol

    best_assignment = None
    best_unsat = m + 1
    
    # Phase 1: Structural Init + WalkSAT
    # Phase 2: Random Init + WalkSAT
    
    # We will run multiple restarts.
    # Each restart:
    # 1. Init (alternating structural/random)
    # 2. WalkSAT with varying noise
    # 3. If not solved, try Greedy Hill Climb on the result
    
    max_restarts = 200
    steps_per_walk = 2000
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Select initialization
        if r == 0:
            assignment = init_structural()
        elif r % 2 == 0:
            assignment = init_structural()
        else:
            assignment = init_random()
            
        # If we have a best_assignment, occasionally restart from it with perturbation
        if r > 5 and best_assignment is not None and rng.random() < 0.2:
            assignment = best_assignment[:]
            # Perturb 3 random variables
            for _ in range(3):
                v = rng.randint(0, n - 1)
                assignment[v] = not assignment[v]
        
        support = make_support(assignment)
        cur_unsat = sum(1 for s in support if s == 0)
        
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        
        if cur_unsat == 0:
            return assignment
        
        # Vary noise
        noise = 0.25
        
        assignment, cur_unsat, solved = walk_sat(assignment, support, cur_unsat, steps_per_walk, noise)
        
        if solved:
            return assignment
        
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        
        # Try to polish with greedy hill climbing
        if time.time() < deadline - 5:
            assignment, cur_unsat, solved = greedy_hill_climb(assignment, support, cur_unsat)
            if solved:
                return assignment
            
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]
        
    # Final Polish on the best assignment found
    if best_assignment is not None and time.time() < deadline - 2:
        assignment = best_assignment[:]
        support = make_support(assignment)
        cur_unsat = sum(1 for s in support if s == 0)
        
        if cur_unsat > 0:
            # One last long greedy climb
            assignment, cur_unsat, solved = greedy_hill_climb(assignment, support, cur_unsat, max_time=deadline - time.time() - 1)
            if solved:
                return assignment
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]
        
        if best_unsat == 0:
            return best_assignment

    if best_assignment is not None:
        return best_assignment
    
    # Fallback
    return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
