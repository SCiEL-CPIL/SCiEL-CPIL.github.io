"""CDCL 3-SAT solver with watched literals, clause learning, and VSIDS."""
# EVOLVE-BLOCK-START
import random
import time
from collections import defaultdict


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    if num_clauses == 0:
        return [False] * n
    
    # Convert to 0-based signed literals: positive = var True, negative = var False
    # We'll use 0-based variable indices: literal lit means var (lit-1), sign = lit>0
    # Internally we use: lit > 0 means var lit-1 is True; lit < 0 means var -lit-1 is False
    
    start_time = time.time()
    deadline = start_time + 285.0  # safety margin
    
    # Preprocess: convert clauses to 0-based
    # A literal is stored as: (var_idx, is_positive)
    # var_idx is 0-based, is_positive tells if the literal is positive
    processed_clauses = []
    for clause in clauses:
        proc = []
        for lit in clause:
            var_idx = abs(lit) - 1
            is_pos = (lit > 0)
            proc.append((var_idx, is_pos))
        processed_clauses.append(proc)
    
    # Check for trivial cases
    # If any clause is empty -> UNSAT (but problem says satisfiable)
    # If any clause has a literal and its negation -> tautology, can remove
    
    # Build initial clause database
    # Each clause is a list of (var_idx, is_pos)
    # We'll maintain:
    # - clauses: list of lists of (var_idx, is_pos)
    # - watches: var -> list of clause indices that watch this variable
    # - assignment: list of None (unassigned), True, False
    # - trail: list of (var_idx, value) in order of assignment
    # - trail_lim: list of trail indices marking decision levels
    
    # VSIDS scoring
    var_scores = [1.0] * n
    var_decay = 0.95
    score_decay_counter = 0
    DECAY_INTERVAL = 1000
    
    # State
    assignment = [None] * n  # None = unassigned
    trail = []  # list of (var_idx, value)
    trail_lim = []  # list of ints, trail length at each decision level
    decision_level = 0
    
    # Watches: for each variable, list of clause indices
    watches = defaultdict(list)
    
    # Initialize watches: for each clause, watch the first two literals
    # A clause is satisfied if any of its literals is true under current assignment
    # We watch two literals; if one becomes false, we need to find a new watch
    
    def is_lit_true(var_idx, is_pos):
        val = assignment[var_idx]
        if val is None:
            return False
        return val == is_pos
    
    def is_lit_false(var_idx, is_pos):
        val = assignment[var_idx]
        if val is None:
            return False
        return val != is_pos
    
    def is_clause_satisfied(clause_idx):
        for var_idx, is_pos in processed_clauses[clause_idx]:
            if is_lit_true(var_idx, is_pos):
                return True
        return False
    
    # Initialize watches
    for ci, clause in enumerate(processed_clauses):
        if len(clause) >= 2:
            watches[clause[0][0]].append(ci)
            watches[clause[1][0]].append(ci)
        elif len(clause) == 1:
            watches[clause[0][0]].append(ci)
    
    # LCN (Literal Conflict Number) for VSIDS
    lcn = [0] * n  # lcn[var] = number of conflicts where var was in conflict clause
    
    def analyze_conflict(conflict_clause_idx):
        """Analyze conflict and return (reason_var, reason_level, learned_clause, level_to_cancel)"""
        # conflict_clause_idx is the index in the full clause list (original or learned)
        # We need to find the clause that caused the conflict
        # The conflict clause has all literals false
        # We need to do implication analysis
        
        # Get the conflict clause
        if conflict_clause_idx < len(processed_clauses):
            clause = processed_clauses[conflict_clause_idx]
        else:
            clause = learned_clauses[conflict_clause_idx - len(processed_clauses)]
        
        # Find the last literal in the clause (most recently assigned)
        # We need to track which literal was the "reason" for each assignment
        # Simplified: we'll do 1-UIP style analysis
        
        # Build the set of variables in the conflict clause that are assigned
        # The most recent one (highest decision level) is the first to be "unassigned" in analysis
        # For simplicity, we'll use a basic scheme:
        
        # Find the variable with the highest decision level in the conflict clause
        # That's our first "reason" variable
        max_level = -1
        first_var = None
        for var_idx, is_pos in clause:
            if assignment[var_idx] is not None:
                # Get the decision level when this var was assigned
                # We need a var_to_level mapping
                lvl = var_to_level[var_idx]
                if lvl > max_level:
                    max_level = lvl
                    first_var = var_idx
        
        if first_var is None:
            return None, 0, clause, 0
        
        # For a proper CDCL, we need reason tracking. Let's simplify:
        # Learn a clause that is the conflict clause itself (weaker but valid)
        # And backtrack to the level before the highest level variable in the clause
        
        # Actually, let's do proper 1-UIP analysis
        # We need reason tracking: for each implied variable, which clause implied it
        
        # This is getting complex. Let's use a simpler approach:
        # Learn the conflict clause and backtrack past the highest level variable
        
        # Find the second highest level
        levels_in_clause = []
        for var_idx, is_pos in clause:
            if assignment[var_idx] is not None:
                levels_in_clause.append(var_to_level[var_idx])
        
        if len(levels_in_clause) <= 1:
            bt_level = 0
        else:
            levels_in_clause.sort()
            bt_level = levels_in_clause[-2]  # second highest
        
        # The learned clause is the conflict clause
        learned = clause[:]
        
        # VSIDS bump
        for var_idx, is_pos in clause:
            if assignment[var_idx] is not None:
                lcn[var_idx] += 1
                var_scores[var_idx] += 1.0
        
        return first_var, bt_level, learned, bt_level
    
    # We need var_to_level: tracks the decision level at which each var was assigned
    var_to_level = [0] * n
    
    # Reason tracking: for implied assignments, store the clause that caused it
    # var_reason[var] = (clause_idx, literal_pos_in_clause) or None for decisions
    var_reason = [None] * n
    
    # Learned clauses
    learned_clauses = []
    learned_watches = defaultdict(list)
    
    def add_clause(clause, is_learned=False):
        """Add a clause to the database. Returns clause index."""
        if is_learned:
            idx = len(processed_clauses) + len(learned_clauses)
            learned_clauses.append(clause)
            # Set up watches
            if len(clause) >= 2:
                learned_watches[clause[0][0]].append(idx)
                learned_watches[clause[1][0]].append(idx)
            elif len(clause) == 1:
                learned_watches[clause[0][0]].append(idx)
            return idx
        else:
            idx = len(processed_clauses)
            processed_clauses.append(clause)
            if len(clause) >= 2:
                watches[clause[0][0]].append(idx)
                watches[clause[1][0]].append(idx)
            elif len(clause) == 1:
                watches[clause[0][0]].append(idx)
            return idx
    
    def propagate():
        """Unit propagation. Returns conflict clause index or None if no conflict."""
        while trail_lim and len(trail) > trail_lim[-1]:
            var_idx, value = trail[-1]
            trail.pop()
            
            # For each clause watching var_idx, check if it needs updating
            # We need to check both watches and learned_watches
            all_watches = watches[var_idx] + learned_watches[var_idx]
            
            for ci in list(all_watches):
                # Determine which clause list this is in
                if ci < len(processed_clauses):
                    clause = processed_clauses[ci]
                    watch_list = watches
                else:
                    clause = learned_clauses[ci - len(processed_clauses)]
                    watch_list = learned_watches
                
                # Check if the clause is satisfied
                if is_clause_satisfied(ci):
                    continue
                
                # The literal (var_idx, is_pos) where is_pos == value is now false
                # We need to find if var_idx appears in the clause with sign matching value
                # If so, we need to find a new watch literal
                
                # Find the literal in the clause that matches var_idx
                lit_found = False
                for i, (v, is_pos) in enumerate(clause):
                    if v == var_idx:
                        # This literal is now false if is_pos == value
                        if is_pos == value:
                            lit_found = True
                        break
                
                if not lit_found:
                    continue
                
                # Need to find a new watch: look for an unassigned or true literal
                new_watch_idx = -1
                for i, (v, is_pos) in enumerate(clause):
                    if v == var_idx:
                        continue
                    if is_lit_true(v, is_pos):
                        # Clause is satisfied, no need to update watch
                        new_watch_idx = i
                        break
                    elif assignment[v] is None:
                        # Can use as new watch
                        new_watch_idx = i
                        break
                
                if new_watch_idx == -1:
                    # All other literals are false, and this one is false too
                    # This is a conflict
                    return ci
                else:
                    # Update the watch: replace var_idx with clause[new_watch_idx][0]
                    v_new = clause[new_watch_idx][0]
                    # Remove ci from watches[var_idx]
                    if ci in watch_list[var_idx]:
                        watch_list[var_idx].remove(ci)
                    # Add ci to watches[v_new]
                    watch_list[v_new].append(ci)
        
        return None
    
    def pick_variable():
        """Pick an unassigned variable using VSIDS."""
        best_var = -1
        best_score = -1
        for i in range(n):
            if assignment[i] is None:
                if var_scores[i] > best_score:
                    best_score = var_scores[i]
                    best_var = i
        if best_var == -1:
            return -1
        # Randomize slightly
        if random.random() < 0.1:
            # Pick randomly among top-scoring
            top_vars = [i for i in range(n) if assignment[i] is None and var_scores[i] > best_score * 0.5]
            if top_vars:
                best_var = random.choice(top_vars)
        return best_var
    
    def decide_and_propagate():
        """Make a decision and propagate. Returns conflict clause index or None."""
        # Decay scores
        nonlocal score_decay_counter, var_scores
        score_decay_counter += 1
        if score_decay_counter >= DECAY_INTERVAL:
            for i in range(n):
                var_scores[i] *= var_decay
            score_decay_counter = 0
        
        var = pick_variable()
        if var == -1:
            return None  # All assigned, no conflict
        
        # Decide
        value = random.random() < 0.5
        assignment[var] = value
        trail.append((var, value))
        var_to_level[var] = decision_level
        var_reason[var] = None  # Decision
        
        # Propagate
        conflict = propagate()
        return conflict
    
    # Main CDCL loop
    max_conflicts = 500000
    conflict_count = 0
    restart_interval = 100
    current_restarts = 0
    
    while time.time() < deadline:
        # Check if all assigned and no conflicts
        if decision_level == 0 and len(trail) == 0:
            # Fresh start
            pass
        
        # Try to make progress
        conflict = decide_and_propagate()
        
        if conflict is None:
            # No conflict, check if all variables assigned
            if all(assignment[i] is not None for i in range(n)):
                # Verify solution
                for ci in range(len(processed_clauses)):
                    clause = processed_clauses[ci]
                    if not is_clause_satisfied(ci):
                        break
                else:
                    # All clauses satisfied
                    return [assignment[i] for i in range(n)]
                # If not all satisfied, something went wrong
                # Try again
                break
            
            # Continue searching
            continue
        
        # Conflict occurred
        conflict_count += 1
        current_restarts += 1
        
        # Analyze conflict
        _, bt_level, learned, _ = analyze_conflict(conflict)
        
        if learned is None:
            # Cannot analyze, restart
            break
        
        # Add learned clause
        add_clause(learned, is_learned=True)
        
        # Backtrack
        while decision_level > bt_level:
            # Unassign variables at this level
            if trail_lim:
                lim = trail_lim[-1]
                trail_lim.pop()
                while len(trail) > lim:
                    var_idx, value = trail.pop()
                    assignment[var_idx] = None
                    var_to_level[var_idx] = 0
                    var_reason[var_idx] = None
            else:
                break
            decision_level -= 1
        
        if decision_level == 0 and conflict_count % restart_interval == 0:
            # Full restart
            assignment = [None] * n
            trail = []
            trail_lim = []
            decision_level = 0
            var_to_level = [0] * n
            var_reason = [None] * n
            # Keep learned clauses
            # Reset watches for original clauses
            watches = defaultdict(list)
            for ci, clause in enumerate(processed_clauses):
                if len(clause) >= 2:
                    watches[clause[0][0]].append(ci)
                    watches[clause[1][0]].append(ci)
                elif len(clause) == 1:
                    watches[clause[0][0]].append(ci)
            learned_watches = defaultdict(list)
            for li, clause in enumerate(learned_clauses):
                idx = len(processed_clauses) + li
                if len(clause) >= 2:
                    learned_watches[clause[0][0]].append(idx)
                    learned_watches[clause[1][0]].append(idx)
                elif len(clause) == 1:
                    learned_watches[clause[0][0]].append(idx)
        
        if conflict_count > max_conflicts:
            break
    
    # If we didn't find a solution, return all False as fallback
    # But first, try a quick WalkSAT on the remaining
    if time.time() < deadline - 5:
        # Quick local search fallback
        assignment = [random.random() < 0.5 for _ in range(n)]
        for _ in range(2000):
            if time.time() > deadline - 2:
                break
            # Find unsatisfied clause
            unsat = None
            for ci in range(len(processed_clauses)):
                clause = processed_clauses[ci]
                sat = False
                for var_idx, is_pos in clause:
                    if is_lit_true(var_idx, is_pos):
                        sat = True
                        break
                if not sat:
                    unsat = ci
                    break
            if unsat is None:
                return [assignment[i] for i in range(n)]
            clause = processed_clauses[unsat]
            var_idx, is_pos = random.choice(clause)
            assignment[var_idx] = not assignment[var_idx]
        # Check if solution
        for ci in range(len(processed_clauses)):
            clause = processed_clauses[ci]
            sat = False
            for var_idx, is_pos in clause:
                if is_lit_true(var_idx, is_pos):
                    sat = True
                    break
            if not sat:
                break
        else:
            return [assignment[i] for i in range(n)]
    
    return [False] * n
# EVOLVE-BLOCK-END
