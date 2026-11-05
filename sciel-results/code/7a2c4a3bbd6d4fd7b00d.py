import time
import random
import sys

def solve(instance):
    START = time.time()
    DEADLINE = START + 290.0  # Safety margin before 300s kill

    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    # Convert to 0-based literals
    # lit > 0 means positive, lit < 0 means negative
    # var = abs(lit) - 1
    clause_lits = []
    for c in clauses:
        clause_lits.append([lit for lit in c])

    # Build variable occurrence lists for VSIDS
    var_occurrences = [[] for _ in range(n)]
    for i, c in enumerate(clause_lits):
        for lit in c:
            var_occurrences[abs(lit) - 1].append(i)

    # Watched literals: for each variable, store list of (clause_idx, literal)
    # We watch two literals per clause initially
    watches = [[] for _ in range(n)]  # watches[var] = list of (clause_idx, literal)
    # clause_watches[clause_idx] = list of two watched literal variables
    clause_watches = [None] * m

    def init_watches():
        for i, c in enumerate(clause_lits):
            if len(c) >= 2:
                clause_watches[i] = [c[0], c[1]]
                watches[abs(c[0]) - 1].append((i, c[0]))
                watches[abs(c[1]) - 1].append((i, c[1]))
            elif len(c) == 1:
                clause_watches[i] = [c[0]]
                watches[abs(c[0]) - 1].append((i, c[0]))
            else:
                clause_watches[i] = []

    init_watches()

    # Assignment: 0 = unassigned, 1 = true, -1 = false
    assignment = [0] * n
    # Trail: list of (var, value) for backtracking
    trail = []
    trail_pos = [0] * n  # trail_pos[var] = position in trail where var was assigned

    # VSIDS activity
    activity = [1.0] * n
    inc = 1.0
    var_inc = 1.0 / (100 * n) if n > 0 else 1.0

    # Decision level
    decision_level = [0] * n
    current_level = 0

    # Queue for unit propagation
    # Use a simple list as queue
    queue = []

    def assign(var, val, level):
        """Assign variable, return False if conflict."""
        if assignment[var] != 0:
            return assignment[var] == val
        assignment[var] = val
        trail.append((var, val))
        trail_pos[var] = len(trail) - 1
        decision_level[var] = level
        # Check watches for this variable
        # If we assign var to val, then literal lit is satisfied if sign(lit) == val
        # For each watch on var, check if the other watch is satisfied
        for (ci, lit) in watches[var]:
            # If this literal is satisfied by the assignment, we're fine
            # lit is satisfied if (lit > 0 and val == 1) or (lit < 0 and val == -1)
            if (lit > 0 and val == 1) or (lit < 0 and val == -1):
                continue
            # This literal is falsified, need to find a new watch
            cw = clause_watches[ci]
            if cw is None:
                continue
            # Find the other watched literal
            other_idx = 0 if cw[0] == lit else 1
            other_lit = cw[other_idx]
            # Check if other literal is satisfied
            ov = abs(other_lit) - 1
            if (other_lit > 0 and assignment[ov] == 1) or (other_lit < 0 and assignment[ov] == -1):
                # Other watch is satisfied, no need to update
                continue
            # Find a new literal in the clause that is not assigned to falsify
            c = clause_lits[ci]
            new_lit = 0
            for l in c:
                v = abs(l) - 1
                if (l > 0 and assignment[v] == 1) or (l < 0 and assignment[v] == -1):
                    continue  # This literal is falsified
                new_lit = l
                break
            if new_lit == 0:
                # All literals falsified -> conflict
                return False
            # Replace the falsified watch
            cw[other_idx] = new_lit
            # Remove old watch from watches[var]
            # We'll do lazy cleanup or just add new watch
            watches[abs(new_lit) - 1].append((ci, new_lit))
            # Note: we leave the old watch in watches[var], it will be skipped
            # when we see it's satisfied or we can clean up later
        return True

    def unassign_all_to(level):
        """Undo all assignments above level."""
        while trail and decision_level[trail[-1][0]] >= level:
            var, val = trail.pop()
            assignment[var] = 0
            trail_pos[var] = -1

    def propagate():
        """Process unit clauses. Returns True if no conflict, False if conflict."""
        while queue:
            lit = queue.pop()
            var = abs(lit) - 1
            val = 1 if lit > 0 else -1
            if assignment[var] != 0:
                if assignment[var] != val:
                    return False
                continue
            if not assign(var, val, current_level):
                return False
        return True

    def analyze_conflict():
        """Analyze the conflict and return the clause to learn."""
        # Simplified: we'll just do a basic analysis
        # Find the clause that caused the conflict
        # This is complex with watched literals, so we'll use a simpler approach
        # For now, return None to indicate we should backtrack
        return None

    # Simpler approach: use a basic DPLL with unit propagation
    # but with better variable ordering

    def dpll_search():
        """Main DPLL search with unit propagation."""
        nonlocal current_level, activity, inc, var_inc, queue

        while True:
            if time.time() > DEADLINE:
                return None

            # Unit propagation
            if not propagate():
                # Conflict
                if current_level == 0:
                    return None
                # Backtrack
                # Find the highest level to backtrack to
                # Simplified: backtrack one level
                unassign_all_to(current_level)
                current_level -= 1
                continue

            # Check if all variables are assigned
            all_assigned = True
            for v in range(n):
                if assignment[v] == 0:
                    all_assigned = False
                    break
            if all_assigned:
                return assignment[:]

            # Select a variable to branch on using VSIDS-like heuristic
            # Count occurrences in active clauses (unassigned)
            var_scores = [0.0] * n
            for v in range(n):
                if assignment[v] != 0:
                    continue
                # Score based on activity and occurrence
                for ci in var_occurrences[v]:
                    c = clause_lits[ci]
                    # Check if clause is active (not satisfied)
                    active = True
                    for l in c:
                        ov = abs(l) - 1
                        if (l > 0 and assignment[ov] == 1) or (l < 0 and assignment[ov] == -1):
                            active = False
                            break
                    if active:
                        var_scores[v] += activity[v]

            # Find best variable
            best_var = -1
            best_score = -1
            for v in range(n):
                if assignment[v] == 0 and var_scores[v] > best_score:
                    best_score = var_scores[v]
                    best_var = v

            if best_var == -1:
                # All assigned
                return assignment[:]

            # Branch on best_var
            current_level += 1

            # Try True first
            if assign(best_var, 1, current_level):
                if propagate():
                    result = dpll_search()
                    if result is not None:
                        return result

            # Try False
            unassign_all_to(current_level)
            if assign(best_var, -1, current_level):
                if propagate():
                    result = dpll_search()
                    if result is not None:
                        return result

            # Backtrack
            unassign_all_to(current_level)
            current_level -= 1

            # Decay activities
            for v in range(n):
                activity[v] *= var_inc
            if inc * var_inc < 1e-3:
                for v in range(n):
                    activity[v] *= 1000.0
                inc *= 1000.0

        return None

    # Try DPLL
    result = dpll_search()

    if result is not None:
        # Convert to boolean list
        return [a == 1 for a in result]

    # Fallback: WalkSAT
    def walk_sat():
        rng = random.Random(42)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        def count_unsat(assign):
            cnt = 0
            for c in clause_lits:
                sat = False
                for lit in c:
                    v = abs(lit) - 1
                    if (lit > 0 and assign[v]) or (lit < 0 and not assign[v]):
                        sat = True
                        break
                if not sat:
                    cnt += 1
            return cnt

        best_assign = assignment[:]
        best_uc = count_unsat(assignment)

        max_steps = 500000
        for step in range(max_steps):
            if time.time() > DEADLINE:
                break
            if best_uc == 0:
                return best_assign

            # Pick random unsat clause
            unsat_clauses = []
            for i, c in enumerate(clause_lits):
                sat = False
                for lit in c:
                    v = abs(lit) - 1
                    if (lit > 0 and assignment[v]) or (lit < 0 and not assignment[v]):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(i)
            if not unsat_clauses:
                return assignment
            ci = rng.choice(unsat_clauses)
            c = clause_lits[ci]

            # Pick variable to flip using phase saving heuristic
            # Try to flip a literal that would satisfy the clause
            # with some probability, otherwise random
            if rng.random() < 0.5:
                # Pick a random literal in the clause
                lit = rng.choice(c)
            else:
                # Pick the literal with minimum flips
                min_flips = 100
                best_lit = c[0]
                for lit in c:
                    v = abs(lit) - 1
                    # Count how many other clauses this variable is in
                    flips = len(var_occurrences[v])
                    if flips < min_flips:
                        min_flips = flips
                        best_lit = lit
                lit = best_lit

            v = abs(lit) - 1
            # Flip
            assignment[v] = not assignment[v]

            uc = count_unsat(assignment)
            if uc < best_uc:
                best_uc = uc
                best_assign = assignment[:]
            else:
                # With some probability, keep the flip (noise)
                if rng.random() > 0.5:
                    assignment[v] = not assignment[v]

        return best_assign

    result = walk_sat()
    return result
