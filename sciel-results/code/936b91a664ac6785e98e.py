# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    rng = random.Random(0)
    start_time = time.time()
    TIME_LIMIT = 280.0  # leave margin under 300s

    # Preprocess: build clause list as lists of signed ints (1-based)
    # Remove tautological clauses (both l and -l in clause)
    clean_clauses = []
    for c in clauses:
        seen = set()
        taut = False
        for l in c:
            if -l in seen:
                taut = True
                break
            seen.add(l)
        if not taut:
            clean_clauses.append(list(c))
    clauses = clean_clauses

    # If no clauses, any assignment works
    if not clauses:
        return [False] * n

    # ---------- DPLL with watched literals, unit prop, pure literal ----------
    # Variables indexed 0..n-1. Literal sign: +var means True, -var means False.
    # assignment: list of None/True/False
    # watches: for each clause, two watched literal indices (into clause list)

    def dpll_solve(clauses, n, rng, time_limit, start_time):
        # returns assignment list of True/False or None if timeout/failed
        assignment = [None] * n
        # Build watched structure
        # For each clause, store as list of literals, and watch indices [0,1]
        # We'll maintain for each variable, list of (clause_idx, watch_idx) where that variable's literal is watched
        # Watched literal value: if literal l is watched, we track it.
        # Standard watched literal algorithm:
        #   watches[var] = list of (clause_idx, position) where clause[position] is a literal on var
        #   When flipping var, for each watch, check if the *other* watched literal is False;
        #   if so, find a new literal in the clause that is not False; if none, clause is unit -> propagate.

        # Simpler: implement unit propagation iteratively with a queue,
        # using a "false" set per clause count.
        # Given time, implement a straightforward but reasonably efficient DPLL:
        #   - Keep assignment
        #   - Keep list of clauses with their literals
        #   - Unit propagation: repeatedly find clauses with 0 false and 1 unassigned -> assign
        #   - Pure literal: if a var appears only with one sign in remaining clauses, assign to satisfy
        #   - Choose branching var (first unassigned, or heuristic)
        #   - Recurse with backtracking, node limit and time check.

        # For efficiency with 150 vars and typical clause counts (~420 for 3-SAT phase transition),
        # a simple DPLL with unit prop + pure literal + random restarts should work well
        # on planted instances.

        node_count = [0]
        max_nodes = 500000

        # Helper: check if clause is satisfied / has false count
        def clause_status(clause):
            # returns (satisfied, false_count, unassigned_list)
            sat = False
            false_count = 0
            unassigned = []
            for l in clause:
                v = abs(l) - 1
                val = assignment[v]
                if val is None:
                    unassigned.append(l)
                else:
                    if val == (l > 0):
                        sat = True
                        break
                    else:
                        false_count += 1
            return sat, false_count, unassigned

        def propagate():
            # Unit propagation with queue
            # Returns (success, conflict)
            # We'll use a worklist of clauses to check
            # To avoid O(m) scan each time, we maintain a queue of clauses that may have become unit
            # Simple approach: repeatedly scan all clauses until no change (expensive but ok for small n)
            # Better: maintain list of clauses, and a "dirty" set.
            # For simplicity and correctness, do iterative scan with a queue of clause indices.
            queue = list(range(len(clauses)))
            in_queue = [True] * len(clauses)
            changed = True
            while changed:
                changed = False
                new_queue = []
                for ci in queue:
                    clause = clauses[ci]
                    sat, false_count, unassigned = clause_status(clause)
                    if sat:
                        continue
                    if false_count >= len(clause):
                        return False, None  # conflict
                    if len(unassigned) == 1:
                        # unit clause: assign the unassigned literal
                        l = unassigned[0]
                        v = abs(l) - 1
                        if assignment[v] is None:
                            assignment[v] = (l > 0)
                            changed = True
                            # All clauses containing var v may now be unit/satisfied; add them
                            for cj, c in enumerate(clauses):
                                if not in_queue[cj] and any(abs(x) - 1 == v for x in c):
                                    in_queue[cj] = True
                                    new_queue.append(cj)
                queue = new_queue
                in_queue = [False] * len(clauses)
                for q in queue:
                    in_queue[q] = True
            return True, None

        def find_pure_literal():
            # Find a variable that appears with only one sign in unsatisfied clauses
            # Returns (var, sign) or (None, None)
            pos_vars = [set() for _ in range(n)]
            neg_vars = [set() for _ in range(n)]
            for ci, clause in enumerate(clauses):
                sat, false_count, unassigned = clause_status(clause)
                if sat:
                    continue
                for l in unassigned:
                    v = abs(l) - 1
                    if l > 0:
                        pos_vars[v].add(ci)
                    else:
                        neg_vars[v].add(ci)
            for v in range(n):
                if assignment[v] is None:
                    if pos_vars[v] and not neg_vars[v]:
                        return v, True
                    if neg_vars[v] and not pos_vars[v]:
                        return v, False
            return None, None

        def choose_var():
            # Heuristic: pick unassigned var in a shortest unsatisfied clause
            best_var = None
            best_len = 10**9
            for ci, clause in enumerate(clauses):
                sat, false_count, unassigned = clause_status(clause)
                if sat:
                    continue
                if len(unassigned) < best_len:
                    best_len = len(unassigned)
                    if best_len <= 1:
                        break
                    if unassigned:
                        # pick random from unassigned in this clause
                        best_var = abs(unassigned[rng.randrange(len(unassigned))]) - 1
            if best_var is None:
                # fallback: first unassigned
                for v in range(n):
                    if assignment[v] is None:
                        best_var = v
                        break
            return best_var

        def dpll(depth):
            if time.time() - start_time > time_limit:
                return None
            node_count[0] += 1
            if node_count[0] > max_nodes:
                return None

            # Propagate
            ok, _ = propagate()
            if not ok:
                return False  # conflict

            # Check if all assigned and all satisfied
            all_assigned = all(a is not None for a in assignment)
            if all_assigned:
                # verify
                for clause in clauses:
                    if not any(assignment[abs(l)-1] == (l>0) for l in clause):
                        return False
                return True

            # Pure literal elimination
            pv, psign = find_pure_literal()
            if pv is not None:
                assignment[pv] = psign
                # Recurse without branching (pure literal is safe)
                return dpll(depth)

            # Choose branching variable
            var = choose_var()
            if var is None:
                # All assigned; verify
                for clause in clauses:
                    if not any(assignment[abs(l)-1] == (l>0) for l in clause):
                        return False
                return True

            # Try True first (with random tie-break on restarts, but here deterministic order with phase saving)
            # Phase saving: try the value that was last tried as False? Keep simple: try True then False.
            assignment[var] = True
            result = dpll(depth + 1)
            if result is True:
                return True
            if result is None:
                assignment[var] = None
                return None  # timeout
            assignment[var] = False
            result = dpll(depth + 1)
            if result is True:
                return True
            if result is None:
                assignment[var] = None
                return None
            assignment[var] = None
            return False

        # Main loop with restarts
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                return None
            assignment = [None] * n
            # For restarts, add some random initial assignments to break symmetry
            if restart > 0:
                # Randomly assign up to 10% of variables
                num_rand = max(1, n // 10)
                vars_to_assign = rng.sample(range(n), min(num_rand, n))
                for v in vars_to_assign:
                    assignment[v] = rng.random() < 0.5
            # Propagate initial assignments
            ok, _ = propagate()
            if not ok:
                continue
            result = dpll(0)
            if result is True:
                return assignment
            if result is None:
                return None  # timeout
        return None

    # Try DPLL first
    dpll_result = dpll_solve(clauses, n, rng, TIME_LIMIT, start_time)
    if dpll_result is not None and all(a is not None for a in dpll_result):
        # Verify
        for clause in clauses:
            if not any(dpll_result[abs(l)-1] == (l>0) for l in clause):
                break
        else:
            return dpll_result

    # ---------- Fallback: WalkSAT with clause weights and min-unsat heuristic ----------
    assignment = [rng.random() < 0.5 for _ in range(n)]

    # Clause weights: increase weight for frequently unsatisfied clauses
    clause_weights = [1.0] * len(clauses)

    def satisfied(clause):
        return any(assignment[abs(l) - 1] == (l > 0) for l in clause)

    def unsat_clauses():
        return [c for c in clauses if not satisfied(c)]

    max_steps = 300000
    for step in range(max_steps):
        if time.time() - start_time > TIME_LIMIT:
            break
        unsat = unsat_clauses()
        if not unsat:
            break

        # Weighted random choice of unsat clause
        total_w = sum(clause_weights[i] for i, c in enumerate(clauses) if not satisfied(c))
        # Build list of (index, weight) for unsat clauses
        unsat_indices = [i for i, c in enumerate(clauses) if not satisfied(c)]
        weights = [clause_weights[i] for i in unsat_indices]
        r = rng.random() * sum(weights)
        acc = 0.0
        chosen_idx = unsat_indices[0]
        for idx, w in zip(unsat_indices, weights):
            acc += w
            if r <= acc:
                chosen_idx = idx
                break
        clause = clauses[chosen_idx]

        # Min-unsat heuristic: flip variable that minimizes number of newly unsatisfied clauses
        # For each literal in clause, compute delta
        best_var = None
        best_delta = 10**9
        for l in clause:
            v = abs(l) - 1
            # If we flip v, what's the change in unsat count?
            # Current assignment of v
            cur_val = assignment[v]
            new_val = not cur_val
            # For each clause containing v, check if it becomes satisfied/unsatisfied
            delta = 0
            for c in clauses:
                if v not in [abs(x)-1 for x in c]:
                    continue
                # Check status before and after flip
                # Before
                sat_before = False
                for x in c:
                    if abs(x)-1 == v:
                        if cur_val == (x > 0):
                            sat_before = True
                            break
                    else:
                        if assignment[abs(x)-1] == (x > 0):
                            sat_before = True
                            break
                # After
                sat_after = False
                for x in c:
                    if abs(x)-1 == v:
                        if new_val == (x > 0):
                            sat_after = True
                            break
                    else:
                        if assignment[abs(x)-1] == (x > 0):
                            sat_after = True
                            break
                if sat_before and not sat_after:
                    delta += 1
                elif not sat_before and sat_after:
                    delta -= 1
            if delta < best_delta:
                best_delta = delta
                best_var = v
                best_literal = l

        if best_var is not None:
            assignment[best_var] = not assignment[best_var]
            # Update clause weights: increase weight of unsat clauses
            # Decay all weights slightly, boost unsat
            for i, c in enumerate(clauses):
                if not satisfied(c):
                    clause_weights[i] = min(clause_weights[i] * 1.05 + 0.1, 100.0)
                else:
                    clause_weights[i] = max(clause_weights[i] * 0.95, 1.0)

            # Random restart with probability
            if rng.random() < 0.001:
                assignment = [rng.random() < 0.5 for _ in range(n)]
                clause_weights = [1.0] * len(clauses)

    # Final check
    for clause in clauses:
        if not satisfied(clause):
            break
    else:
        return assignment

    # If still unsatisfied, return best effort (assignment as-is)
    return assignment

# EVOLVE-BLOCK-END
