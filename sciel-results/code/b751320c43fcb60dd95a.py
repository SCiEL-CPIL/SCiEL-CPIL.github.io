"""Hybrid DPLL + Simulated Annealing 3-SAT solver for planted instances."""
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0  # 10s safety margin before 300s limit

    # Preprocess
    clauses = [list(c) for c in clauses]
    m = len(clauses)

    # Build occurrence index
    occ = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            occ[v].append((ci, sign))

    def dpll_with_unit_propagation():
        """DPLL with unit propagation. Returns assignment or None if UNSAT/timeout."""
        # Use a simplified DPLL: try random variable ordering with unit propagation
        assignment = [None] * n  # None = unassigned
        active_clauses = [set(c) for c in clauses]
        
        def unit_propagate():
            """Perform unit propagation. Returns False if conflict, True otherwise."""
            changed = True
            while changed:
                changed = False
                for ci in range(len(active_clauses)):
                    cl = active_clauses[ci]
                    if not cl:
                        return False  # empty clause = conflict
                    if len(cl) == 1:
                        lit = next(iter(cl))
                        v = abs(lit) - 1
                        val = lit > 0
                        if assignment[v] is not None and assignment[v] != val:
                            return False  # conflict
                        if assignment[v] is None:
                            assignment[v] = val
                            changed = True
                            # Remove this literal from all clauses, remove negation
                            neg_lit = -lit
                            for cj in range(len(active_clauses)):
                                if cj == ci:
                                    continue
                                if lit in active_clauses[cj]:
                                    active_clauses[cj].discard(lit)
                                if neg_lit in active_clauses[cj]:
                                    active_clauses[cj].discard(neg_lit)
                            active_clauses[ci].clear()
            return True

        def check_satisfied():
            for cl in active_clauses:
                if not cl:
                    continue
                for lit in cl:
                    v = abs(lit) - 1
                    if assignment[v] is not None and assignment[v] == (lit > 0):
                        break
                else:
                    return False
            return True

        # Initial unit propagation
        if not unit_propagate():
            return None

        if check_satisfied():
            # Fill remaining unassigned vars randomly
            for i in range(n):
                if assignment[i] is None:
                    assignment[i] = random.random() < 0.5
            return assignment

        # Try a few DPLL branches with random variable selection
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Save state
            saved_assignment = assignment[:]
            saved_clauses = [set(c) for c in active_clauses]
            
            # Pick a random unassigned variable
            unassigned = [i for i in range(n) if assignment[i] is None]
            if not unassigned:
                break
            v = random.choice(unassigned)
            
            # Try both values
            for val in [True, False]:
                assignment[v] = val
                lit = v + 1 if val else -(v + 1)
                neg_lit = -lit
                
                # Simplify clauses
                new_clauses = []
                conflict = False
                for cl in saved_clauses:
                    if lit in cl:
                        continue  # clause satisfied
                    new_cl = cl - {neg_lit}
                    if not new_cl:
                        conflict = True
                        break
                    new_clauses.append(new_cl)
                
                if conflict:
                    continue
                
                # Try unit propagation on simplified problem
                test_assignment = saved_assignment[:]
                test_assignment[v] = val
                test_clauses = new_clauses
                
                tp_changed = True
                tp_conflict = False
                while tp_changed and not tp_conflict:
                    tp_changed = False
                    for ci in range(len(test_clauses)):
                        cl = test_clauses[ci]
                        if not cl:
                            tp_conflict = True
                            break
                        if len(cl) == 1:
                            ulit = next(iter(cl))
                            uv = abs(ulit) - 1
                            uval = ulit > 0
                            if test_assignment[uv] is not None and test_assignment[uv] != uval:
                                tp_conflict = True
                                break
                            if test_assignment[uv] is None:
                                test_assignment[uv] = uval
                                tp_changed = True
                                neg_ulit = -ulit
                                for cj in range(len(test_clauses)):
                                    if cj == ci:
                                        continue
                                    if ulit in test_clauses[cj]:
                                        test_clauses[cj].discard(ulit)
                                    if neg_ulit in test_clauses[cj]:
                                        test_clauses[cj].discard(neg_ulit)
                                test_clauses[ci].clear()
                
                if not tp_conflict:
                    # Check if all clauses satisfied
                    all_sat = True
                    for cl in test_clauses:
                        if not cl:
                            continue
                        sat = False
                        for ulit in cl:
                            uv = abs(ulit) - 1
                            if test_assignment[uv] is not None and test_assignment[uv] == (ulit > 0):
                                sat = True
                                break
                        if not sat:
                            all_sat = False
                            break
                    if all_sat:
                        for i in range(n):
                            if test_assignment[i] is None:
                                test_assignment[i] = random.random() < 0.5
                        return test_assignment
            
            # Restore state
            assignment = saved_assignment
            active_clauses = saved_clauses

        return None

    def simulated_annealing_trial(rng, max_steps, t0=1.0, t1=0.01):
        """Simulated annealing for 3-SAT. Returns (unsat_count, assignment)."""
        assign = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute initial energy
        clause_unsat = [False] * m
        for ci, cl in enumerate(clauses):
            sat = False
            for lit in cl:
                v = abs(lit) - 1
                if assign[v] == (lit > 0):
                    sat = True
                    break
            clause_unsat[ci] = not sat
        unsat_count = sum(clause_unsat)
        
        if unsat_count == 0:
            return 0, assign

        best_unsat = unsat_count
        best_assign = assign[:]
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if unsat_count == 0:
                return 0, assign

            # Temperature schedule: exponential decay
            t = t0 * math.exp(-step / max_steps * 5)
            
            # Pick a random variable to flip
            v = rng.randint(0, n - 1)
            
            # Compute delta energy
            old_val = assign[v]
            new_val = not old_val
            delta = 0
            for (cidx, sign) in occ[v]:
                cl = clauses[cidx]
                # Check if clause is currently satisfied by other literals
                other_sat = False
                for lit in cl:
                    v2 = abs(lit) - 1
                    if v2 == v:
                        continue
                    if assign[v2] == (lit > 0):
                        other_sat = True
                        break
                
                # Current state of clause
                currently_satisfied = not clause_unsat[cidx]
                # Will clause be satisfied after flip?
                this_lit_satisfies = (new_val == (sign > 0))
                will_be_satisfied = other_sat or this_lit_satisfies
                
                if currently_satisfied and not will_be_satisfied:
                    delta += 1  # breaks a satisfied clause
                elif not currently_satisfied and will_be_satisfied:
                    delta -= 1  # satisfies an unsatisfied clause
            
            # Accept with probability based on temperature
            if delta <= 0:
                accept = True
            else:
                accept = rng.random() < math.exp(-delta / t)
            
            if accept:
                assign[v] = new_val
                # Update clause_unsat
                for (cidx, sign) in occ[v]:
                    cl = clauses[cidx]
                    sat = False
                    for lit in cl:
                        v2 = abs(lit) - 1
                        if assign[v2] == (lit > 0):
                            sat = True
                            break
                    new_unsat = not sat
                    if new_unsat and not clause_unsat[cidx]:
                        unsat_count += 1
                    elif not new_unsat and clause_unsat[cidx]:
                        unsat_count -= 1
                    clause_unsat[cidx] = new_unsat
                
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assign = assign[:]
        
        return best_unsat, best_assign

    # Strategy: First try DPLL, then fall back to simulated annealing
    result = dpll_with_unit_propagation()
    if result is not None:
        return result

    # DPLL failed, use simulated annealing with restarts
    rng = random.Random(42)
    best_unsat = m + 1
    best_assignment = None
    trial_num = 0

    while time.time() < deadline:
        # Adaptive step budget
        max_steps = 10000 + trial_num * 2000
        if trial_num > 10:
            max_steps = 25000

        unsat_count, assign = simulated_annealing_trial(rng, max_steps)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assign[:]
        if unsat_count == 0:
            return assign

        trial_num += 1

    return best_assignment if best_assignment is not None else [False] * n
