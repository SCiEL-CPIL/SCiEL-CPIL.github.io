"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285.0  # safety margin before 300s limit

    # Pre-process clauses into list of (var, is_positive) for speed
    # var is 0-indexed
    pre_clauses = []
    # Also build var_clauses: var -> list of (clause_index, is_positive)
    var_clauses = [[] for _ in range(n)]
    
    for ci, c in enumerate(clauses):
        pc = []
        for lit in c:
            if lit > 0:
                v = lit - 1
                pc.append((v, True))
                var_clauses[v].append((ci, True))
            else:
                v = -lit - 1
                pc.append((v, False))
                var_clauses[v].append((ci, False))
        pre_clauses.append(pc)

    m = len(pre_clauses)

    # Initial state tracking
    best_assignment = None
    best_unsat = m

    rng = random.Random(42)

    # Parameters for WalkSAT
    noise = 0.25
    max_steps_per_restart = 30000
    max_restarts = 2000  # High number, but we break on time

    # Precompute for each clause, the list of (var, is_pos) as a tuple for faster access
    clause_vars = [tuple(c) for c in pre_clauses]

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize clause_sat_count
        clause_sat_count = [0] * m
        for ci in range(m):
            cnt = 0
            for v, is_pos in clause_vars[ci]:
                if assignment[v] == is_pos:
                    cnt += 1
            clause_sat_count[ci] = cnt
            
        unsat_count = sum(1 for cnt in clause_sat_count if cnt == 0)

        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        # WalkSAT loop
        steps = 0
        while steps < max_steps_per_restart:
            if time.time() > deadline:
                break

            if unsat_count == 0:
                return assignment

            # Pick a random unsatisfied clause
            ci = rng.randrange(m)
            if clause_sat_count[ci] != 0:
                # Retry a few times, then fallback to scanning
                found = False
                for _ in range(20):
                    ci = rng.randrange(m)
                    if clause_sat_count[ci] == 0:
                        found = True
                        break
                if not found:
                    # Fallback: scan to find one
                    for i in range(m):
                        if clause_sat_count[i] == 0:
                            ci = i
                            found = True
                            break
                    if not found:
                        break

            clause = clause_vars[ci]
            
            # Variable selection heuristic: MinConflicts - pick var in most other unsat clauses
            best_v = -1
            best_unsat_occ = -1
            
            for v, is_pos in clause:
                unsat_occ = 0
                for cj, is_pos_j in var_clauses[v]:
                    if clause_sat_count[cj] == 0:
                        unsat_occ += 1
                if unsat_occ > best_unsat_occ:
                    best_unsat_occ = unsat_occ
                    best_v = v
            
            # Apply noise
            if rng.random() < noise:
                v = rng.choice(clause)[0]
            else:
                v = best_v

            # Flip variable v
            old_val = assignment[v]
            assignment[v] = not old_val
            
            # Update clause_sat_count and unsat_count incrementally
            for cj, is_pos_j in var_clauses[v]:
                was_sat = (old_val == is_pos_j)
                is_sat_now = (assignment[v] == is_pos_j)
                
                if was_sat and not is_sat_now:
                    clause_sat_count[cj] -= 1
                    if clause_sat_count[cj] == 0:
                        unsat_count += 1
                elif not was_sat and is_sat_now:
                    clause_sat_count[cj] += 1
                    if clause_sat_count[cj] == 1:
                        unsat_count -= 1

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
            
            steps += 1

    return best_assignment
# EVOLVE-BLOCK-END
