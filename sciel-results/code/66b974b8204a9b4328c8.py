"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from collections import defaultdict


def solve(instance):
    """Simulated annealing with clause-weighted variable selection, spectral initialization, and restarts for 3-SAT."""
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285  # safety margin before 330s kill

    # Precompute literal indices (0-based) and sign for each clause
    clause_lits = []
    for c in clauses:
        lits = [(abs(l) - 1, 1 if l > 0 else -1) for l in c]
        clause_lits.append(lits)

    # For each variable, track which clauses contain it
    pos_clauses = [[] for _ in range(n)]
    neg_clauses = [[] for _ in range(n)]
    for ci, lits in enumerate(clause_lits):
        for var, sign in lits:
            if sign == 1:
                pos_clauses[var].append(ci)
            else:
                neg_clauses[var].append(ci)

    # Variable occurrence counts for weighted selection
    var_occurrence = [len(pos_clauses[v]) + len(neg_clauses[v]) for v in range(n)]
    total_occurrence = sum(var_occurrence)
    if total_occurrence == 0:
        total_occurrence = 1

    def spectral_init(rng):
        """Initialize assignment using spectral method on literal co-occurrence matrix."""
        # Build literal co-occurrence matrix
        # For each pair of literals in the same clause, they should be "compatible"
        # If both are positive or both negative, they correlate in the same direction
        # If one is positive and one negative, they correlate in opposite directions
        
        # We'll use a simpler approach: count how often each literal appears
        # and use that as a heuristic, combined with spectral analysis
        
        # Build a matrix where entry (i,j) counts co-occurrence of literals i and j
        # Literals are represented as (var, sign) -> index var*2 + (0 if sign==-1 else 1)
        
        # Actually, let's use a simpler spectral approach:
        # Build a matrix M where M[v1][v2] = number of clauses where v1 and v2 appear together
        # with the same sign requirement, minus those with opposite sign requirement
        
        # For planted 3-SAT, the planted solution has certain structure.
        # Let's try: for each variable, compute a score based on clause structure
        
        # Simpler spectral method:
        # Build adjacency matrix A where A[i][j] = 1 if variables i and j appear together in a clause
        # with compatible signs (both positive or both negative), -1 if incompatible
        # Then the principal eigenvector gives a soft assignment
        
        # For efficiency with n=150, we can build this matrix
        A = np.zeros((n, n), dtype=np.float64)
        
        for ci, lits in enumerate(clause_lits):
            for i in range(len(lits)):
                for j in range(i+1, len(lits)):
                    v1, s1 = lits[i]
                    v2, s2 = lits[j]
                    # If signs are the same, they're "compatible" (both want to be true or both want to be false)
                    # If signs differ, they're "incompatible"
                    if s1 == s2:
                        A[v1][v2] += 1
                        A[v2][v1] += 1
                    else:
                        A[v1][v2] -= 1
                        A[v2][v1] -= 1
        
        # Add self-loops based on occurrence frequency
        for v in range(n):
            A[v][v] += 0.1 * var_occurrence[v]
        
        # Compute principal eigenvector
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(A)
            # Principal eigenvector is the one with largest eigenvalue
            idx = np.argmax(eigenvalues)
            v_vec = eigenvectors[:, idx]
            
            # Convert to assignment: positive values -> True, negative -> False
            assignment = [bool(x > 0) for x in v_vec]
        except Exception:
            # Fallback to random
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        return assignment

    def init_assignment(rng, use_spectral=True):
        """Initialize assignment with spectral method or clause-weighted bias."""
        if use_spectral:
            return spectral_init(rng)
        else:
            # Clause-weighted bias initialization
            assignment = []
            for v in range(n):
                if rng.random() < 0.3:
                    pos_count = len(pos_clauses[v])
                    neg_count = len(neg_clauses[v])
                    if pos_count > neg_count:
                        assignment.append(True)
                    elif neg_count > pos_count:
                        assignment.append(False)
                    else:
                        assignment.append(rng.random() < 0.5)
                else:
                    assignment.append(rng.random() < 0.5)
            return assignment

    def compute_clause_sat(assignment, clause_lits):
        """Compute satisfaction count for each clause."""
        clause_sat = [0] * m
        for ci, lits in enumerate(clause_lits):
            sat_count = 0
            for var, sign in lits:
                if assignment[var] == (sign == 1):
                    sat_count += 1
            clause_sat[ci] = sat_count
        return clause_sat

    def weighted_var_select(assignment, clause_sat, pos_clauses, neg_clauses, var_occurrence, rng, unsat_count):
        """Select variable with bias toward those in unsatisfied clauses."""
        if unsat_count == 0:
            return rng.randint(0, n - 1)

        # Build weight for each variable based on unsatisfied clauses
        weights = [1.0] * n  # base weight
        for ci in range(m):
            if clause_sat[ci] == 0:
                # Unsat clause: boost all its variables
                for var, sign in clause_lits[ci]:
                    weights[var] += 3.0  # strong boost for vars in unsat clauses

        # Also boost vars that appear in many clauses
        for v in range(n):
            weights[v] += 0.1 * var_occurrence[v]

        # Normalize and sample
        total_w = sum(weights)
        if total_w <= 0:
            return rng.randint(0, n - 1)

        # Sample using simple random walk for speed
        r = rng.random() * total_w
        cumulative = 0.0
        for v in range(n):
            cumulative += weights[v]
            if cumulative >= r:
                return v
        return n - 1

    def sa_search(assignment, clause_sat, unsat_count, deadline, rng, max_steps_per_phase):
        """Run one phase of simulated annealing."""
        T_start = 1.5
        T_end = 0.005
        step = 0
        best_unsat_local = unsat_count
        best_assignment_local = assignment[:]

        while time.time() < deadline and step < max_steps_per_phase:
            progress = min(step / max_steps_per_phase, 1.0)
            T = T_start * math.exp(-8.0 * progress)
            if T < T_end:
                T = T_end

            if unsat_count == 0:
                return assignment, clause_sat, 0, best_unsat_local, best_assignment_local

            # Weighted variable selection
            v = weighted_var_select(assignment, clause_sat, pos_clauses, neg_clauses, var_occurrence, rng, unsat_count)

            # Compute delta
            delta = 0
            if assignment[v]:
                for ci in pos_clauses[v]:
                    if clause_sat[ci] == 1:
                        delta += 1
            else:
                for ci in pos_clauses[v]:
                    if clause_sat[ci] == 0:
                        delta -= 1

            if not assignment[v]:
                for ci in neg_clauses[v]:
                    if clause_sat[ci] == 1:
                        delta += 1
            else:
                for ci in neg_clauses[v]:
                    if clause_sat[ci] == 0:
                        delta -= 1

            # Accept or reject
            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                assignment[v] = not assignment[v]
                if assignment[v]:
                    for ci in pos_clauses[v]:
                        clause_sat[ci] += 1
                    for ci in neg_clauses[v]:
                        clause_sat[ci] -= 1
                else:
                    for ci in neg_clauses[v]:
                        clause_sat[ci] += 1
                    for ci in pos_clauses[v]:
                        clause_sat[ci] -= 1

                unsat_count += delta

                if unsat_count < best_unsat_local:
                    best_unsat_local = unsat_count
                    best_assignment_local = assignment[:]

            step += 1

        return assignment, clause_sat, unsat_count, best_unsat_local, best_assignment_local

    def polish(assignment, deadline):
        """Polish: try flipping single vars that reduce unsat."""
        polish_assignment = assignment[:]
        polish_clause_sat = compute_clause_sat(polish_assignment, clause_lits)
        polish_unsat = sum(1 for s in polish_clause_sat if s == 0)

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for v in range(n):
                if polish_unsat == 0:
                    break
                # Try flipping v
                delta = 0
                if polish_assignment[v]:
                    for ci in pos_clauses[v]:
                        if polish_clause_sat[ci] == 1:
                            delta += 1
                else:
                    for ci in pos_clauses[v]:
                        if polish_clause_sat[ci] == 0:
                            delta -= 1
                if not polish_assignment[v]:
                    for ci in neg_clauses[v]:
                        if polish_clause_sat[ci] == 1:
                            delta += 1
                else:
                    for ci in neg_clauses[v]:
                        if polish_clause_sat[ci] == 0:
                            delta -= 1

                if delta < 0:
                    # Apply flip
                    polish_assignment[v] = not polish_assignment[v]
                    if polish_assignment[v]:
                        for ci in pos_clauses[v]:
                            polish_clause_sat[ci] += 1
                        for ci in neg_clauses[v]:
                            polish_clause_sat[ci] -= 1
                    else:
                        for ci in neg_clauses[v]:
                            polish_clause_sat[ci] += 1
                        for ci in pos_clauses[v]:
                            polish_clause_sat[ci] -= 1
                    polish_unsat += delta
                    improved = True

        return polish_assignment, polish_unsat

    # Main loop with restarts
    rng = random.Random(42)
    global_best_unsat = float('inf')
    global_best_assignment = None

    # Restart strategy: mix spectral and random initializations
    num_restarts = 10
    phase_time_budget = 285.0 / num_restarts

    for restart in range(num_restarts):
        phase_deadline = time.time() + phase_time_budget

        # Alternate between spectral and random initialization
        use_spectral = (restart % 2 == 0)
        assignment = init_assignment(rng, use_spectral=use_spectral)
        clause_sat = compute_clause_sat(assignment, clause_lits)
        unsat_count = sum(1 for s in clause_sat if s == 0)

        # Run SA phase
        assignment, clause_sat, unsat_count, best_unsat_local, best_assignment_local = sa_search(
            assignment, clause_sat, unsat_count, phase_deadline, rng, 200000
        )

        # Update global best
        if best_unsat_local < global_best_unsat:
            global_best_unsat = best_unsat_local
            global_best_assignment = best_assignment_local[:]

        if global_best_unsat == 0:
            return global_best_assignment

        # If we're close to solved, do a focused polish
        if global_best_unsat <= 10 and time.time() < deadline:
            polish_assignment, polish_unsat = polish(global_best_assignment, deadline)

            if polish_unsat < global_best_unsat:
                global_best_unsat = polish_unsat
                global_best_assignment = polish_assignment[:]

            if global_best_unsat == 0:
                return global_best_assignment

    # Final polish if time remains
    if global_best_assignment is not None and global_best_unsat > 0 and time.time() < deadline - 10:
        polish_assignment, polish_unsat = polish(global_best_assignment, deadline)
        if polish_unsat < global_best_unsat:
            global_best_unsat = polish_unsat
            global_best_assignment = polish_assignment[:]

    return global_best_assignment
# EVOLVE-BLOCK-END
