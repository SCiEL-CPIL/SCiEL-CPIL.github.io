"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s kill

    # Build a graph for spectral initialization
    # For each pair of variables that appear together in a clause,
    # create an edge with weight indicating whether they should agree or disagree
    # If literal a and literal b are in the same clause, they should NOT both be false
    # This creates a constraint that can be modeled as a weighted graph
    
    # First, build adjacency structure for spectral method
    # We'll use a soft Max-Cut formulation where we want to minimize disagreements
    # that violate clause constraints
    
    # Build incidence: for each variable pair, count how many clauses they appear in
    # and whether they need to agree or disagree
    pair_counts = {}  # (var1, var2) -> [agree_count, disagree_count]
    
    for clause in clauses:
        vars_in_clause = set(abs(l) - 1 for l in clause)
        lits_in_clause = [(abs(l) - 1, l > 0) for l in clause]
        
        # For each pair of literals in the clause
        for i in range(len(lits_in_clause)):
            for j in range(i + 1, len(lits_in_clause)):
                v1, p1 = lits_in_clause[i]
                v2, p2 = lits_in_clause[j]
                if v1 == v2:
                    continue
                key = (min(v1, v2), max(v1, v2))
                if key not in pair_counts:
                    pair_counts[key] = [0, 0]
                # If both literals are positive or both negative, they should agree
                # If one is positive and one negative, they should disagree
                if p1 == p2:
                    pair_counts[key][0] += 1  # agree
                else:
                    pair_counts[key][1] += 1  # disagree
    
    # Build Laplacian matrix for spectral bisection
    # Weight = agreement - disagreement (positive means they should agree)
    L = np.zeros((n, n))
    degree = np.zeros(n)
    
    for (v1, v2), (agree, disagree) in pair_counts.items():
        weight = agree - disagree
        L[v1, v2] = -weight
        L[v2, v1] = -weight
        degree[v1] += weight
        degree[v2] += weight
    
    # Add diagonal
    for i in range(n):
        L[i, i] = degree[i]
    
    # Compute Fiedler vector (second smallest eigenvector)
    # Use numpy's eigh for symmetric matrices
    try:
        if n > 2:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
            fiedler = eigenvectors[:, 1]
            # Initialize assignment based on sign of Fiedler vector
            # Positive -> True, Negative -> False
            assignment = [fiedler[i] > 0 for i in range(n)]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
    except Exception:
        # Fallback to random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Apply unit propagation on top of spectral initialization
    # Convert to None-based for propagation
    assignment_none = [None] * n
    for i in range(n):
        assignment_none[i] = assignment[i]
    
    def propagate():
        """Unit propagation: assign literals forced by unit clauses."""
        changed = True
        while changed:
            changed = False
            for clause in clauses:
                unassigned = [l for l in clause if assignment_none[abs(l) - 1] is None]
                satisfied = any(assignment_none[abs(l) - 1] == (l > 0) for l in clause)
                if satisfied:
                    continue
                if len(unassigned) == 0:
                    return False  # conflict
                if len(unassigned) == 1:
                    l = unassigned[0]
                    if assignment_none[abs(l) - 1] is None:
                        assignment_none[abs(l) - 1] = (l > 0)
                        changed = True
        return True
    
    if propagate():
        # Fill any remaining unassigned variables using spectral hint
        for i in range(n):
            if assignment_none[i] is None:
                # Use the Fiedler vector if available, else random
                try:
                    assignment_none[i] = fiedler[i] > 0
                except NameError:
                    assignment_none[i] = rng.random() < 0.5
        assignment = assignment_none
    else:
        # Conflict during propagation; fall back to pure random
        assignment = [rng.random() < 0.5 for _ in range(n)]
    
    def is_satisfied(clause):
        return any(assignment[abs(l) - 1] == (l > 0) for l in clause)

    def count_unsat():
        return sum(1 for c in clauses if not is_satisfied(c))

    # Track unsatisfied clauses for efficiency
    def get_unsat_clauses():
        return [c for c in clauses if not is_satisfied(c)]

    # WalkSAT with gain/loss heuristic
    p = 0.55  # probability of using gain heuristic
    max_steps = 50000
    restarts = 0
    max_restarts = 50

    while time.time() < deadline and restarts < max_restarts:
        unsat = get_unsat_clauses()
        if not unsat:
            break

        for step in range(max_steps):
            if time.time() >= deadline:
                break
            unsat = get_unsat_clauses()
            if not unsat:
                break

            # Pick a random unsatisfied clause
            clause = rng.choice(unsat)

            # Compute gain/loss for each literal in clause
            best_var = -1
            best_gain = -1000
            best_is_gain = False

            for l in clause:
                var = abs(l) - 1
                # Count how many unsatisfied clauses would be satisfied
                # and how many satisfied clauses would become unsatisfied
                gain = 0
                loss = 0
                for c in clauses:
                    if is_satisfied(c):
                        # Would this flip make it unsatisfied?
                        # Only if the literal l is the only satisfied literal in c
                        has_l = any(abs(ll) - 1 == var for ll in c)
                        if has_l:
                            # Check if l is the only satisfied literal
                            only_sat = all(
                                (assignment[abs(ll) - 1] != (ll > 0)) or (abs(ll) - 1 == var)
                                for ll in c
                            )
                            if only_sat:
                                loss += 1
                    else:
                        # Would this flip satisfy it?
                        if any(abs(ll) - 1 == var and (ll > 0) == (not assignment[var]) for ll in c):
                            gain += 1

                net = gain - loss
                if net > best_gain:
                    best_gain = net
                    best_var = var
                    best_is_gain = gain > 0

            # With probability p, use the best gain variable; otherwise random
            if best_var >= 0 and (rng.random() < p or best_gain > 0):
                var_to_flip = best_var
            else:
                # Random variable from the clause
                l = rng.choice(clause)
                var_to_flip = abs(l) - 1

            assignment[var_to_flip] = not assignment[var_to_flip]

        # If we exhausted steps without solving, restart
        unsat = get_unsat_clauses()
        if not unsat:
            break
        restarts += 1
        # Reinitialize using spectral method for better restart quality
        try:
            if n > 2:
                # Recompute Fiedler vector with slight perturbation for diversity
                L_perturbed = L.copy()
                noise = rng.gauss(0, 0.1, (n, n))
                noise = (noise + noise.T) / 2  # symmetric
                L_perturbed += noise
                eigenvalues, eigenvectors = np.linalg.eigh(L_perturbed)
                fiedler_restart = eigenvectors[:, 1]
                assignment = [fiedler_restart[i] > 0 for i in range(n)]
            else:
                assignment = [rng.random() < 0.5 for _ in range(n)]
        except Exception:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Apply unit propagation on restart
        assignment_none = [None] * n
        for i in range(n):
            assignment_none[i] = assignment[i]
        
        if propagate():
            for i in range(n):
                if assignment_none[i] is None:
                    try:
                        assignment_none[i] = fiedler[i] > 0
                    except NameError:
                        assignment_none[i] = rng.random() < 0.5
            assignment = assignment_none
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]

    return assignment
# EVOLVE-BLOCK-END
