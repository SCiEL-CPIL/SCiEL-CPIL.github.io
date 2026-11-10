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
    pair_counts = {}  # (var1, var2) -> [agree_count, disagree_count]
    
    for clause in clauses:
        vars_in_clause = set(abs(l) - 1 for l in clause)
        lits_in_clause = [(abs(l) - 1, l > 0) for l in clause]
        
        for i in range(len(lits_in_clause)):
            for j in range(i + 1, len(lits_in_clause)):
                v1, p1 = lits_in_clause[i]
                v2, p2 = lits_in_clause[j]
                if v1 == v2:
                    continue
                key = (min(v1, v2), max(v1, v2))
                if key not in pair_counts:
                    pair_counts[key] = [0, 0]
                if p1 == p2:
                    pair_counts[key][0] += 1  # agree
                else:
                    pair_counts[key][1] += 1  # disagree
    
    # Build Laplacian matrix for spectral bisection
    L = np.zeros((n, n))
    degree = np.zeros(n)
    
    for (v1, v2), (agree, disagree) in pair_counts.items():
        weight = agree - disagree
        L[v1, v2] = -weight
        L[v2, v1] = -weight
        degree[v1] += weight
        degree[v2] += weight
    
    for i in range(n):
        L[i, i] = degree[i]
    
    # Compute Fiedler vector for spectral initialization
    try:
        if n > 2:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            fiedler = eigenvectors[:, 1]
            assignment = [fiedler[i] > 0 for i in range(n)]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
    except Exception:
        assignment = [rng.random() < 0.5 for _ in range(n)]

    # Apply unit propagation
    assignment_none = [None] * n
    for i in range(n):
        assignment_none[i] = assignment[i]
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for clause in clauses:
                unassigned = [l for l in clause if assignment_none[abs(l) - 1] is None]
                satisfied = any(assignment_none[abs(l) - 1] == (l > 0) for l in clause)
                if satisfied:
                    continue
                if len(unassigned) == 0:
                    return False
                if len(unassigned) == 1:
                    l = unassigned[0]
                    if assignment_none[abs(l) - 1] is None:
                        assignment_none[abs(l) - 1] = (l > 0)
                        changed = True
        return True
    
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

    def is_satisfied(clause):
        return any(assignment[abs(l) - 1] == (l > 0) for l in clause)

    def get_unsat_clauses():
        return [c for c in clauses if not is_satisfied(c)]

    # Precompute clause literal variables for faster access
    clause_vars = [[abs(l) - 1 for l in c] for c in clauses]
    clause_signs = [[l > 0 for l in c] for c in clauses]
    
    # Build variable-to-clauses index for faster gain/loss computation
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            var_to_clauses[abs(l) - 1].append(i)

    p = 0.55
    max_steps = 30000
    restarts = 0
    max_restarts = 100

    def walksat_round(assignment):
        """Run a WalkSAT round with improved gain/loss heuristic."""
        for step in range(max_steps):
            if time.time() >= deadline:
                break
            
            unsat = get_unsat_clauses()
            if not unsat:
                break

            # Pick a random unsatisfied clause
            clause_idx = rng.choice(unsat)
            clause = clauses[clause_idx]

            # Compute gain/loss for each literal in clause using indexed lookup
            best_var = -1
            best_gain = -1000

            for l in clause:
                var = abs(l) - 1
                gain = 0
                loss = 0
                
                # Check all clauses containing this variable
                for ci in var_to_clauses[var]:
                    c = clauses[ci]
                    c_sat = is_satisfied(c)
                    
                    if c_sat:
                        # Check if flipping var would make this clause unsatisfied
                        # Only if the current assignment of var is the only satisfied literal
                        only_sat = True
                        for ll in c:
                            v2 = abs(ll) - 1
                            if v2 != var:
                                if assignment[v2] == (ll > 0):
                                    only_sat = False
                                    break
                        if only_sat:
                            loss += 1
                    else:
                        # Check if flipping var would satisfy this clause
                        # The literal with var in c must become true after flip
                        for ll in c:
                            v2 = abs(ll) - 1
                            if v2 == var:
                                # After flip, assignment[var] changes
                                # We want (ll > 0) == (not assignment[var]) to become True
                                # i.e., (ll > 0) != assignment[var]
                                if (ll > 0) != assignment[var]:
                                    gain += 1
                                break

                net = gain - loss
                if net > best_gain:
                    best_gain = net
                    best_var = var

            # With probability p, use the best gain variable; otherwise random
            if best_var >= 0 and (rng.random() < p or best_gain > 0):
                var_to_flip = best_var
            else:
                l = rng.choice(clause)
                var_to_flip = abs(l) - 1

            assignment[var_to_flip] = not assignment[var_to_flip]

        return assignment

    while time.time() < deadline and restarts < max_restarts:
        unsat = get_unsat_clauses()
        if not unsat:
            break

        assignment = walksat_round(assignment)

        unsat = get_unsat_clauses()
        if not unsat:
            break
        
        restarts += 1
        
        # Reinitialize using spectral method for better restart quality
        try:
            if n > 2:
                L_perturbed = L.copy()
                noise = rng.gauss(0, 0.1, (n, n))
                noise = (noise + noise.T) / 2
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
