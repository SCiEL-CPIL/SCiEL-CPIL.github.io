"""Improved 3-SAT solver using WalkSAT with spectral initialization and literal-count tracking."""

import random
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# Module-level state for time management
_start_time = time.time()
# Assume 300s total limit, leave 10s buffer for safety/overhead
_DEADLINE = _start_time + 290.0


def solve(instance):
    """
    Solves a 3-SAT instance.
    Returns a list of booleans representing the assignment.
    """
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    # Pre-process: Map variable indices to clauses containing them
    var_to_clauses = [[] for _ in range(n)]
    # For each clause, store the position of each variable's literal
    # clause_var_pos[c_idx][var_idx] = position in clause (0,1,2) or -1 if not present
    clause_var_pos = []
    # clause_literals[c_idx] = list of (var_idx, is_pos)
    clause_literals = []

    for idx, clause in enumerate(clauses):
        var_pos = {}
        lits = []
        for pos, lit in enumerate(clause):
            var_idx = abs(lit) - 1
            is_pos = lit > 0
            var_pos[var_idx] = pos
            lits.append((var_idx, is_pos))
        clause_var_pos.append(var_pos)
        clause_literals.append(lits)
        for lit in clause:
            var_idx = abs(lit) - 1
            var_to_clauses[var_idx].append(idx)

    def compute_state(assignment):
        """Compute clause_satisfied and literal_true_counts for a given assignment."""
        clause_satisfied = [False] * m
        lit_true_counts = [0] * m  # number of true literals per clause
        for i, lits in enumerate(clause_literals):
            cnt = 0
            for var_idx, is_pos in lits:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            lit_true_counts[i] = cnt
            clause_satisfied[i] = (cnt > 0)
        unsat_count = sum(1 for s in clause_satisfied if not s)
        return clause_satisfied, lit_true_counts, unsat_count

    # Spectral initialization: build a weighted graph from literal conflicts
    # For each clause, pairs of literals with opposite signs create an edge
    # The Fiedler vector (2nd eigenvector of Laplacian) gives a soft partition
    def spectral_init(clauses, n):
        """Use Max-Cut spectral heuristic to initialize assignment."""
        # Build adjacency: edge between var_i and var_j if they appear in a clause
        # with "conflicting" literals (one positive, one negative)
        # Weight = number of such conflicting pairs
        row_idx = []
        col_idx = []
        data = []
        
        for clause in clauses:
            # For 3-CNF, check all pairs in the clause
            for i in range(len(clause)):
                for j in range(i+1, len(clause)):
                    lit_i = clause[i]
                    lit_j = clause[j]
                    var_i = abs(lit_i) - 1
                    var_j = abs(lit_j) - 1
                    if var_i == var_j:
                        continue
                    # Conflicting if one is positive and other is negative
                    # This means they can't both be true simultaneously
                    if (lit_i > 0 and lit_j < 0) or (lit_i < 0 and lit_j > 0):
                        row_idx.append(var_i)
                        col_idx.append(var_j)
                        data.append(1.0)
                        row_idx.append(var_j)
                        col_idx.append(var_i)
                        data.append(1.0)
        
        if not row_idx:
            # No conflicting pairs, just return random
            return [random.random() < 0.5 for _ in range(n)]
        
        # Build Laplacian
        deg = np.zeros(n)
        for i, j, w in zip(row_idx, col_idx, data):
            deg[i] += w
        
        # Laplacian L = D - A
        # Build sparse matrices
        L_rows = []
        L_cols = []
        L_data = []
        
        # Diagonal
        for i in range(n):
            L_rows.append(i)
            L_cols.append(i)
            L_data.append(deg[i])
        
        # Off-diagonal (negative weights)
        for i, j, w in zip(row_idx, col_idx, data):
            L_rows.append(i)
            L_cols.append(j)
            L_data.append(-w)
        
        L = csr_matrix((L_data, (L_rows, L_cols)), shape=(n, n))
        
        # Get Fiedler vector (2nd smallest eigenvalue's eigenvector)
        try:
            if n > 2:
                eigenvalues, eigenvectors = eigsh(L, k=2, which='SM', return_eigenvectors=True)
                # Fiedler vector is the second one
                fiedler = eigenvectors[:, 1]
            else:
                fiedler = np.ones(n) / np.sqrt(n)
        except Exception:
            return [random.random() < 0.5 for _ in range(n)]
        
        # Threshold at 0 to get binary assignment
        # Variables with fiedler[i] > 0 -> True, else False
        # Add small random perturbation to break ties
        fiedler = fiedler + 0.01 * np.random.randn(n)
        assignment = [fiedler[i] > 0 for i in range(n)]
        return assignment
    
    # Use spectral initialization
    assignment = spectral_init(clauses, n)
    clause_satisfied, lit_true_counts, unsat_count = compute_state(assignment)

    # Best solution tracking
    best_assignment = assignment.copy()
    best_unsat_count = unsat_count

    if unsat_count == 0:
        return assignment

    # WalkSAT with restarts and adaptive parameters
    max_restarts = 200
    base_steps = 3000
    flip_threshold = 0.35  # 35% random, 65% best (more exploitation)
    
    # Maintain list of unsatisfied clause indices for O(1) random access
    def get_unsat_list(clause_satisfied, m):
        return [i for i in range(m) if not clause_satisfied[i]]

    # Pre-compute for each variable, the list of (clause_idx, position_in_clause)
    var_clause_pos = [[] for _ in range(n)]
    for c_idx in range(m):
        for var_idx, pos in clause_var_pos[c_idx].items():
            var_clause_pos[var_idx].append((c_idx, pos))

    for restart in range(max_restarts):
        # Check time
        if time.time() > _DEADLINE - 2:
            return best_assignment

        # Reset to random assignment for each restart
        assignment = [random.random() < 0.5 for _ in range(n)]
        clause_satisfied, lit_true_counts, unsat_count = compute_state(assignment)

        if unsat_count == 0:
            return assignment

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()

        # Adaptive steps: more steps early, fewer later
        steps = base_steps

        for step in range(steps):
            # Check time every 100 steps for efficiency
            if step % 100 == 0 and time.time() > _DEADLINE - 2:
                return best_assignment

            if unsat_count == 0:
                return assignment

            # Build list of unsat indices
            unsat_indices = [i for i in range(m) if not clause_satisfied[i]]

            # Pick random unsatisfied clause
            clause_idx = random.choice(unsat_indices)
            clause = clause_literals[clause_idx]

            # Decide: random flip or best flip
            if random.random() < flip_threshold:
                # Random flip: pick random variable from clause
                var_idx, is_pos = random.choice(clause)
            else:
                # Best flip: pick variable that minimizes increase in unsat count
                # For 3-CNF, delta computation is fast with lit_true_counts
                best_var = clause[0][0]
                best_delta = float('inf')

                for var_idx, is_pos in clause:
                    # Calculate delta if we flip var_idx
                    delta = 0
                    for c_idx, pos in var_clause_pos[var_idx]:
                        # Current state of this clause
                        old_cnt = lit_true_counts[c_idx]
                        old_sat = old_cnt > 0

                        # Was the literal for var_idx in this clause true?
                        v, p = clause_literals[c_idx][pos]
                        lit_was_true = (assignment[v] == p)

                        # New count after flipping
                        if lit_was_true:
                            new_cnt = old_cnt - 1
                        else:
                            new_cnt = old_cnt + 1

                        new_sat = new_cnt > 0

                        if old_sat and not new_sat:
                            delta += 1
                        elif not old_sat and new_sat:
                            delta -= 1

                    if delta < best_delta:
                        best_delta = delta
                        best_var = var_idx
                    elif delta == best_delta and random.random() < 0.5:
                        best_var = var_idx

                var_idx = best_var

            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]

            # Update clause satisfaction efficiently using lit_true_counts
            for c_idx, pos in var_clause_pos[var_idx]:
                v, p = clause_literals[c_idx][pos]
                # The literal truth value just flipped
                if assignment[v] == p:
                    # Literal is now true
                    lit_true_counts[c_idx] += 1
                    if not clause_satisfied[c_idx]:
                        clause_satisfied[c_idx] = True
                        unsat_count -= 1
                else:
                    # Literal is now false
                    lit_true_counts[c_idx] -= 1
                    if lit_true_counts[c_idx] == 0:
                        clause_satisfied[c_idx] = False
                        unsat_count += 1

            if unsat_count == 0:
                return assignment

            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()

    return best_assignment
