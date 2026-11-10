"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s kill

    # Preprocess: convert clauses to 0-based indices with sign
    # clause_list[i] = list of (var_idx, sign) where sign is +1 or -1
    clause_list = []
    for c in clauses:
        parsed = []
        for lit in c:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            parsed.append((var_idx, sign))
        clause_list.append(parsed)

    m = len(clause_list)

    # Build variable-to-clauses index
    var_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clause_list):
        for var_idx, sign in c:
            var_clauses[var_idx].append(i)

    best_assignment = None
    best_unsat = m + 1

    # Spectral initialization: build literal co-occurrence matrix
    # For planted 3-SAT, literals that co-occur in clauses with the same sign
    # relative to the planted solution are more likely to be true.
    # We build a matrix where M[i][j] counts co-occurrences of literals i and j.
    # Then we use the principal eigenvector to get a soft assignment.
    
    # We represent each literal as (var, sign). 
    # Actually, it's easier to work with variables directly.
    # Let's build a matrix A where A[i][j] is the sum of weights for variables i and j.
    # If two variables appear together in a clause, and their signs are compatible 
    # (e.g. both positive or both negative in a way that suggests they are both true), 
    # we add to A[i][j].
    
    # A simpler spectral approach for SAT:
    # Create a matrix B of size n x n.
    # For each clause (l1, l2, l3), where li is a literal on variable vi with sign si:
    # If we assume the planted solution makes all literals true, then vi = si.
    # So we want to find a vector v such that v_i * s_i is positive for all literals.
    # This is similar to finding a vector that aligns with the signs.
    
    # Let's use the following:
    # For each clause, for each pair of literals (u, v) in the clause:
    # If they have the same sign, add 1 to M[u.var][v.var]
    # If they have different signs, add 1 to M[u.var][v.var] but with a negative weight?
    # No, that's not quite right.
    
    # Alternative: Use the matrix of literal correlations.
    # Let's stick to a simpler heuristic first: count occurrences of each literal.
    # Then use a spectral method on the variable interaction graph.
    
    # Let's try a different spectral method:
    # Build a matrix M where M[i][j] = 1 if variables i and j appear together in a clause, 0 otherwise.
    # Then the principal eigenvector gives a "centrality" measure.
    # This doesn't directly give the assignment.
    
    # Better: Use the sign structure.
    # For each variable i, let s_i be the sum of signs of literals in clauses containing i.
    # If s_i > 0, variable i is likely true. If s_i < 0, variable i is likely false.
    # This is a simple heuristic, not spectral.
    
    # Let's implement a proper spectral method:
    # Create a matrix A of size n x n.
    # For each clause (l1, l2, l3):
    #   For each pair (li, lj):
    #     A[vi][vj] += si * sj
    #     A[vj][vi] += si * sj
    # Then the principal eigenvector of A gives a vector v.
    # If v_i > 0, set variable i to True; else False.
    
    # This is a reasonable approximation.
    
    if m > 0:
        # Build the matrix
        A = np.zeros((n, n), dtype=np.float64)
        for c in clause_list:
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    v1, s1 = c[i]
                    v2, s2 = c[j]
                    w = s1 * s2
                    A[v1][v2] += w
                    A[v2][v1] += w
        
        # Compute principal eigenvector
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(A)
            # eigh returns eigenvalues in ascending order
            principal_vec = eigenvectors[:, -1]
            
            # Create initial assignment based on principal eigenvector
            spectral_assignment = [x > 0 for x in principal_vec]
            
            # Evaluate spectral assignment
            unsat_count = 0
            clause_sat = [False] * m
            for i, c in enumerate(clause_list):
                sat = False
                for var_idx, sign in c:
                    if spectral_assignment[var_idx] == (sign == 1):
                        sat = True
                        break
                clause_sat[i] = sat
                if not sat:
                    unsat_count += 1
            
            if unsat_count == 0:
                return spectral_assignment
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = spectral_assignment[:]
                
            # Use spectral assignment as a starting point for local search
            # We'll run a few local search iterations from this point
            assignment = spectral_assignment[:]
            
            # Run local search from spectral initialization
            max_steps = 2000
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                if unsat_count == 0:
                    return assignment
                
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]
                
                # Select an unsat clause
                unsat_indices = [i for i in range(m) if not clause_sat[i]]
                if not unsat_indices:
                    return assignment
                
                # Pick a random unsat clause
                chosen_idx = unsat_indices[random.randint(0, len(unsat_indices) - 1)]
                c = clause_list[chosen_idx]
                
                # Find the best flip
                best_var = None
                best_delta = None
                for var_idx, sign in c:
                    delta = 0
                    for ci in var_clauses[var_idx]:
                        cur_sat = clause_sat[ci]
                        if cur_sat:
                            other_true = False
                            for v2, s2 in clause_list[ci]:
                                if v2 != var_idx and assignment[v2] == (s2 == 1):
                                    other_true = True
                                    break
                            if not other_true:
                                delta += 1
                        else:
                            if (not assignment[var_idx]) == (sign == 1):
                                delta -= 1
                    if best_delta is None or delta < best_delta:
                        best_delta = delta
                        best_var = var_idx
                
                # Flip the best variable
                old_val = assignment[best_var]
                assignment[best_var] = not old_val
                
                # Update clause satisfaction
                for ci in var_clauses[best_var]:
                    was_sat = clause_sat[ci]
                    now_sat = False
                    for v2, s2 in clause_list[ci]:
                        if assignment[v2] == (s2 == 1):
                            now_sat = True
                            break
                    if was_sat and not now_sat:
                        unsat_count += 1
                    elif not was_sat and now_sat:
                        unsat_count -= 1
                    clause_sat[ci] = now_sat
                    
        except Exception:
            pass

    # Multiple restarts with WalkSAT variants
    rng = random.Random(42)

    for restart in range(1000):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Compute initial unsat count and per-clause satisfaction status
        clause_sat = [False] * m
        unsat_count = 0
        for i, c in enumerate(clause_list):
            sat = False
            for var_idx, sign in c:
                if assignment[var_idx] == (sign == 1):
                    sat = True
                    break
            clause_sat[i] = sat
            if not sat:
                unsat_count += 1

        if unsat_count == 0:
            return assignment

        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        # Simulated annealing local search
        max_steps = 5000
        T_start = 3.0
        T_end = 0.1
        cooling_rate = (T_end / T_start) ** (1.0 / max_steps)

        for step in range(max_steps):
            if time.time() > deadline:
                break

            if unsat_count == 0:
                return assignment

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]

            # Current temperature
            T = T_start * (cooling_rate ** step)

            # Select an unsat clause
            unsat_indices = [i for i in range(m) if not clause_sat[i]]
            if not unsat_indices:
                return assignment

            # Weighted selection: prefer clauses closer to being satisfied
            weights = []
            for i in unsat_indices:
                false_count = 0
                for var_idx, sign in clause_list[i]:
                    if assignment[var_idx] != (sign == 1):
                        false_count += 1
                w = 3 - false_count + 1  # at least 1
                weights.append(w)

            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen_idx = unsat_indices[0]
            for j, i in enumerate(unsat_indices):
                cum += weights[j]
                if r <= cum:
                    chosen_idx = i
                    break

            c = clause_list[chosen_idx]

            # Evaluate all possible flips in the clause and pick one via SA
            best_var = None
            best_delta = None
            all_deltas = []  # list of (var_idx, delta)

            for var_idx, sign in c:
                # Compute delta if we flip var_idx
                delta = 0
                for ci in var_clauses[var_idx]:
                    cur_sat = clause_sat[ci]
                    if cur_sat:
                        # Check if any other literal in this clause is true
                        other_true = False
                        for v2, s2 in clause_list[ci]:
                            if v2 != var_idx and assignment[v2] == (s2 == 1):
                                other_true = True
                                break
                        if not other_true:
                            delta += 1  # becomes unsat
                    else:
                        # Currently unsat; after flip, is this literal now true?
                        if (not assignment[var_idx]) == (sign == 1):
                            delta -= 1  # becomes sat
                all_deltas.append((var_idx, delta))
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_var = var_idx

            # Simulated annealing acceptance
            # Find the minimum delta (best move)
            min_delta = min(d for _, d in all_deltas)

            # If the best move is non-worsening (delta <= 0), accept it
            # Otherwise, use SA to decide whether to accept a worsening move
            if min_delta <= 0:
                # Accept the best non-worsening move
                var_idx = best_var
            else:
                # All moves worsen; use SA probability
                # Accept with probability exp(-delta/T) where delta is the worsening amount
                # Use the best (least worsening) move as candidate
                delta_worst = min_delta  # this is positive (worsening)
                accept_prob = math.exp(-delta_worst / max(T, 1e-10))

                if rng.random() < accept_prob:
                    # Accept the least worsening move
                    var_idx = best_var
                else:
                    # Try a random move from the clause (more exploratory at high T)
                    # Or reject and try a different strategy
                    # At high temperatures, we're more likely to accept worsening moves
                    # At low temperatures, we stick to the best available
                    # For now, just pick the best (least bad) option
                    var_idx = best_var

            # Perform the flip
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val

            # Update clause satisfaction
            for ci in var_clauses[var_idx]:
                was_sat = clause_sat[ci]
                now_sat = False
                for v2, s2 in clause_list[ci]:
                    if assignment[v2] == (s2 == 1):
                        now_sat = True
                        break
                if was_sat and not now_sat:
                    unsat_count += 1
                elif not was_sat and now_sat:
                    unsat_count -= 1
                clause_sat[ci] = now_sat

        # After max_steps, if we found a better assignment, keep it
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
