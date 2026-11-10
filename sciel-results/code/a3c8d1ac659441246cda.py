"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys


def _dpll_fallback(assignment, n, clauses, clause_lits, clause_sat, max_depth=50):
    """Try to fix remaining unsat clauses using limited DPLL-style search."""
    unsat_clauses = [ci for ci in range(len(clauses)) if clause_sat[ci] == 0]
    if not unsat_clauses:
        return assignment[:]
    
    # Try to fix by flipping variables in unsat clauses
    # Use a simple backtracking with limited depth
    def backtrack(assn, depth, remaining_unsat):
        if depth > max_depth:
            return None
        if not remaining_unsat:
            return assn[:]
        
        # Pick an unsat clause
        ci = remaining_unsat[0]
        rest = remaining_unsat[1:]
        
        # Try flipping each literal in this clause
        for var, sign in clause_lits[ci]:
            new_assn = assn[:]
            new_assn[var] = (sign == 1)  # set to satisfy this literal
            
            # Recompute clause_sat for affected clauses
            new_clause_sat = clause_sat[:]
            # Update for variable var
            if new_assn[var]:
                for c2 in pos_clauses_global[var]:
                    new_clause_sat[c2] += 1
                for c2 in neg_clauses_global[var]:
                    new_clause_sat[c2] -= 1
            else:
                for c2 in neg_clauses_global[var]:
                    new_clause_sat[c2] += 1
                for c2 in pos_clauses_global[var]:
                    new_clause_sat[c2] -= 1
            
            # Find new unsat clauses
            new_unsat = [c for c in range(m_global) if new_clause_sat[c] == 0]
            
            if len(new_unsat) < len(remaining_unsat):
                result = backtrack(new_assn, depth + 1, new_unsat)
                if result is not None:
                    return result
        
        return None
    
    result = backtrack(assignment, 0, unsat_clauses)
    return result


# Global references for DPLL fallback
pos_clauses_global = None
neg_clauses_global = None
m_global = 0


def solve(instance):
    """Hybrid local search: clause-weighted WalkSAT + DPLL fallback with restarts."""
    global pos_clauses_global, neg_clauses_global, m_global
    
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    m_global = m
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
    
    pos_clauses_global = pos_clauses
    neg_clauses_global = neg_clauses

    # Variable degrees for weighted sampling
    var_degree = [len(pos_clauses[v]) + len(neg_clauses[v]) for v in range(n)]
    total_degree = sum(var_degree)
    
    # Build cumulative distribution for weighted variable selection
    cum_degree = [0] * n
    for v in range(n):
        cum_degree[v] = (cum_degree[v-1] if v > 0 else 0) + var_degree[v]

    rng = random.Random(42)
    best_unsat = float('inf')
    best_assignment = None

    # Multiple restarts with different strategies
    num_restarts = 8
    restart_time = (deadline - time.time()) / num_restarts

    for restart in range(num_restarts):
        if time.time() >= deadline:
            break
        
        # Initialize assignment - vary initialization strategy per restart
        if restart == 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        elif restart == 1:
            # All False
            assignment = [False] * n
        elif restart == 2:
            # All True
            assignment = [True] * n
        else:
            # Random with slight bias
            bias = 0.3 + 0.4 * rng.random()
            assignment = [rng.random() < bias for _ in range(n)]

        # Compute initial clause satisfaction status
        clause_sat = [0] * m
        for ci, lits in enumerate(clause_lits):
            sat_count = 0
            for var, sign in lits:
                if assignment[var] == (sign == 1):
                    sat_count += 1
            clause_sat[ci] = sat_count

        unsat_count = sum(1 for s in clause_sat if s == 0)

        # If already satisfied, return immediately
        if unsat_count == 0:
            return assignment

        # Try DPLL fallback if close to solution
        if unsat_count <= 5:
            result = _dpll_fallback(assignment, n, clauses, clause_lits, clause_sat)
            if result is not None:
                return result

        # Track best for this restart
        cur_best_unsat = unsat_count
        cur_best_assignment = assignment[:]

        # Clause weights for WalkSAT
        clause_weights = [1.0] * m
        weight_decay = 0.995
        weight_increase = 1.5

        # Local search parameters
        flip_limit = 10000
        max_flips_per_restart = min(flip_limit, int(restart_time * 10000))
        steps = 0
        
        while steps < max_flips_per_restart:
            if time.time() >= deadline - 5:
                break
            
            # Check if solved
            if unsat_count == 0:
                return assignment
            
            # Try DPLL fallback periodically when close
            if unsat_count <= 3 and steps % 100 == 0:
                result = _dpll_fallback(assignment, n, clauses, clause_lits, clause_sat)
                if result is not None:
                    return result
            
            # Select variable to flip
            # Weighted selection: prefer variables in unsat clauses
            if unsat_count > 0 and rng.random() < 0.7:
                # Pick a random unsat clause, then a variable in it
                unsat_cis = [ci for ci in range(m) if clause_sat[ci] == 0]
                if unsat_cis:
                    ci = unsat_cis[rng.randint(0, len(unsat_cis) - 1)]
                    # Pick a variable in this clause, weighted by clause weight
                    lits_in_clause = clause_lits[ci]
                    # Weighted choice based on clause weight
                    v = lits_in_clause[rng.randint(0, len(lits_in_clause) - 1)][0]
                else:
                    # Fallback to weighted random variable
                    r = rng.random() * total_degree
                    v = 0
                    for i in range(n):
                        if cum_degree[i] >= r:
                            v = i
                            break
            else:
                # Weighted random variable selection
                r = rng.random() * total_degree
                v = 0
                for i in range(n):
                    if cum_degree[i] >= r:
                        v = i
                        break

            # Compute delta: change in unsat_count if we flip v
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

            # Accept flip (WalkSAT always accepts, but we can be selective)
            # Use a simple heuristic: accept if delta <= 0, or with probability based on delta
            if delta <= 0 or rng.random() < max(0.1, 0.9 * math.exp(-delta / 3.0)):
                # Apply flip
                assignment[v] = not assignment[v]
                
                # Update clause_sat
                if assignment[v]:
                    for ci in pos_clauses[v]:
                        clause_sat[ci] += 1
                        # Increase weight if this clause became satisfied
                        if clause_sat[ci] == 1:
                            clause_weights[ci] *= weight_increase
                    for ci in neg_clauses[v]:
                        clause_sat[ci] -= 1
                        # Decrease weight if this clause became unsat
                        if clause_sat[ci] == 0:
                            clause_weights[ci] *= weight_increase
                else:
                    for ci in neg_clauses[v]:
                        clause_sat[ci] += 1
                        if clause_sat[ci] == 1:
                            clause_weights[ci] *= weight_increase
                    for ci in pos_clauses[v]:
                        clause_sat[ci] -= 1
                        if clause_sat[ci] == 0:
                            clause_weights[ci] *= weight_increase

                # Update unsat_count
                unsat_count += delta

                # Decay weights slightly
                # (skip for speed; do periodic decay)
                if steps % 1000 == 0:
                    for i in range(m):
                        clause_weights[i] *= weight_decay
                        if clause_weights[i] < 0.1:
                            clause_weights[i] = 0.1

                # Track best
                if unsat_count < cur_best_unsat:
                    cur_best_unsat = unsat_count
                    cur_best_assignment = assignment[:]
            
            steps += 1

        # After local search, try DPLL if close
        if cur_best_unsat <= 5:
            result = _dpll_fallback(cur_best_assignment, n, clauses, clause_lits, clause_sat)
            if result is not None:
                return result

        # Update global best
        if cur_best_unsat < best_unsat:
            best_unsat = cur_best_unsat
            best_assignment = cur_best_assignment[:]

    # Return best assignment found
    if best_unsat == 0 and best_assignment is not None:
        return best_assignment
    if best_assignment is not None:
        return best_assignment
    # Fallback: return random assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
