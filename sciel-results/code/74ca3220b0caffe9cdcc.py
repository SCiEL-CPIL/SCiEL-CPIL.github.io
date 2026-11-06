"""Improved planted 3-SAT solver using bitmask-accelerated WalkSAT with greedy refinement."""
# EVOLVE-BLOCK-START
import random
import time
from math import log, exp

DEADLINE_FRACTION = 0.95  # safety margin: stop at 95% of 330s


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline guard
    t_start = time.time()
    deadline = t_start + 330 * DEADLINE_FRACTION

    # Precompute bitmasks: for each variable, which clauses contain its positive/negative literal
    # pos_mask[v]: bitmask of clauses where literal +v appears
    # neg_mask[v]: bitmask of clauses where literal -v appears
    # Also for each clause, store its literals for fast satisfaction checking
    # We'll use Python integers as arbitrary-precision bitmasks

    # Build occurrence maps
    pos_mask = [0] * n
    neg_mask = [0] * n
    # clause_literals: list of (var, is_positive) for each clause
    clause_vars = []  # clause_vars[c] = list of (var_idx, sign) where sign=True means positive
    # For each clause, also store the variable and sign for quick lookup
    clause_var_sign = []  # clause_var_sign[c] = list of (var_idx, sign) where sign=True means positive

    for ci, clause in enumerate(clauses):
        bit = 1 << ci
        for lit in clause:
            v = abs(lit) - 1
            if lit > 0:
                pos_mask[v] |= bit
            else:
                neg_mask[v] |= bit
        clause_vars.append([(abs(l) - 1, l > 0) for l in clause])
        clause_var_sign.append([(abs(l) - 1, l > 0) for l in clause])

    all_mask = (1 << m) - 1

    def satisfied_mask(assignment):
        sm = 0
        for v in range(n):
            if assignment[v]:
                sm |= pos_mask[v]
            else:
                sm |= neg_mask[v]
        return sm

    def is_satisfied(assignment):
        return satisfied_mask(assignment) == all_mask

    def popcount(x):
        return bin(x).count('1')

    # Optimized flip_gain: precompute satisfied mask once, then for each variable
    # compute gain using bitmask operations
    def compute_all_gains(assignment, cur_sat_mask):
        """Compute flip gain for all variables. Returns list of gains."""
        cur_unsat_mask = all_mask ^ cur_sat_mask
        gains = [0] * n
        for v in range(n):
            current_val = assignment[v]
            if current_val:
                # Currently satisfying via pos_mask[v], after flip via neg_mask[v]
                # Newly satisfied: clauses in neg_mask[v] that are currently unsat
                newly_sat = neg_mask[v] & cur_unsat_mask
                # Newly unsatisfied: clauses in pos_mask[v] that were satisfied ONLY by v
                # A clause c in pos_mask[v] is satisfied only by v if:
                #   c is in cur_sat_mask AND c is NOT in the satisfied mask from other variables
                # This is complex. Let's use a different approach:
                # gain = popcount(newly_sat) - popcount(clauses that become unsat)
                # A clause becomes unsat if it was satisfied only by v's positive literal
                # i.e., c in pos_mask[v] AND c not satisfied by any other variable
                # We can compute: for each clause in pos_mask[v], check if it's satisfied by others
                # But that's expensive. Alternative:
                # newly_unsat = pos_mask[v] & cur_sat_mask & ~(satisfied by others)
                # satisfied by others = OR of pos_mask[u] or neg_mask[u] for u != v
                # This is O(n) per variable, total O(n^2) which is fine for n=50
                # But let's try a faster approach:
                # For each clause in pos_mask[v] that is currently satisfied,
                # check if it's satisfied by at least one other variable
                newly_unsat_count = 0
                pm = pos_mask[v]
                while pm:
                    lsb = pm & (-pm)
                    ci = lsb.bit_length() - 1
                    pm ^= lsb
                    if not (cur_sat_mask & lsb):
                        continue  # clause not satisfied, can't become unsat
                    # Check if satisfied by another variable
                    cvars = clause_var_sign[ci]
                    satisfied_by_other = False
                    for vv, pos in cvars:
                        if vv == v:
                            continue
                        if assignment[vv] == pos:
                            satisfied_by_other = True
                            break
                    if not satisfied_by_other:
                        newly_unsat_count += 1
                gain = popcount(newly_sat) - newly_unsat_count
            else:
                # Currently satisfying via neg_mask[v], after flip via pos_mask[v]
                newly_sat = pos_mask[v] & cur_unsat_mask
                newly_unsat_count = 0
                nm = neg_mask[v]
                while nm:
                    lsb = nm & (-nm)
                    ci = lsb.bit_length() - 1
                    nm ^= lsb
                    if not (cur_sat_mask & lsb):
                        continue
                    cvars = clause_var_sign[ci]
                    satisfied_by_other = False
                    for vv, pos in cvars:
                        if vv == v:
                            continue
                        if assignment[vv] == pos:
                            satisfied_by_other = True
                            break
                    if not satisfied_by_other:
                        newly_unsat_count += 1
                gain = popcount(newly_sat) - newly_unsat_count
            gains[v] = gain
        return gains

    # Initialize assignment with bias toward satisfying more clauses
    rng = random.Random(42)
    
    # Try a few initializations and pick the best
    best_init = None
    best_init_unsat = m + 1
    for _ in range(20):
        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat = m - popcount(satisfied_mask(assignment))
        if unsat < best_init_unsat:
            best_init_unsat = unsat
            best_init = assignment[:]
    
    assignment = best_init
    best_assignment = assignment[:]
    best_unsat_count = best_init_unsat

    # WalkSAT with greedy refinement
    # Parameters tuned for planted 3-SAT with 50 variables
    walksat_restarts = 200
    max_steps_per_restart = 5000
    noise_param = 0.25  # probability of random flip vs greedy

    for restart in range(walksat_restarts):
        if time.time() > deadline:
            break

        # Random restart assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        cur_sat_mask = satisfied_mask(assignment)
        cur_unsat = m - popcount(cur_sat_mask)

        if cur_unsat == 0:
            return assignment

        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]

        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break

            if cur_unsat == 0:
                return assignment

            if cur_unsat < best_unsat_count:
                best_unsat_count = cur_unsat
                best_assignment = assignment[:]

            # Choose a variable to flip
            if rng.random() < noise_param:
                # Random flip
                v = rng.randint(0, n - 1)
            else:
                # Greedy: find flip with max gain
                gains = compute_all_gains(assignment, cur_sat_mask)
                best_gain = -1
                best_v = -1
                for v in range(n):
                    g = gains[v]
                    if g > best_gain or (g == best_gain and rng.random() < 0.5):
                        best_gain = g
                        best_v = v
                v = best_v

            # Flip variable v
            assignment[v] = not assignment[v]
            
            # Update cur_sat_mask incrementally
            # Before flip: v contributed pos_mask[v] if assignment[v] was True, else neg_mask[v]
            # After flip: v contributes the opposite
            # But we need to recompute because other variables' contributions don't change,
            # but the union might change
            # Actually, we can recompute cur_sat_mask from scratch for simplicity
            # Or update: remove v's old contribution, add new
            # But since it's a union, we can't simply subtract
            # Let's just recompute
            cur_sat_mask = satisfied_mask(assignment)
            cur_unsat = m - popcount(cur_sat_mask)

    # Final polish: try local search from best_assignment
    if time.time() < deadline:
        assignment = best_assignment[:]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur_sat_mask = satisfied_mask(assignment)
            cur_unsat = m - popcount(cur_sat_mask)
            if cur_unsat == 0:
                return assignment
            gains = compute_all_gains(assignment, cur_sat_mask)
            for v in range(n):
                g = gains[v]
                if g > 0:
                    assignment[v] = not assignment[v]
                    improved = True
                    break

    return best_assignment
# EVOLVE-BLOCK-END
