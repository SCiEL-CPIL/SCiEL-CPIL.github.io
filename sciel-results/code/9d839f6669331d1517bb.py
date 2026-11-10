"""BP-guided WalkSAT with min-conflicts, restarts, and deadline guard for planted 3-SAT."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 330s kill

    # Pre-extract clause literals as (var_idx, sign) for speed
    # var_idx is 0-based, sign is +1 for positive, -1 for negative
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])

    num_clauses = len(clause_data)

    # Build factor graph adjacency: var_to_clauses and clause_to_vars
    var_to_clauses = [[] for _ in range(n)]
    clause_to_vars = []
    for ci, clause in enumerate(clause_data):
        vars_in_clause = []
        for var_idx, sign in clause:
            var_to_clauses[var_idx].append(ci)
            vars_in_clause.append((var_idx, sign))
        clause_to_vars.append(vars_in_clause)

    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        count = 0
        for clause in clause_data:
            sat = False
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def is_satisfied(assignment):
        """Check if all clauses are satisfied."""
        for clause in clause_data:
            sat = False
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def belief_propagation(max_iters=100, damping=0.8):
        """
        Run Belief Propagation on the factor graph.
        Returns log-likelihood ratio (LLR) for each variable.
        Positive LLR means P(var=True) > P(var=False).
        """
        # Initialize messages: var->clause and clause->var
        # msg_vc[var][clause] = LLR sent from var to clause
        # msg_cv[clause][var] = LLR sent from clause to var
        msg_vc = [[0.0] * num_clauses for _ in range(n)]
        msg_cv = [[0.0] * 3 for _ in range(num_clauses)]  # each clause has 3 vars

        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break

            # Update clause->var messages
            for ci in range(num_clauses):
                clause_vars = clause_to_vars[ci]
                for vi, (var_idx, sign) in enumerate(clause_vars):
                    # Compute message from clause to var
                    # For a 3-SAT clause, the clause is unsatisfied only if all literals are false
                    # The message tells the var how much the clause "wants" it to be true/false
                    # LLR = log(P(clause_satisfied | var=True) / P(clause_satisfied | var=False))
                    
                    # If var is set to satisfy its literal, clause is satisfied regardless of others
                    # If var is set to not satisfy, clause depends on other two vars
                    
                    # Simplified: count how many other vars in clause are currently believed to satisfy
                    other_msgs = []
                    for vj, (other_var, other_sign) in enumerate(clause_vars):
                        if vj != vi:
                            # Get message from that var to this clause
                            m = msg_vc[other_var][ci]
                            # Convert LLR to probability of satisfying the literal
                            # P(lit_sat) = 1/(1+exp(-m*sign)) if m is LLR for var
                            # Actually m is already in the right frame for this clause
                            # P(other_var satisfies its literal) = 1/(1+exp(-m))
                            p_sat = 1.0 / (1.0 + math.exp(-m)) if abs(m) < 50 else (1.0 if m > 0 else 0.0)
                            other_msgs.append(p_sat)
                    
                    # Probability clause is satisfied given var's setting
                    if len(other_msgs) == 2:
                        p_clause_sat_if_var_satisfies = 1.0  # var satisfies, clause is sat
                        # If var doesn't satisfy, clause is sat if at least one other satisfies
                        p_clause_sat_if_var_not = 1.0 - (1.0 - other_msgs[0]) * (1.0 - other_msgs[1])
                    else:
                        p_clause_sat_if_var_satisfies = 1.0
                        p_clause_sat_if_var_not = 1.0
                    
                    # LLR = log(P_sat_if_true / P_sat_if_false)
                    # Need to account for sign
                    if sign == 1:
                        # var=True satisfies, var=False doesn't
                        p_true = p_clause_sat_if_var_satisfies
                        p_false = p_clause_sat_if_var_not
                    else:
                        # var=False satisfies, var=True doesn't
                        p_true = p_clause_sat_if_var_not
                        p_false = p_clause_sat_if_var_satisfies
                    
                    if p_true > 0 and p_false > 0:
                        new_msg = math.log(p_true / p_false)
                    elif p_true > 0:
                        new_msg = 50.0
                    elif p_false > 0:
                        new_msg = -50.0
                    else:
                        new_msg = 0.0
                    
                    # Damping
                    msg_cv[ci][vi] = damping * msg_cv[ci][vi] + (1 - damping) * new_msg

            # Update var->clause messages
            for vi in range(n):
                for ci in var_to_clauses[vi]:
                    # Find position of this var in the clause
                    clause_vars = clause_to_vars[ci]
                    pos = -1
                    for vj, (var_idx, sign) in enumerate(clause_vars):
                        if var_idx == vi:
                            pos = vj
                            break
                    if pos == -1:
                        continue
                    
                    # Sum of all other clause->var messages for this variable
                    # The var->clause message is the product (in log domain, sum) of all incoming messages
                    # from other clauses, plus the prior
                    total = 0.0
                    for other_ci in var_to_clauses[vi]:
                        if other_ci != ci:
                            # Find position in other clause
                            other_clause_vars = clause_to_vars[other_ci]
                            for vj, (var_idx2, sign2) in enumerate(other_clause_vars):
                                if var_idx2 == vi:
                                    total += msg_cv[other_ci][vj]
                                    break
                    # Add prior (uniform, so 0)
                    msg_vc[vi][ci] = damping * msg_vc[vi][ci] + (1 - damping) * total

        # Compute final beliefs: LLR for each variable
        llr = [0.0] * n
        for vi in range(n):
            for ci in var_to_clauses[vi]:
                clause_vars = clause_to_vars[ci]
                for vj, (var_idx, sign) in enumerate(clause_vars):
                    if var_idx == vi:
                        llr[vi] += msg_cv[ci][vj]
                        break

        return llr

    def llr_to_assignment(llr, confidence_threshold=0.0):
        """Convert LLRs to an assignment. High confidence vars get fixed, others random."""
        assignment = []
        for i in range(n):
            if abs(llr[i]) > confidence_threshold:
                assignment.append(llr[i] > 0)
            else:
                assignment.append(random.random() < 0.5)
        return assignment

    def walk_one(assignment, rng, max_steps, p_flip):
        """Run WalkSAT from current assignment for up to max_steps with flip probability p_flip."""
        # Track unsatisfied clauses incrementally
        clause_sat = [False] * num_clauses
        for i, clause in enumerate(clause_data):
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    clause_sat[i] = True
                    break

        unsat_set = set()
        for i in range(num_clauses):
            if not clause_sat[i]:
                unsat_set.add(i)

        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False

            if not unsat_set:
                return assignment, True

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_set))
            clause = clause_data[ci]

            # Score each literal: count how many other unsatisfied clauses contain this variable
            var_unsat_count = {}
            for uc in unsat_set:
                for var_idx, sign in clause_data[uc]:
                    var_unsat_count[var_idx] = var_unsat_count.get(var_idx, 0) + 1

            # Weighted selection with noise
            weights = []
            for var_idx, sign in clause:
                base_w = var_unsat_count.get(var_idx, 0) + 1
                noise = rng.random() * p_flip * 5
                weights.append(base_w + noise)

            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            chosen_idx = 0
            for idx, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    chosen_idx = idx
                    break

            var_idx, sign = clause[chosen_idx]
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val

            # Update clause satisfaction for all clauses containing this variable
            for ci2 in var_to_clauses[var_idx]:
                clause2 = clause_data[ci2]
                was_sat = clause_sat[ci2]
                is_sat_now = False
                for v, s in clause2:
                    if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                        is_sat_now = True
                        break

                if was_sat and not is_sat_now:
                    unsat_set.add(ci2)
                elif not was_sat and is_sat_now:
                    unsat_set.discard(ci2)
                clause_sat[ci2] = is_sat_now

        return assignment, False

    # Main algorithm: BP to get initial beliefs, then guided local search
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    # Phase 1: Run BP to get variable beliefs
    llr = belief_propagation(max_iters=150, damping=0.7)

    # Create initial assignment from BP beliefs
    bp_assignment = llr_to_assignment(llr, confidence_threshold=0.5)

    # Phase 2: Try to solve starting from BP-guided assignment
    max_steps_per_restart = 30000
    num_restarts = 500  # max restarts, will stop at deadline

    # First try with BP-guided start
    assignment = bp_assignment[:]
    p_flip = 0.3
    assignment, solved = walk_one(assignment, rng, max_steps_per_restart, p_flip)
    if solved:
        return assignment

    uc = count_unsat(assignment)
    if uc < best_unsat_count:
        best_unsat_count = uc
        best_assignment = assignment[:]

    # Phase 3: Additional restarts with mix of BP-guided and random starts
    for restart in range(1, num_restarts):
        if time.time() > deadline:
            break

        # Alternate between BP-guided and random starts
        if restart % 3 == 0:
            # Use BP-guided start with some noise
            assignment = []
            for i in range(n):
                if abs(llr[i]) > 0.5 and rng.random() < 0.8:
                    assignment.append(llr[i] > 0)
                else:
                    assignment.append(rng.random() < 0.5)
        elif restart % 3 == 1:
            # Pure BP-guided
            assignment = llr_to_assignment(llr, confidence_threshold=0.3)
        else:
            # Random start
            assignment = [rng.random() < 0.5 for _ in range(n)]

        # Vary the flip probability
        p_flip = 0.2 + 0.3 * (restart % 5) / 4.0

        # Try to solve
        assignment, solved = walk_one(assignment, rng, max_steps_per_restart, p_flip)

        if solved:
            return assignment

        # Track best
        uc = count_unsat(assignment)
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: BP-guided assignment
    return bp_assignment

# EVOLVE-BLOCK-END
