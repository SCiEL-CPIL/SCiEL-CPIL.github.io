"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    # Deadline guard: 325s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 325.0

    # Pre-compute clause structure for fast updates
    # For each clause, store (var_idx, is_positive) pairs
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)

    # For each variable, store list of (clause_idx, is_positive)
    var_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for vi, pos in cv:
            var_clauses[vi].append((ci, pos))

    # State: assignment array, clause satisfaction tracking
    # For each clause, track:
    #   sat_count[ci] = number of literals satisfied in clause ci
    #   false_vars[ci] = set of var indices whose literal is False in clause ci

    def init_state(assignment):
        sat_count = [0] * m
        false_vars = [set() for _ in range(m)]
        for ci, cv in enumerate(clause_vars):
            for vi, pos in cv:
                if assignment[vi] == pos:
                    sat_count[ci] += 1
                else:
                    false_vars[ci].add(vi)
        return sat_count, false_vars

    def count_unsat(sat_count):
        cnt = 0
        for sc in sat_count:
            if sc == 0:
                cnt += 1
        return cnt

    def find_unsat_clause(sat_count):
        for ci in range(m):
            if sat_count[ci] == 0:
                return ci
        return -1

    def flip_variable(vi, assignment, sat_count, false_vars):
        """Flip variable vi and update state. Returns number of newly unsatisfied clauses."""
        old_val = assignment[vi]
        new_val = not old_val
        assignment[vi] = new_val

        delta = 0  # net change in unsat count
        for (ci, pos) in var_clauses[vi]:
            if old_val == pos:
                # Was satisfied, now unsatisfied
                sat_count[ci] -= 1
                false_vars[ci].add(vi)
                if sat_count[ci] == 0:
                    delta += 1
            else:
                # Was unsatisfied, now satisfied
                sat_count[ci] += 1
                false_vars[ci].discard(vi)
                if sat_count[ci] == 1:
                    delta -= 1
        return delta

    def belief_propagation(assignment, max_iters=50, tol=1e-4):
        """
        Run Sum-Product Algorithm (Belief Propagation) on the 3-SAT factor graph
        to estimate variable marginals. Returns an assignment based on beliefs.
        
        The factor graph has:
        - Variable nodes: x_0, ..., x_{n-1}
        - Factor nodes: C_0, ..., C_{m-1} (one per clause)
        
        Messages:
        - From variable v to factor c: m_{v->c}(x_v) = [P(x_v=True|neighbors), P(x_v=False|neighbors)]
        - From factor c to variable v: m_{c->v}(x_v) = P(clause c satisfied | x_v, other vars in c)
        
        For 3-SAT with OR clauses:
        - A clause is satisfied if at least one literal is True
        - m_{c->v}(x_v=True) = 1 - product of P(lit_i=False) for other literals i in c
        - m_{c->v}(x_v=False) = product of P(lit_i=True) for other literals i in c
        """
        # Initialize messages
        # var_to_factor[v][c] = [msg_true, msg_false] from variable v to factor c
        # factor_to_var[c][v] = [msg_true, msg_false] from factor c to variable v
        
        # Only store messages for edges that exist
        var_to_factor = {}
        factor_to_var = {}
        
        for v in range(n):
            var_to_factor[v] = {}
            for (c, pos) in var_clauses[v]:
                var_to_factor[v][c] = [1.0, 1.0]
        
        for c in range(m):
            factor_to_var[c] = {}
            for (v, pos) in clause_vars[c]:
                factor_to_var[c][v] = [1.0, 1.0]
        
        # Initialize with uniform beliefs
        for v in range(n):
            for c in var_to_factor[v]:
                var_to_factor[v][c] = [1.0, 1.0]
        
        for c in range(m):
            for v in factor_to_var[c]:
                factor_to_var[c][v] = [1.0, 1.0]
        
        # Run BP iterations
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            max_change = 0.0
            
            # Update messages from factors to variables
            for c in range(m):
                cv = clause_vars[c]
                for vi, pos in cv:
                    # Compute message from factor c to variable vi
                    # For an OR clause: P(clause sat | x_vi) = 1 - P(all other literals false)
                    # if pos: literal is x_vi, so x_vi=True satisfies
                    # if not pos: literal is not x_vi, so x_vi=False satisfies
                    
                    # Message for x_vi=True:
                    # Clause is satisfied if x_vi=True (when pos) OR any other literal is true
                    # P(clause sat | x_vi=True) = 1 - P(all other lits false)
                    # P(other lit i false) depends on sign
                    p_all_other_false = 1.0
                    for vj, posj in cv:
                        if vj == vi:
                            continue
                        # Get belief about x_vj from all other factors
                        # Use incoming messages to variable vj
                        # m_{vj->c}(x_vj) represents P(x_vj | all other factors)
                        # For simplicity, use the product of incoming messages
                        # m_{vj->c}(x_vj=True) * m_{vj->c}(x_vj=False) normalized
                        incoming_true = 1.0
                        incoming_false = 1.0
                        for c2 in var_to_factor[vj]:
                            if c2 == c:
                                continue
                            incoming_true *= var_to_factor[vj][c2][0]
                            incoming_false *= var_to_factor[vj][c2][1]
                        
                        # Normalize
                        total = incoming_true + incoming_false
                        if total < 1e-10:
                            p_xj_true = 0.5
                        else:
                            p_xj_true = incoming_true / total
                        
                        # P(lit_j is false)
                        if posj:
                            p_lit_false = 1.0 - p_xj_true
                        else:
                            p_lit_false = p_xj_true
                        
                        p_all_other_false *= p_lit_false
                    
                    msg_true = 1.0 - p_all_other_false
                    
                    # Message for x_vi=False:
                    # Clause is satisfied if x_vi=False (when not pos) OR any other literal is true
                    # P(clause sat | x_vi=False) = P(any other lit true) = 1 - P(all other lits false)
                    # Wait, this is the same calculation! Let me reconsider.
                    
                    # Actually for factor-to-variable messages in BP for CSP/SAT:
                    # m_{c->v}(x_v) = sum_{x_{N(c)\v}} [f_c(x_c) * prod_{u in N(c)\v} m_{u->c}(x_u)]
                    # where f_c is the clause indicator (1 if satisfied, 0 otherwise)
                    
                    # For OR clause: f_c = 1 if any literal true, 0 otherwise
                    # m_{c->v}(x_v=True) = sum over other assignments where clause is sat given x_v=True
                    # = 1 - P(all other lits false | x_v=True)
                    # = 1 - prod_{u in N(c)\v} P(lit_u=false)
                    
                    # m_{c->v}(x_v=False) = sum over other assignments where clause is sat given x_v=False
                    # = P(at least one other lit true)
                    # = 1 - P(all other lits false)
                    # Same as above!
                    
                    # Hmm, that means the message is the same regardless of x_v value?
                    # That can't be right. Let me reconsider.
                    
                    # Actually, the clause satisfaction depends on x_v:
                    # If pos (literal is x_v):
                    #   x_v=True => clause satisfied (regardless of others)
                    #   x_v=False => clause satisfied iff at least one other lit is true
                    # If not pos (literal is not x_v):
                    #   x_v=True => clause satisfied iff at least one other lit is true
                    #   x_v=False => clause satisfied (regardless of others)
                    
                    # So:
                    if pos:
                        # x_v=True: clause definitely satisfied
                        msg_true = 1.0
                        # x_v=False: clause satisfied iff at least one other lit true
                        # P(at least one other lit true) = 1 - P(all other lits false)
                        msg_false = 1.0 - p_all_other_false
                    else:
                        # x_v=False: clause definitely satisfied
                        msg_false = 1.0
                        # x_v=True: clause satisfied iff at least one other lit true
                        msg_true = 1.0 - p_all_other_false
                    
                    # Update
                    old_msg = factor_to_var[c][vi]
                    max_change = max(max_change, abs(msg_true - old_msg[0]), abs(msg_false - old_msg[1]))
                    factor_to_var[c][vi] = [msg_true, msg_false]
            
            # Update messages from variables to factors
            for v in range(n):
                for (c, pos) in var_clauses[v]:
                    # m_{v->c}(x_v) = prod_{c' in N(v)\c} m_{c'->v}(x_v)
                    # This is the belief about x_v from all other factors
                    msg_true = 1.0
                    msg_false = 1.0
                    for c2 in var_clauses[v]:
                        if c2[0] == c:
                            continue
                        msg_true *= factor_to_var[c2[0]][v][0]
                        msg_false *= factor_to_var[c2[0]][v][1]
                    
                    # Normalize
                    total = msg_true + msg_false
                    if total < 1e-10:
                        msg_true = 0.5
                        msg_false = 0.5
                    else:
                        msg_true /= total
                        msg_false /= total
                    
                    old_msg = var_to_factor[v][c]
                    max_change = max(max_change, abs(msg_true - old_msg[0]), abs(msg_false - old_msg[1]))
                    var_to_factor[v][c] = [msg_true, msg_false]
            
            if max_change < tol:
                break
        
        # Compute marginal beliefs for each variable
        # P(x_v=True) proportional to prod_{c in N(v)} m_{c->v}(x_v=True)
        # P(x_v=False) proportional to prod_{c in N(v)} m_{c->v}(x_v=False)
        
        bp_assignment = []
        for v in range(n):
            p_true = 1.0
            p_false = 1.0
            for (c, pos) in var_clauses[v]:
                p_true *= factor_to_var[c][v][0]
                p_false *= factor_to_var[c][v][1]
            
            total = p_true + p_false
            if total < 1e-10:
                bp_assignment.append(assignment[v])  # Keep original
            else:
                bp_assignment.append(p_true / total > 0.5)
        
        return bp_assignment

    # WalkSAT with multiple restarts and adaptive strategy
    rng = random.Random(0)

    # Weight parameter: probability of flipping a literal in the chosen unsat clause
    # vs. flipping a random variable that appears in unsat clauses
    w = 0.4
    w_min = 0.0
    w_max = 1.0
    w_decay = 0.98

    best_unsat = m + 1
    best_assignment = [False] * n

    max_restarts = 200
    base_steps = 5000

    # First, try BP to get a good initial assignment
    initial_assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Run BP to refine initial assignment
    if time.time() < deadline - 10.0:
        bp_assignment = belief_propagation(initial_assignment, max_iters=30, tol=1e-3)
        sat_count, false_vars = init_state(bp_assignment)
        cur_unsat = count_unsat(sat_count)
        
        if cur_unsat == 0:
            return bp_assignment
        
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = bp_assignment[:]
        
        # Use BP result as starting point for first WalkSAT restart
        initial_assignment = bp_assignment[:]

    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break

        # Initialize: first restart uses BP result, others random
        if restart == 0:
            assignment = initial_assignment[:]
        else:
            # Mix: 50% chance to use BP-guided init, 50% random
            if rng.random() < 0.3 and time.time() < deadline - 15.0:
                # Run a quick BP from random start
                rand_init = [rng.random() < 0.5 for _ in range(n)]
                bp_init = belief_propagation(rand_init, max_iters=15, tol=1e-3)
                assignment = bp_init
            else:
                assignment = [rng.random() < 0.5 for _ in range(n)]
        
        sat_count, false_vars = init_state(assignment)
        cur_unsat = count_unsat(sat_count)

        if cur_unsat == 0:
            return assignment

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]

        # Adaptive step budget: more steps if we're making progress
        # Scale steps based on restart number and current difficulty
        steps = base_steps + restart * 2000

        for step in range(steps):
            if time.time() > deadline - 2.0:
                break

            if cur_unsat == 0:
                return assignment

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

            # Find an unsatisfied clause
            ci = find_unsat_clause(sat_count)
            if ci == -1:
                return assignment

            # Choose which variable to flip
            if rng.random() < w:
                # Flip a random literal in the chosen unsat clause
                cv = clause_vars[ci]
                idx = rng.randrange(len(cv))
                vi = cv[idx][0]
            else:
                # Flip a variable that appears in some unsat clause
                # Weight by how many unsat clauses it appears in
                # Pick from false_vars of the chosen clause, or a random unsat clause
                # Use the chosen clause's false variables
                fv = false_vars[ci]
                if fv:
                    vi = rng.choice(list(fv))
                else:
                    cv = clause_vars[ci]
                    idx = rng.randrange(len(cv))
                    vi = cv[idx][0]

            # Try the flip
            old_cur_unsat = cur_unsat
            new_delta = flip_variable(vi, assignment, sat_count, false_vars)
            cur_unsat += new_delta

            # WalkSAT-style: if we increased unsat, consider reverting
            # but with some probability keep it to escape local minima
            if new_delta > 0:
                # With some probability, revert
                if rng.random() > 0.5:
                    flip_variable(vi, assignment, sat_count, false_vars)
                    cur_unsat = old_cur_unsat

        # Decay w to explore more broadly as we continue
        w = max(w_min, min(w_max, w * w_decay))

    # Return best assignment found
    return best_assignment
# EVOLVE-BLOCK-END
