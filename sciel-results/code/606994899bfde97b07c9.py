"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n = instance["n"]
    clauses_raw = instance["clauses"]
    
    # Convert to 0-based: store literal as signed int (positive = var+1, negative = -(var+1))
    clause_literals = []  # list of lists of (var_idx, is_pos)
    for c in clauses_raw:
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
        clause_literals.append(cv)
        
    m = len(clause_literals)
    
    # Build variable to clause index with polarity
    var_to_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_literals):
        for var_idx, is_pos in cv:
            var_to_clauses[var_idx].append((i, is_pos))
    
    # Precompute: for each clause i and each variable v in clause i,
    # store the list of (other_var, other_is_pos) pairs so we can quickly
    # check if other literals are satisfied.
    # clause_other_satisfied[i][j] = list of (var_idx, is_pos) for literals in clause i
    #   except the one at position j (i.e., the "other" literals)
    # Actually, for break count we need: for clause c and variable v in c,
    #   is_any_other_literal_satisfied(c, v) = any(assignment[w] == p for (w,p) in c if w != v)
    # We can precompute for each clause a list of its literals, and at query time
    # check the other 2 literals (since it's 3-SAT).
    
    rng = random.Random(42)
    deadline = time.monotonic() + 290.0
    
    def init_clause_counts(assignment):
        counts = [0] * m
        for i, cv in enumerate(clause_literals):
            cnt = 0
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            counts[i] = cnt
        return counts
    
    def get_unsat_from_counts(counts):
        return [i for i in range(m) if counts[i] == 0]
    
    # Belief Propagation to get informed initial assignments
    def belief_propagation(num_iterations=50):
        """Run BP on the factor graph to get variable biases."""
        # Messages: var_to_clause[var][clause] = log-odds message from var to clause
        #           clause_to_var[clause][var] = log-odds message from clause to var
        # For 3-SAT with OR clauses:
        #   Clause message to var v: P(clause sat | v=0) - P(clause sat | v=1) in log-odds
        #   For OR clause with literals: P(clause sat | v=false) = P(at least one other lit true)
        #   P(clause sat | v=true) = 1 (if v's literal is positive and v=true)
        
        # Initialize messages to 0 (uniform prior)
        var_to_clause = [[0.0] * len(var_to_clauses[v]) for v in range(n)]
        clause_to_var = [[0.0] * len(clause_literals[c]) for c in range(m)]
        
        # Map var_to_clauses indices to actual clause indices
        var_clause_map = []
        for v in range(n):
            var_clause_map.append([c_idx for c_idx, _ in var_to_clauses[v]])
        
        # Map clause_literals indices to actual var indices
        clause_var_map = []
        for c in range(m):
            clause_var_map.append([var_idx for var_idx, _ in clause_literals[c]])
        
        for it in range(num_iterations):
            # Update clause-to-var messages
            for c in range(m):
                cv = clause_literals[c]
                for j, (var_idx, is_pos) in enumerate(cv):
                    # Compute P(clause satisfied | var_idx = not is_pos) 
                    # i.e., probability that at least one OTHER literal is satisfied
                    # Using incoming messages from other variables
                    log_prob_other_true = []
                    for k, (v2, p2) in enumerate(cv):
                        if k == j:
                            continue
                        # Message from var v2 to clause c
                        msg = var_to_clause[v2][var_to_clauses[v2].index((c, p2))] if False else 0.0
                        # We need to look up the message properly
                        # var_to_clauses[v2] is list of (clause_idx, is_pos)
                        # Find index of (c, p2) in var_to_clauses[v2]
                        idx_in_v2 = -1
                        for ii, (ci, pi) in enumerate(var_to_clauses[v2]):
                            if ci == c and pi == p2:
                                idx_in_v2 = ii
                                break
                        if idx_in_v2 >= 0:
                            msg = var_to_clause[v2][idx_in_v2]
                        else:
                            msg = 0.0
                        # msg is log-odds that v2 takes value p2
                        # P(v2 = p2) = sigmoid(msg)
                        # P(v2 != p2) = 1 - sigmoid(msg)
                        # For the clause to be satisfied by v2, we need v2 = p2
                        # log P(v2 = p2) = log(sigmoid(msg)) = -log(1 + exp(-msg))
                        # Use stable computation
                        if msg > 0:
                            log_p_true = -math.log1p(math.exp(-msg))
                        else:
                            log_p_true = msg - math.log1p(math.exp(msg))
                        log_prob_other_true.append(log_p_true)
                    
                    # P(at least one other lit true) = 1 - prod(1 - P(other_i true))
                    # log(1 - prod(1 - p_i)) 
                    # Use: log(1 - exp(sum(log(1-p_i))))
                    # log(1-p_i) = log(1 - sigmoid(msg_i))
                    log_prod_complement = 0.0
                    for log_p_true in log_prob_other_true:
                        # log(1 - p) where p = sigmoid(msg)
                        # = -log(1 + exp(msg)) if we use the complement
                        # Actually: 1 - sigmoid(x) = sigmoid(-x)
                        # log(sigmoid(-x)) = -log(1 + exp(x))
                        # But we have log_p_true = log(sigmoid(msg_i))
                        # So log(1 - p_i) = log(sigmoid(-msg_i))
                        # We need msg_i, not log_p_true
                        pass
                    
                    # Simpler approach: work with probabilities directly for small sizes
                    # P(clause sat | v_j = false) = 1 - prod over other i of P(v_i != p_i)
                    # P(v_i != p_i) = 1 - sigmoid(msg_i)
                    prod_complement = 1.0
                    for k, (v2, p2) in enumerate(cv):
                        if k == j:
                            continue
                        idx_in_v2 = -1
                        for ii, (ci, pi) in enumerate(var_to_clauses[v2]):
                            if ci == c and pi == p2:
                                idx_in_v2 = ii
                                break
                        if idx_in_v2 >= 0:
                            msg = var_to_clause[v2][idx_in_v2]
                        else:
                            msg = 0.0
                        # P(v2 != p2) = 1 - sigmoid(msg) = sigmoid(-msg)
                        if msg > 20:
                            p_complement = 0.0
                        elif msg < -20:
                            p_complement = 1.0
                        else:
                            p_complement = 1.0 / (1.0 + math.exp(msg))
                        prod_complement *= p_complement
                    
                    p_clause_sat_given_false = 1.0 - prod_complement
                    # P(clause sat | v_j = true) = 1.0 (the literal is satisfied)
                    # Message: log(P(clause sat | v=1) / P(clause sat | v=0))
                    # But we need to be careful about polarity
                    # If is_pos, then v=true satisfies the literal
                    # If not is_pos, then v=false satisfies the literal
                    # The message from clause to var should encourage the value that satisfies
                    # msg = log(P(clause sat | v satisfies) / P(clause sat | v doesn't satisfy))
                    # = log(1.0 / p_clause_sat_given_false)
                    # Wait, this isn't quite right for BP. Let me use a simpler formulation.
                    
                    # Standard BP for CSP/SAT:
                    # Clause-to-var message: m_{c->v}(v) = -log(1 - prod_{u in N(c)\v} (1 - m_{u->c}(u)))
                    # where m_{u->c}(u) is the message from var u to clause c, interpreted as
                    # the "confidence" that u takes the value that satisfies its literal in c.
                    
                    # Let's redefine: var_to_clause[v][idx] = confidence that v takes the 
                    # value that satisfies its literal in the corresponding clause.
                    # This is sigmoid of the log-odds.
                    
                    # For now, use a simplified approach:
                    # msg_{c->v_j} = log(1 / (1 - p_clause_sat_given_false)) if is_pos
                    #              = log(1 / p_clause_sat_given_false) if not is_pos... 
                    # This is getting complicated. Let me use a different, simpler method.
                    
                    clause_to_var[c][j] = 0.0  # placeholder
                    
            # This BP is getting too complex for the time budget. 
            # Use a simpler approach: just count constraint satisfaction bias.
            break
        
        # Fallback: simple heuristic - for each variable, count how many clauses
        # would be satisfied if the variable takes each value, given current random assignment
        # Actually, let's just return None and use random init with a twist
        return None
    
    # Optimized WalkSAT with precomputed break counts
    def optimized_walksat(assignment, max_steps, sa_temp=1.0):
        counts = init_clause_counts(assignment)
        unsat_set = set(get_unsat_from_counts(counts))
        
        # Precompute for each clause and each position in it, the other two literal positions
        # For 3-SAT, clause has exactly 3 literals
        # clause_other[i] = [(j1, j2)] meaning for literal at position 0, others are at 1,2
        # We'll just access clause_literals[c][k] for k != j
        
        for step in range(max_steps):
            if time.monotonic() > deadline - 2.0:
                return False, assignment, unsat_set, counts
                
            if not unsat_set:
                return True, assignment, unsat_set, counts
            
            # Select unsatisfied clause
            c_idx = rng.choice(list(unsat_set))
            cv = clause_literals[c_idx]
            
            # Compute break count for each variable in the clause
            best_var = None
            best_score = float('inf')
            
            for j, (var_idx, is_pos) in enumerate(cv):
                # Break count: number of satisfied clauses containing var_idx
                # that would become unsatisfied if we flip var_idx
                break_count = 0
                for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                    if counts[c_aff] == 0:
                        continue  # already unsat, flipping can't make it worse
                    cv_aff = clause_literals[c_aff]
                    # Check if any OTHER literal in c_aff is satisfied
                    # For 3-SAT, check the other 2 literals
                    still_sat = False
                    for k2, (v2, p2) in enumerate(cv_aff):
                        if v2 == var_idx:
                            continue
                        if assignment[v2] == p2:
                            still_sat = True
                            break
                    if not still_sat:
                        break_count += 1
                
                # WalkSAT: minimize break count, random tiebreak
                score = break_count + rng.random() * 0.5
                
                if score < best_score:
                    best_score = score
                    best_var = var_idx
            
            if best_var is None:
                best_var = cv[0][0]
            
            var_idx = best_var
            
            old_unsat = len(unsat_set)
            
            # Flip variable
            assignment[var_idx] = not assignment[var_idx]
            
            # Update counts incrementally
            for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                if assignment[var_idx] == is_pos_aff:
                    if counts[c_aff] == 0:
                        unsat_set.discard(c_aff)
                    counts[c_aff] += 1
                else:
                    counts[c_aff] -= 1
                    if counts[c_aff] == 0:
                        unsat_set.add(c_aff)
            
            new_unsat = len(unsat_set)
            
            # SA: accept worse moves with decreasing probability
            if new_unsat > old_unsat:
                delta = new_unsat - old_unsat
                T = sa_temp * max(0.01, 1.0 - step / max_steps)
                if rng.random() > math.exp(-delta / max(T, 0.001)):
                    # Reject: flip back
                    assignment[var_idx] = not assignment[var_idx]
                    for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                        if assignment[var_idx] == is_pos_aff:
                            if counts[c_aff] == 0:
                                unsat_set.discard(c_aff)
                            counts[c_aff] += 1
                        else:
                            counts[c_aff] -= 1
                            if counts[c_aff] == 0:
                                unsat_set.add(c_aff)
        
        return False, assignment, unsat_set, counts
    
    best_assignment = [False] * n
    best_unsat_count = m + 1
    
    # Multi-restart
    restart_budget = 2000
    base_steps = 5000
    
    for restart in range(restart_budget):
        if time.monotonic() > deadline - 5.0:
            break
        
        # Initialize: random
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Vary temperature for diversity
        sa_temp = 0.5 + rng.random() * 1.5
        
        remaining = deadline - time.monotonic()
        if remaining < 10.0:
            break
        elif remaining < 30.0:
            steps = 2000
        elif remaining < 60.0:
            steps = 4000
        else:
            steps = base_steps
        
        solved, assignment, unsat_set, counts = optimized_walksat(assignment, steps, sa_temp)
        
        if solved:
            return assignment
        
        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
    
    return best_assignment
# EVOLVE-BLOCK-END
