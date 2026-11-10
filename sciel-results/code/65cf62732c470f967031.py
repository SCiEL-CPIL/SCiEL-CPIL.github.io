import random
import time
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290  # 10s buffer
    
    # Pre-process clauses
    # clauses is a list of lists of signed literals
    # Build adjacency: var -> list of (clause_index, sign)
    var_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_clauses[v].append((i, sign))
            
    # Belief Propagation
    # Initialize beliefs to 0.5
    beliefs = [0.5] * n
    
    # Run BP for a few iterations or until time is up
    max_bp_iters = 100
    for iter in range(max_bp_iters):
        if time.time() > deadline - 10:
            break
        new_beliefs = [0.5] * n
        for v in range(n):
            # Calculate message from all clauses to v
            # m_c_to_v = 1 - prod(1 - b_u^sign) for u in c \ {v}
            # Belief update: b_v proportional to prod(m_c_to_v)
            # Since we only care about the ratio, we can just multiply the messages
            # But wait, the message m_c_to_v depends on the other variables' beliefs.
            # This is a simple BP update.
            
            prod_msg = 1.0
            for c_idx, sign in var_clauses[v]:
                clause = clauses[c_idx]
                # Calculate probability that clause is satisfied by others
                prod_other_false = 1.0
                for lit in clause:
                    u = abs(lit) - 1
                    if u == v:
                        continue
                    # Probability that u is false (considering sign)
                    if lit > 0:
                        p_false = 1.0 - beliefs[u]
                    else:
                        p_false = beliefs[u]
                    prod_other_false *= p_false
                m_c_to_v = 1.0 - prod_other_false
                prod_msg *= m_c_to_v
                
            # Normalize?
            # The belief is proportional to prod_msg.
            # But we need to map this to a probability in [0,1].
            # In standard BP, we would also have a prior.
            # Here, the "evidence" is just the clause constraints.
            # If prod_msg is very small, it means the variable is constrained to be false?
            # Actually, if the variable is true, it helps satisfy clauses.
            # The message m_c_to_v is the probability that c is satisfied WITHOUT v.
            # If v is true, c is definitely satisfied (if v is in c).
            # If v is false, c is satisfied with prob m_c_to_v.
            # So the likelihood of v=True is 1 (for all c containing v).
            # The likelihood of v=False is prod(m_c_to_v).
            # So P(v=True) / P(v=False) = 1 / prod(m_c_to_v)
            # P(v=True) = 1 / (1 + prod(m_c_to_v))
            
            if prod_msg < 1e-10:
                prod_msg = 1e-10
            new_beliefs[v] = 1.0 / (1.0 + prod_msg)
        
        beliefs = new_beliefs
        
        # Check if assignment is satisfying
        assignment = [b > 0.5 for b in beliefs]
        unsat_count = 0
        for c in clauses:
            sat = False
            for lit in c:
                if assignment[abs(lit)-1] == (lit > 0):
                    sat = True
                    break
            if not sat:
                unsat_count += 1
        if unsat_count == 0:
            return assignment

    # Use beliefs to initialize WalkSAT
    assignment = [b > 0.5 for b in beliefs]
    
    # WalkSAT
    rng = random.Random(0)
    flip_prob = 0.5
    max_steps_per_restart = 5000
    best_assignment = assignment[:]
    best_unsat = sum(1 for c in clauses if not any(assignment[abs(l)-1] == (l > 0) for l in c))
    
    while time.time() < deadline:
        # Restart with current best or random?
        # Let's restart with the current assignment if it's not too bad, or random if stuck.
        # Actually, standard WalkSAT restarts with random assignment.
        # But we have a good start from BP. Let's try to improve it.
        # If we fail, we can restart from BP assignment again or random.
        
        current_assignment = assignment[:]
        steps = 0
        while steps < max_steps_per_restart:
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = [c for c in clauses if not any(current_assignment[abs(l)-1] == (l > 0) for l in c)]
            if not unsat_clauses:
                return current_assignment
            
            # Update best
            if len(unsat_clauses) < best_unsat:
                best_unsat = len(unsat_clauses)
                best_assignment = current_assignment[:]
            
            # Pick a clause
            clause = rng.choice(unsat_clauses)
            
            # Pick a literal to flip
            if rng.random() < flip_prob:
                # Pick random literal from clause
                lit = rng.choice(clause)
            else:
                # Pick literal that minimizes unsat count (MinConflicts heuristic)
                # This is expensive. Let's stick to random for speed, or a simple heuristic.
                # For 150 vars, checking 3 literals is cheap.
                min_unsat = float('inf')
                best_lit = None
                for lit in clause:
                    v = abs(lit) - 1
                    # Temporarily flip
                    old_val = current_assignment[v]
                    current_assignment[v] = not old_val
                    # Count unsat
                    u = 0
                    for c in clauses:
                        if not any(current_assignment[abs(l)-1] == (l > 0) for l in c):
                            u += 1
                    current_assignment[v] = old_val
                    if u < min_unsat:
                        min_unsat = u
                        best_lit = lit
                lit = best_lit
            
            v = abs(lit) - 1
            current_assignment[v] = not current_assignment[v]
            steps += 1
        
        # If we exited the inner loop without finding a solution, restart
        # Use the best assignment found in this restart or the global best?
        # Let's use the global best to avoid getting stuck in a bad local minimum.
        assignment = best_assignment[:]
        # Or restart from BP assignment?
        # Let's alternate: sometimes BP, sometimes best.
        # For now, just use best_assignment.

    return best_assignment
