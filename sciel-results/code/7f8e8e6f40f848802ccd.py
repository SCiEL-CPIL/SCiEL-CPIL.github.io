"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from collections import defaultdict

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    # Deadline guard: 325s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 325.0

    # Pre-compute clause structure
    # clause_vars: (m, 3) 0-based variable indices
    # clause_polarity: (m, 3) 1 for positive, -1 for negative
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_polarity = np.zeros((m, 3), dtype=np.int8)
    for ci, c in enumerate(clauses):
        for li, l in enumerate(c):
            clause_vars[ci, li] = abs(l) - 1
            clause_polarity[ci, li] = 1 if l > 0 else -1

    # Build adjacency lists
    # var_to_clauses: list of lists of (clause_idx, polarity)
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for li in range(3):
            vi = int(clause_vars[ci, li])
            pol = int(clause_polarity[ci, li])
            var_to_clauses[vi].append((ci, pol))

    # Helper: Check if an assignment satisfies all clauses
    def check_solution(assignment):
        # assignment is a list of bools or 0/1 ints
        # We use 0/1 ints internally
        lit_vals = np.array(assignment, dtype=np.int8)
        lit_vals = 2 * lit_vals - 1 # -1 for 0, 1 for 1
        clause_lit_vals = lit_vals[clause_vars]
        clause_products = clause_lit_vals * clause_polarity
        # If any product is positive, the literal is satisfied.
        # A clause is unsatisfied if all 3 products are negative.
        unsat_clauses = (clause_products < 0).all(axis=1)
        return not unsat_clauses.any()

    # --- Belief Propagation (Sum-Product) ---
    # Messages:
    # m_c_v[c, l]: Message from clause c to variable v (at position l in clause c)
    # m_v_c[v, c]: Message from variable v to clause c
    
    # Initialize messages to 1.0 (neutral)
    # To save memory and time, we can store messages in dictionaries or arrays.
    # Since structure is fixed, we use arrays indexed by (clause, var_position) and (var, clause_occurrence).
    
    # For variable v, it appears in k_v clauses.
    # We need to map (v, c) to an index for m_v_c.
    var_clause_occurrence_idx = {} # (v, c) -> index in v's message array
    var_msg_counts = [len(cl) for cl in var_to_clauses]
    
    # m_v_c[v] is an array of size var_msg_counts[v]
    # m_c_v[c] is an array of size 3 (one for each literal position)
    
    # We will implement a simplified BP that updates messages iteratively.
    
    def run_bp(initial_probs=None, max_iters=100, tol=1e-6):
        """
        Runs Belief Propagation.
        Returns marginal probabilities for each variable being True (1).
        """
        # Initialize messages
        # m_c_v[c, l] = P(literal l in clause c is True | rest of clause)
        # Actually, standard BP messages are:
        # m_{c->v}(x_v) : probability of x_v given messages from other vars in c
        # m_{v->c}(x_v) : probability of x_v given messages from other clauses in v
        
        # Let's use the standard form:
        # m_{c->v}(x_v) = sum_{x_{c\setminus v}} P(c | x_c) prod_{v' in c\setminus v} m_{v'->c}(x_{v'})
        # For 3-SAT clause (l1, l2, l3), P(c|x) is 1 if any li is True, 0 otherwise.
        # This is equivalent to 1 - P(all li are False).
        
        # Let's define messages as log-probabilities or just probabilities.
        # Probabilities are easier to handle for 3-SAT.
        
        # m_c_v[c, l] represents the message from clause c to the variable at position l.
        # It is a pair (p0, p1) or just p1 (prob of being True).
        # Let's store m_c_v[c, l] as prob that the literal at (c,l) is True, given the rest of the clause.
        # Actually, the message m_{c->v} depends on the assignment of v.
        # m_{c->v}(0) and m_{c->v}(1).
        
        # Simplification for 3-SAT:
        # The clause is satisfied unless all literals are false.
        # m_{c->v}(0) = 1 (Clause is satisfied if v is False, regardless of others? No.)
        # If v is False, the clause is satisfied if at least one of the other two is True.
        # m_{c->v}(0) = 1 - P(other1 is False) * P(other2 is False)
        #              = 1 - (1 - m_{v1->c}(1)) * (1 - m_{v2->c}(1))
        
        # m_{c->v}(1) = 1 (Clause is satisfied if v is True, regardless of others)
        
        # So we only need to track m_{v->c}(1) (prob variable v is True, given other clauses).
        # Let's call this msg_vc[v, idx].
        
        # Initialize msg_vc to 0.5 (uninformed)
        msg_vc = []
        for v in range(n):
            msg_vc.append(np.full(var_msg_counts[v], 0.5, dtype=np.float64))
            
        # msg_vc[v, k] is the message from variable v to its k-th clause.
        
        # We also need to map (v, c) to the index k in msg_vc[v].
        # And for each clause c, we need to know which (v, k) correspond to its 3 literals.
        # clause_msg_info[c] = list of 3 tuples (v, k)
        
        clause_msg_info = []
        for c in range(m):
            info = []
            for l in range(3):
                v = int(clause_vars[c, l])
                # Find the index k in var_to_clauses[v] where clause is c
                # We pre-computed var_to_clauses. Let's create a lookup.
                # This is slow to do inside BP. Pre-compute it.
                pass
            
        # Pre-compute lookup for (v, c) -> index in var_to_clauses[v]
        # And for each clause c, list of (v, idx_in_v)
        
        v_c_to_idx = {}
        for v in range(n):
            for k, (c, pol) in enumerate(var_to_clauses[v]):
                v_c_to_idx[(v, c)] = k
                
        clause_to_var_info = []
        for c in range(m):
            info = []
            for l in range(3):
                v = int(clause_vars[c, l])
                idx = v_c_to_idx[(v, c)]
                info.append((v, idx))
            clause_to_var_info.append(info)
        
        # BP Iteration
        # Update msg_vc based on msg_vc of neighbors?
        # No, msg_vc[v] depends on messages from other clauses to v.
        # Let's define msg_cv[c, l] as the message from clause c to variable v (at pos l).
        # msg_cv[c, l] is a vector [p0, p1] or we can just compute the marginal update.
        
        # Let's use the standard belief update:
        # Belief b_v(x_v) \propto prod_{c \in N(v)} m_{c->v}(x_v)
        
        # Messages m_{c->v}(x_v):
        # For clause c with literals l1, l2, l3.
        # m_{c->v1}(0) = 1 - (1 - m_{v2->c}(1)) * (1 - m_{v3->c}(1))
        # m_{c->v1}(1) = 1
        
        # So we only need to store m_{v->c}(1). Let's call it M[v, c].
        # M is stored in msg_vc as above.
        
        # Update rule for M[v, c] (message from v to c):
        # M[v, c](1) \propto prod_{c' \in N(v) \setminus {c}} m_{c'->v}(1)
        # m_{c'->v}(1) = 1 (as derived above for 3-SAT clause satisfaction)
        # Wait, if m_{c'->v}(1) is always 1, then the product is 1.
        # This implies M[v, c](1) is constant?
        # No, the normalization matters.
        # b_v(1) \propto m_{c1->v}(1) * m_{c2->v}(1) * ...
        # b_v(0) \propto m_{c1->v}(0) * m_{c2->v}(0) * ...
        
        # m_{c->v}(0) depends on M of other vars in c.
        # m_{c->v}(1) = 1.
        
        # So:
        # b_v(1) = 1
        # b_v(0) = prod_{c \in N(v)} m_{c->v}(0)
        
        # m_{c->v}(0) = 1 - (1 - M[v1', c]) * (1 - M[v2', c]) where v1', v2' are the other two vars in c.
        
        # So we can compute b_v(0) and b_v(1) directly.
        # Then M[v, c](1) is proportional to b_v(1) / (b_v(0) + b_v(1))?
        # No, M[v, c](1) is the message from v to c.
        # M[v, c](1) \propto prod_{c' \in N(v) \setminus {c}} m_{c'->v}(1) = 1.
        # M[v, c](0) \propto prod_{c' \in N(v) \setminus {c}} m_{c'->v}(0).
        
        # Let's maintain M[v, c] as the ratio P(v=1)/P(v=0) given all clauses except c?
        # Or just store the messages as probabilities.
        
        # Let's store M[v, c] as the probability that v is True, given all clauses in N(v) except c.
        # M[v, c](1) = b_v(1) / (b_v(0) + b_v(1)) where b_v is computed excluding c.
        
        # Iteration:
        # 1. For each clause c, for each var v in c:
        #    Compute m_{c->v}(0) using current M of other vars in c.
        #    m_{c->v}(0) = 1 - (1 - M[v1, c]) * (1 - M[v2, c])
        #    m_{c->v}(1) = 1
        #    Store these temporarily? Or use them immediately to update M[v, c]?
        #    M[v, c] depends on m_{c'->v} for c' != c.
        #    So we need to update all M[v, c] simultaneously.
        
        # Let's use a double buffer.
        M_new = [np.array(m, dtype=np.float64) for m in msg_vc]
        
        for _ in range(max_iters):
            # Compute m_{c->v}(0) for all (c, v)
            # m_c_v_0[c, l]
            m_c_v_0 = np.ones((m, 3), dtype=np.float64)
            
            for c in range(m):
                # Get the 3 (v, idx) pairs
                v0, i0 = clause_to_var_info[c][0]
                v1, i1 = clause_to_var_info[c][1]
                v2, i2 = clause_to_var_info[c][2]
                
                # M[v0, c] is msg_vc[v0][i0]
                p0 = msg_vc[v0][i0]
                p1 = msg_vc[v1][i1]
                p2 = msg_vc[v2][i2]
                
                # m_{c->v0}(0) = 1 - (1 - p1) * (1 - p2)
                m_c_v_0[c, 0] = 1 - (1 - p1) * (1 - p2)
                m_c_v_0[c, 1] = 1 - (1 - p0) * (1 - p2)
                m_c_v_0[c, 2] = 1 - (1 - p0) * (1 - p1)
                
                # m_{c->v}(1) is always 1, so we don't need to store it.
            
            # Update M[v, c]
            # M[v, c] = P(v=1 | N(v)\c)
            # P(v=1 | N(v)\c) \propto prod_{c' \in N(v)\c} m_{c'->v}(1) = 1
            # P(v=0 | N(v)\c) \propto prod_{c' \in N(v)\c} m_{c'->v}(0)
            
            # Let prod_0 = prod_{c' \in N(v)\c} m_{c'->v}(0)
            # M[v, c] = 1 / (1 + prod_0)
            
            for v in range(n):
                # Calculate total product of m_{c'->v}(0) for all c' in N(v)
                # Then for each c in N(v), divide by m_{c->v}(0)
                
                # Get all clauses for v
                # var_to_clauses[v] is list of (c, pol)
                # We need m_c_v_0[c, l] where l is the position of v in c.
                
                # Let's pre-compute the position of v in each clause c for v's neighbors.
                # This is a bit slow in the inner loop.
                # Let's just loop.
                
                total_prod = 1.0
                # List of (c, m_c_v_0_val)
                neighbor_msgs = []
                for c, pol in var_to_clauses[v]:
                    # Find position of v in clause c
                    # clause_vars[c] contains v.
                    # We can find l by checking which position has v.
                    # Since it's small (3), we can just check.
                    l = 0
                    if clause_vars[c, 0] == v:
                        l = 0
                    elif clause_vars[c, 1] == v:
                        l = 1
                    else:
                        l = 2
                        
                    msg_val = m_c_v_0[c, l]
                    total_prod *= msg_val
                    neighbor_msgs.append((c, msg_val))
                
                # Now update M[v, c] for each c
                for k, (c, msg_val) in enumerate(neighbor_msgs):
                    if msg_val > 1e-12:
                        prod_excl_c = total_prod / msg_val
                    else:
                        prod_excl_c = 0.0 # If any message is 0, product is 0
                    
                    # M[v, c] = 1 / (1 + prod_excl_c)
                    # Add small epsilon to avoid division by zero or overflow
                    M_new[v][k] = 1.0 / (1.0 + prod_excl_c)
            
            # Check convergence
            # Compare M_new and msg_vc
            # If max diff < tol, break
            # For speed, we can just check every 10 iterations or use a simpler metric.
            # Let's just update and continue for fixed iterations to avoid overhead.
            msg_vc = M_new
            
        # Compute final marginals
        # b_v(1) \propto 1
        # b_v(0) \propto prod_{c \in N(v)} m_{c->v}(0)
        
        # We need m_c_v_0 from the last iteration.
        # Recalculate m_c_v_0 using final msg_vc
        m_c_v_0 = np.ones((m, 3), dtype=np.float64)
        for c in range(m):
            v0, i0 = clause_to_var_info[c][0]
            v1, i1 = clause_to_var_info[c][1]
            v2, i2 = clause_to_var_info[c][2]
            
            p0 = msg_vc[v0][i0]
            p1 = msg_vc[v1][i1]
            p2 = msg_vc[v2][i2]
            
            m_c_v_0[c, 0] = 1 - (1 - p1) * (1 - p2)
            m_c_v_0[c, 1] = 1 - (1 - p0) * (1 - p2)
            m_c_v_0[c, 2] = 1 - (1 - p0) * (1 - p1)
            
        marginals = np.zeros(n, dtype=np.float64)
        for v in range(n):
            prod_0 = 1.0
            for c, pol in var_to_clauses[v]:
                # Find position l
                if clause_vars[c, 0] == v:
                    l = 0
                elif clause_vars[c, 1] == v:
                    l = 1
                else:
                    l = 2
                prod_0 *= m_c_v_0[c, l]
            
            # P(v=1) = 1 / (1 + prod_0)
            marginals[v] = 1.0 / (1.0 + prod_0)
            
        return marginals

    # --- Greedy Assignment with Unit Propagation ---
    
    def greedy_solve(marginals, seed):
        """
        Assigns variables greedily based on marginals.
        Returns a list of bools or None if contradiction.
        """
        rng = random.Random(seed)
        
        # Copy marginals
        margins = marginals.copy()
        
        # State:
        # assignment: -1 (unassigned), 0 (False), 1 (True)
        assignment = np.full(n, -1, dtype=np.int8)
        
        # Active clauses: initially all.
        # We need to track for each clause how many literals are satisfied/falsified.
        # clause_sat_count[c]: number of literals satisfied
        # clause_unsat_count[c]: number of literals falsified
        
        # Instead of updating all clauses, we can just check at the end.
        # But for unit propagation, we need to know if a clause becomes unit.
        
        # Let's use a simple approach:
        # 1. Sort variables by confidence (|marginal - 0.5|)
        # 2. Iterate through sorted vars.
        # 3. Assign var.
        # 4. Check if any clause is now unsatisfied (all 3 literals false). If so, fail.
        # 5. Check if any clause is now unit (1 unassigned, 2 satisfied). If so, force the unassigned var.
        #    - If the forced var contradicts previous assignment, fail.
        #    - If not, assign it and repeat.
        
        # To make this efficient, we need to track clause states.
        # clause_state[c]: 0=unknown, 1=satisfied, 2=unsatisfied
        # unassigned_vars_in_clause[c]: list of vars still unassigned
        
        # This is complex to maintain dynamically.
        # Given n=150, we can afford to re-check clauses for each assignment if we do it smartly.
        # Or we can just assign all vars greedily and check at the end.
        # If it fails, we can try a different ordering or restart.
        
        # Let's try the simple "assign all then check" first.
        # It might fail often, but it's fast.
        
        # Better: "Assign, Check, Backtrack" is hard without full DPLL.
        # Let's use the "Confidence Order" and just assign.
        
        order = np.argsort(np.abs(margins - 0.5))[::-1] # Most confident first
        
        for v in order:
            # Determine value
            if margins[v] >= 0.5:
                val = 1
            else:
                val = 0
            assignment[v] = val
            
        # Check solution
        if check_solution(assignment):
            return assignment.tolist()
        
        # If simple greedy failed, try a few random perturbations of the order or values
        # Or just return None to trigger WalkSAT fallback.
        return None

    # --- WalkSAT Fallback ---
    
    def walksat_fallback(init_assignment=None, seed=42):
        """
        Optimized WalkSAT.
        """
        rng = np.random.default_rng(seed)
        
        if init_assignment is not None:
            assignment = np.array(init_assignment, dtype=np.int8)
        else:
            assignment = rng.random(n).astype(np.int8)
            
        # Pre-compute clause structures for fast flip
        # We already have clause_vars, clause_polarity
        # We need var_clause_indices and var_clause_polarities
        var_clause_indices = []
        var_clause_polarities = []
        for vi in range(n):
            if var_to_clauses[vi]:
                var_clause_indices.append(np.array([x[0] for x in var_to_clauses[vi]], dtype=np.int32))
                var_clause_polarities.append(np.array([x[1] for x in var_to_clauses[vi]], dtype=np.int8))
            else:
                var_clause_indices.append(np.array([], dtype=np.int32))
                var_clause_polarities.append(np.array([], dtype=np.int8))
                
        def init_clause_unsat(assignment):
            lit_vals = 2 * assignment.astype(np.int8) - 1
            clause_lit_vals = lit_vals[clause_vars]
            clause_products = clause_lit_vals * clause_polarity
            return (clause_products < 0).sum(axis=1).astype(np.int8)
            
        clause_unsat_count = init_clause_unsat(assignment)
        current_unsat = int((clause_unsat_count >= 3).sum())
        
        if current_unsat == 0:
            return assignment.tolist()
            
        best_assignment = assignment.copy()
        best_unsat = current_unsat
        
        max_steps = 200000
        walk_prob = 0.2
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            # Find unsat clause
            unsat_mask = clause_unsat_count >= 3
            if not unsat_mask.any():
                return assignment.tolist()
                
            unsat_indices = np.where(unsat_mask)[0]
            unsat_idx = int(rng.choice(unsat_indices))
            cv = clause_vars[unsat_idx]
            
            # Choose variable to flip
            if rng.random() < walk_prob:
                vi = int(rng.choice(cv))
            else:
                # Greedy
                best_gain = -10
                best_vi = -1
                for li in range(3):
                    v_cand = int(cv[li])
                    # Compute gain
                    cidx = var_clause_indices[v_cand]
                    cpol = var_clause_polarities[v_cand]
                    old_val = assignment[v_cand]
                    old_lit_val = 2 * old_val - 1
                    was_sat = (old_lit_val * cpol) > 0
                    
                    gain = 0
                    sat_clauses = cidx[was_sat]
                    if len(sat_clauses) > 0:
                        counts = clause_unsat_count[sat_clauses]
                        gain -= int((counts == 2).sum())
                        
                    unsat_clauses = cidx[~was_sat]
                    if len(unsat_clauses) > 0:
                        counts = clause_unsat_count[unsat_clauses]
                        gain += int((counts == 3).sum())
                        
                    g = gain + rng.random() * 0.01
                    if g > best_gain:
                        best_gain = g
                        best_vi = v_cand
                vi = best_vi
                
            # Flip
            old_val = assignment[vi]
            assignment[vi] = 1 - old_val
            
            # Update clause unsat counts
            cidx = var_clause_indices[vi]
            cpol = var_clause_polarities[vi]
            if len(cidx) > 0:
                old_lit_val = 2 * old_val - 1
                was_sat = (old_lit_val * cpol) > 0
                clause_unsat_count[cidx[was_sat]] += 1
                clause_unsat_count[cidx[~was_sat]] -= 1
                
            # Recalculate total unsat
            # This is O(m). Can we do better?
            # We can maintain a count of unsat clauses.
            # But for simplicity and correctness, let's just check if we solved it.
            # Actually, updating total_unsat incrementally is easy.
            # We didn't do that here. Let's just check the specific clause?
            # No, other clauses might have changed status.
            
            # For efficiency, let's just count unsat clauses every 100 steps or so?
            # No, we need to know when we hit 0.
            # Let's just compute it. m is ~600. 600 ops is fast.
            current_unsat = int((clause_unsat_count >= 3).sum())
            
            if current_unsat == 0:
                return assignment.tolist()
                
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment.copy()
                walk_prob = max(0.1, walk_prob * 0.95)
            else:
                walk_prob = min(0.8, walk_prob * 1.01)
                
        return best_assignment.tolist()

    # --- Main Logic ---
    
    # 1. Try BP + Greedy
    # Run BP with a few different initializations/seeds?
    # BP is deterministic if initialized randomly? No, we initialized to 0.5.
    # So BP result is unique for this instance.
    
    try:
        marginals = run_bp(max_iters=50)
        
        # Try greedy assignment
        sol = greedy_solve(marginals, seed=42)
        if sol:
            return sol
            
        # If greedy failed, use marginals to initialize WalkSAT
        # Convert marginals to binary assignment
        init_assign = (marginals >= 0.5).astype(np.int8)
        
        # Try WalkSAT with this initialization
        sol = walksat_fallback(init_assignment=init_assign.tolist(), seed=42)
        if sol and check_solution(sol):
            return sol
            
    except Exception:
        pass
        
    # 2. Fallback to pure WalkSAT with random init
    # Multiple restarts
    for restart in range(10):
        if time.time() > deadline:
            break
        sol = walksat_fallback(init_assignment=None, seed=42 + restart)
        if sol and check_solution(sol):
            return sol
            
    # 3. Final fallback: Random
    rng = np.random.default_rng(42)
    return (rng.random(n) < 0.5).tolist()

# EVOLVE-BLOCK-END
