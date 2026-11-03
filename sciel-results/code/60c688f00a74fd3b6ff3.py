"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from itertools import combinations

def _generate_supports(m):
    """Generate all C(D, m) supports."""
    supports = []
    for comb in combinations(range(D), m):
        supports.append(comb)
    return supports

def construct():
    deadline = time.time() + 110  # Safety margin before 150s limit
    
    # Strategy:
    # 1. Select supports (subsets of coordinates of size m) such that no two supports share more than m/2 indices.
    #    For m=4, max intersection size is 2.
    # 2. This is equivalent to finding a Max Independent Set in a graph where nodes are supports and edges connect supports with intersection > 2.
    # 3. We model this as a Max-SAT problem:
    #    Variables: x_s = 1 if support s is selected.
    #    Clauses: For every pair (s1, s2) with |s1 ∩ s2| > 2, clause (!x_s1 OR !x_s2).
    #    Goal: Maximize sum(x_s).
    # 4. Solve using WalkSAT (Stochastic Local Search).
    
    m = 4
    max_intersection = m // 2 # 2
    
    supports = _generate_supports(m)
    n_supports = len(supports)
    
    # Build conflict list: for each support i, list of supports j that conflict with i.
    # Conflict if intersection size > max_intersection.
    # To speed up, we can precompute a bitmask or just check intersection.
    # With n_supports ~ 330, checking all pairs is 330^2 / 2 ~ 54k checks. Very fast.
    
    conflicts = [[] for _ in range(n_supports)]
    for i in range(n_supports):
        s_i = set(supports[i])
        for j in range(i + 1, n_supports):
            s_j = set(supports[j])
            inter = len(s_i.intersection(s_j))
            if inter > max_intersection:
                conflicts[i].append(j)
                conflicts[j].append(i)
    
    # WalkSAT Parameters
    # State: boolean array of length n_supports.
    # Objective: Number of satisfied clauses.
    # Clause (i, j) is satisfied if NOT (x_i AND x_j).
    # Unsatisfied if x_i AND x_j.
    
    # Initialize randomly
    # Probability of selecting a support. 
    # Since we want a large set, we might start with high probability, but conflicts will prune.
    # Let's start with random assignment with p=0.5.
    
    best_score = -1
    best_state = None
    
    # Precompute clause indices for faster evaluation?
    # Clause list is just pairs (i, j) from conflicts.
    clauses = []
    for i in range(n_supports):
        for j in conflicts[i]:
            if i < j: # Avoid duplicates
                clauses.append((i, j))
    
    n_clauses = len(clauses)
    
    def evaluate(state):
        """Count satisfied clauses."""
        sat = 0
        for i, j in clauses:
            if not (state[i] and state[j]):
                sat += 1
        return sat
        
    def num_unsatisfied_for_var(var, state):
        """Count how many clauses involving 'var' are currently unsatisfied."""
        count = 0
        for j in conflicts[var]:
            if state[var] and state[j]:
                count += 1
        return count

    # WalkSAT Loop
    # We run multiple restarts if time permits.
    # Each restart: random init, then flip bits.
    
    max_time = deadline
    restarts = 0
    
    while time.time() < max_time:
        restarts += 1
        # Random initialization
        # Heuristic: Start with all False? Or random?
        # All False is trivially valid but score 0.
        # Random 0.5 might be better to find large sets.
        state = np.random.randint(0, 2, size=n_supports).astype(bool)
        
        current_score = evaluate(state)
        
        if current_score > best_score:
            best_score = current_score
            best_state = state.copy()
            
        # Local Search
        # Standard WalkSAT:
        # 1. Pick a random unsatisfied clause.
        # 2. Pick a variable in that clause.
        #    - With prob p (e.g., 0.5), pick randomly from the clause variables.
        #    - With prob 1-p, pick the variable that maximizes the number of newly satisfied clauses (greedy).
        # 3. Flip the variable.
        
        # Optimization: Instead of picking a random unsatisfied clause, we can iterate.
        # But finding an unsatisfied clause requires scanning.
        # Since n_clauses is small (~few thousand), scanning is fast.
        
        # To speed up, we can maintain a list of unsatisfied clauses? 
        # Or just scan.
        
        steps = 0
        max_steps = 5000 # Limit steps per restart to avoid infinite loops
        
        while steps < max_steps and time.time() < max_time:
            # Find an unsatisfied clause
            # If none, we have a valid solution.
            unsat_clause_idx = -1
            for idx, (i, j) in enumerate(clauses):
                if state[i] and state[j]:
                    unsat_clause_idx = idx
                    break
            
            if unsat_clause_idx == -1:
                # All clauses satisfied. We have a valid independent set.
                # We can try to add more supports?
                # WalkSAT flips bits. If we are at a local optimum where all clauses are sat,
                # we can't add more without breaking clauses unless we flip a False to True 
                # that doesn't conflict.
                # Actually, if all clauses are sat, the current set is an independent set.
                # To improve, we would need to find a variable that is False, 
                # and setting it to True doesn't violate any clause.
                # This is equivalent to checking if any neighbor of a False node is also False? No.
                # If x_i is False, clause (i, j) is satisfied regardless of x_j.
                # So we can set x_i to True if all its neighbors x_j are False?
                # No, if x_j is True, clause (i, j) becomes (True, True) -> Unsat.
                # So we can only set x_i to True if all neighbors x_j are False.
                
                # Let's try to greedily add nodes.
                improved = True
                while improved and time.time() < max_time:
                    improved = False
                    for i in range(n_supports):
                        if not state[i]:
                            # Check if all neighbors are False
                            can_add = True
                            for j in conflicts[i]:
                                if state[j]:
                                    can_add = False
                                    break
                            if can_add:
                                state[i] = True
                                improved = True
                                # Update score
                                # New score = old score + (number of clauses involving i that were sat and remain sat?)
                                # Actually, if all neighbors are False, all clauses (i, j) were sat (because state[i] was False).
                                # After setting to True, they are still sat (because state[j] is False).
                                # So score doesn't change? 
                                # Wait, score is number of SATISFIED clauses.
                                # Clause (i, j) is sat if NOT (xi AND xj).
                                # If xi=0, it's sat. If xi=1, xj=0, it's sat.
                                # So score remains the same.
                                # But the size of the set increases!
                                # The objective function for MIS is maximize sum(x).
                                # My evaluate() function counts satisfied clauses.
                                # Maximizing satisfied clauses is equivalent to minimizing conflicts.
                                # But we want to maximize the number of selected nodes.
                                # These are not exactly the same.
                                # If I have a set with 0 conflicts, score = n_clauses.
                                # If I add a node that doesn't conflict, score stays n_clauses.
                                # So WalkSAT on "satisfied clauses" will stop improving score once it reaches n_clauses.
                                # It won't try to add more nodes if it's already conflict-free.
                                # So the "greedily add nodes" step is crucial.
                                
                # After adding nodes, update best
                current_size = np.sum(state)
                # The actual score for the task is the number of vectors.
                # Each support gives 2^m vectors.
                # So total vectors = current_size * (2**m).
                # We should track current_size as the optimization target for the search.
                # But WalkSAT optimizes clause satisfaction.
                # Let's switch objective to: Minimize conflicts (unsat clauses) + Maximize nodes?
                # Or just use the "add nodes" step whenever we hit a valid state.
                
                if current_size > best_score: # Reusing best_score as best_size for simplicity in this block
                    best_score = current_size
                    best_state = state.copy()
                
                # Continue searching? 
                # If we can't add any more nodes, we are at a maximal independent set.
                # We should restart to find a different one.
                break

            # Pick a variable to flip from the unsatisfied clause
            i, j = clauses[unsat_clause_idx]
            
            # Heuristic: Flip the one that causes fewer NEW conflicts?
            # Or random?
            # Let's try: with prob 0.5 random, else greedy (fewer conflicts).
            
            if random.random() < 0.5:
                var = i if random.random() < 0.5 else j
            else:
                # Calculate "cost" of flipping i to False (if it is True) or True (if it is False)?
                # Wait, in an unsat clause (xi=1, xj=1), we MUST flip one to 0.
                # So we are choosing which one to turn OFF.
                # Cost of turning off i: number of clauses involving i that are currently SATISFIED 
                # and will become UNSATISFIED? No.
                # If we turn off i, clauses (i, k) where xk=1 become SATISFIED (were they? No, if xi=1, xk=1, they were UNSAT).
                # So turning off i SATISFIES all currently UNSAT clauses involving i.
                # Does it UNSATISFY any currently SATISFIED clauses?
                # Clause (i, k) with xk=0 is SAT. If we set xi=0, it remains SAT.
                # Clause (i, k) with xk=1 is UNSAT. If we set xi=0, it becomes SAT.
                # So turning off a variable in an unsat clause NEVER creates new unsat clauses.
                # It only fixes existing ones.
                # So the "greedy" choice is to turn off the variable that is part of the MOST unsat clauses.
                
                cost_i = num_unsatisfied_for_var(i, state)
                cost_j = num_unsatisfied_for_var(j, state)
                
                if cost_i >= cost_j:
                    var = i
                else:
                    var = j
            
            # Flip
            state[var] = not state[var]
            steps += 1
            
            # Update score? 
            # We don't need to update clause score every step if we just scan at the start of the loop.
            # But we do need to check if we reached a valid state to trigger "add nodes".
            # The scan at the start of the loop handles this.
            
            # Optional: Check size periodically?
            # The "add nodes" step is only triggered when no unsat clauses are found.
            # This means we only add nodes when we have a valid independent set.
            # This is a good strategy: find a valid set, then maximize it greedily.
            
            # To avoid getting stuck in a small valid set, the random restarts will help.
            # Also, the flipping process might move us to a different valid set.
            
    # Construct final vectors from best_state
    if best_state is None:
        # Fallback: just pick one support
        best_state = np.zeros(n_supports, dtype=bool)
        best_state[0] = True
        
    # best_state is a boolean mask of selected supports
    # For each selected support, generate all 2^m sign vectors.
    
    selected_vectors = []
    for idx in range(n_supports):
        if best_state[idx]:
            support = supports[idx]
            # Generate all 2^m sign combinations
            for mask in range(1 << m):
                v = np.zeros(D, dtype=np.int64)
                for k, pos in enumerate(support):
                    if mask & (1 << k):
                        v[pos] = 1
                    else:
                        v[pos] = -1
                selected_vectors.append(v)
    
    if len(selected_vectors) == 0:
        # Should not happen, but fallback
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        return np.array([v])
        
    return np.array(selected_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
