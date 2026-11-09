import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance_sa(values, time_budget):
        n = len(values)
        if n <= 2:
            return list(values)
        
        v_set = set(values)
        
        # Build conflict graph adjacency matrix (numpy for speed)
        # adj[i, j] = 1 if values[i] and values[j] cannot both be in the sum-free set
        # Condition for conflict between i and j (i != j):
        # 1. values[i] + values[j] is in v_set
        # 2. 2 * values[i] == values[j]
        # 3. 2 * values[j] == values[i]
        
        adj = np.zeros((n, n), dtype=np.int8)
        
        # Optimization: Precompute sums? 
        # n is max 240. n^2 is 57600. 
        # Checking membership in a set of 240 items is fast.
        
        for i in range(n):
            vi = values[i]
            for j in range(i + 1, n):
                vj = values[j]
                if (vi + vj) in v_set or 2 * vi == vj or 2 * vj == vi:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        # Greedy Initialization
        # Sort by degree (number of conflicts)
        degrees = adj.sum(axis=1)
        order = np.argsort(degrees) # Lowest degree first
        
        x = np.zeros(n, dtype=np.int8)
        # Greedily add nodes if no conflict with already added nodes
        for idx in order:
            if x[idx] == 0:
                # Check if adding idx conflicts with any currently selected node
                # conflict = (adj[idx] & x).any()
                if not (adj[idx] & x).any():
                    x[idx] = 1
                    
        current_score = x.sum()
        best_x = x.copy()
        best_score = current_score
        
        # Simulated Annealing
        # Objective: Maximize sum(x)
        # Penalty for conflicts: We want to avoid states with conflicts.
        # Energy E = -sum(x) + lambda * (number of conflicts in current set)
        # Actually, simpler: 
        # Let's define Energy as:
        # E = -current_size + penalty_factor * num_conflicts
        # We want to minimize E.
        # If we flip a bit, we change size by +/- 1.
        # We change conflicts by some delta.
        
        # To make SA work well, we need a fast way to calculate the change in conflicts.
        # Let's stick to a simpler approach:
        # Randomly flip a bit. Check if the new state is valid (sum-free).
        # If valid and size increased, accept.
        # If valid and size same or decreased, accept with prob exp(-deltaE/T).
        # If invalid, we might still accept it if we use a penalty, but strictly speaking, 
        # the problem requires the final result to be sum-free.
        # It is safer to restrict the search space to valid configurations or use a heavy penalty.
        # Given the graph is sparse, we can afford to check validity of neighbors.
        
        # Better SA formulation for MIS:
        # State: binary vector x.
        # Score: sum(x).
        # Constraint: No edges in induced subgraph.
        # If we flip x[i] from 0 to 1:
        #   It is valid only if no neighbor j with x[j]=1 exists.
        #   i.e., (adj[i] & x).sum() == 0.
        #   If valid, score increases by 1. Always accept.
        # If we flip x[i] from 1 to 0:
        #   It is always valid (removing a node never creates a conflict).
        #   Score decreases by 1. Accept with prob exp(1/T).
        
        # This is a "repair" style SA or just hill climbing with restarts?
        # Standard SA allows moving to worse states. Here, "worse" means smaller set.
        # But we can't move to an *invalid* state if we want to keep the invariant "current state is valid".
        # If we allow invalid states, we need a penalty.
        # Let's try: Allow invalid states, but penalize heavily.
        # Energy E = -sum(x) + P * num_conflicts
        # We want to minimize E.
        
        # Let's use the penalty approach as it allows more exploration.
        P = 10.0 # Penalty per conflict
        
        def get_energy(x):
            # x is numpy array of 0/1
            # sum(x) is the size
            # num_conflicts: count of edges (i,j) where x[i]=1 and x[j]=1
            # Since adj is symmetric, sum(adj * x[:, None] * x[None, :]) / 2
            # Or simply: for each i where x[i]=1, count neighbors where x[j]=1.
            # conflicts = (adj * x).sum() / 2 ? No.
            # adj[i,j] is 1 if conflict.
            # If x[i]=1 and x[j]=1, and adj[i,j]=1, that's 1 conflict.
            # Total conflicts = sum_{i<j} adj[i,j] x[i] x[j]
            # = 0.5 * sum_{i,j} adj[i,j] x[i] x[j]
            # = 0.5 * (x @ adj @ x)
            conflicts = 0.5 * (x @ adj @ x)
            return -x.sum() + P * conflicts

        current_energy = get_energy(x)
        
        T = 5.0
        T_min = 0.01
        cooling = 0.999 # Adjust based on steps
        
        # Estimate steps
        # We have time_budget. 
        # Each step involves a few numpy ops.
        # Let's run until time runs out.
        
        rng = np.random.default_rng(42)
        
        # Precompute indices for faster access?
        # We will pick a random index i.
        # Flip x[i].
        
        # To speed up energy calculation, we can update incrementally.
        # But numpy is fast enough for n=240.
        
        while time.time() - start_time < start_time + time_budget:
            # Check deadline
            if time.time() > DEADLINE:
                break
                
            i = rng.integers(n)
            
            # Flip x[i]
            x[i] = 1 - x[i]
            
            new_energy = get_energy(x)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                # Accept
                current_energy = new_energy
                if x.sum() > best_score:
                    best_score = x.sum()
                    best_x = x.copy()
            else:
                # Reject
                x[i] = 1 - x[i]
                
            T *= cooling
            if T < T_min:
                T = T_min
                # Optional: Restart? Or just keep going?
                # With T_min, it becomes hill climbing.
                # We can restart from best_x occasionally.
                if rng.random() < 0.01:
                    x = best_x.copy()
                    current_energy = get_energy(x)
                    T = 1.0 # Reheat slightly?
        
        # Final check: ensure best_x is valid
        # It should be, but let's verify.
        # If best_x has conflicts, we need to clean it up.
        # Greedy cleanup: remove nodes involved in conflicts.
        s = set(np.where(best_x == 1)[0])
        changed = True
        while changed:
            changed = False
            for i in list(s):
                # Check if i conflicts with any other in s
                # neighbors = np.where(adj[i] == 1)[0]
                # if any neighbor in s:
                if (adj[i] & best_x).sum() > 0:
                    # Remove i
                    s.remove(i)
                    best_x[i] = 0
                    changed = True
                    break
        
        return [values[i] for i in s]

    def greedy_fallback(values):
        n = len(values)
        v_set = set(values)
        # Simple greedy: sort by degree, pick min degree, remove neighbors
        # Build adj quickly
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            vi = values[i]
            for j in range(i + 1, n):
                vj = values[j]
                if (vi + vj) in v_set or 2 * vi == vj or 2 * vj == vi:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        degrees = adj.sum(axis=1)
        x = np.zeros(n, dtype=np.int8)
        order = np.argsort(degrees)
        for idx in order:
            if x[idx] == 0 and not (adj[idx] & x).any():
                x[idx] = 1
        return [values[i] for i in np.where(x == 1)[0]]

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    # Allocate time
    # SA is stochastic and benefits from time.
    # ILP was exact but potentially slow for larger n (though sparse).
    # SA is fast per step.
    
    total_budget = DEADLINE - (time.time() - start_time)
    # Distribute time roughly equally, maybe slightly more to larger instances?
    # n=60, 120, 240.
    # Complexity of step is O(n) for energy update if done smartly, O(n^2) if naive.
    # My get_energy is O(n^2) due to matrix multiplication.
    # For n=240, 240^2 = 57600 ops. Very fast in numpy.
    # So we can run many steps.
    
    # Let's give 30% to first, 30% to second, 40% to third.
    budgets = [total_budget * 0.3, total_budget * 0.3, total_budget * 0.4]
    
    for i, (size, seed) in enumerate(instances):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 2:
             vals = instance(size, seed)
             res = greedy_fallback(vals)
             if not is_sum_free(res):
                 res = [x for x in vals if x % 2 != 0]
             results.append(res)
             continue
        
        # Use allocated budget, but don't exceed remaining
        budget = min(budgets[i], remaining - 1.0)
        if budget < 0.5:
             budget = 0.5
             
        vals = instance(size, seed)
        
        # Run SA
        res = solve_instance_sa(vals, budget)
        
        # Verify
        if not is_sum_free(res):
             # Cleanup
             s = set(res)
             changed = True
             while changed:
                 changed = False
                 for a in list(s):
                     # Check if a conflicts with any b in s
                     # Conflict if a+b in s or 2a in s
                     for b in s:
                         if a != b and (a+b) in s:
                             s.remove(a)
                             changed = True
                             break
                         if 2*a == b:
                             s.remove(a)
                             changed = True
                             break
                     if changed:
                         break
             res = list(s)
             
             if not is_sum_free(res):
                 res = greedy_fallback(vals)
                 if not is_sum_free(res):
                     res = [x for x in vals if x % 2 != 0]
        
        results.append(res)
        
    return results

# Keep the dummy function to satisfy signature if needed, though not used in main logic
def greedy_mis(values):
    pass
