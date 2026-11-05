"""Weighted 2-SAT / WalkSAT approach for large sum-free subsets."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def solve_instance(elements, seed_val, time_budget):
        n = len(elements)
        rng = random.Random(seed_val)
        elem_to_idx = {v: i for i, v in enumerate(elements)}
        elem_set = set(elements)
        
        # Build clauses (triples of indices)
        # Clause: (not a or not b or not c)
        clause_set = set()
        clauses = []
        var_to_clauses = [[] for _ in range(n)]
        
        for c_idx, c in enumerate(elements):
            for a_idx, a in enumerate(elements):
                b = c - a
                if b in elem_set:
                    b_idx = elem_to_idx[b]
                    tri = tuple(sorted([a_idx, b_idx, c_idx]))
                    if tri not in clause_set:
                        clause_set.add(tri)
                        clauses.append(tri)
                        c_idx_in_list = len(clauses) - 1
                        for idx in tri:
                            var_to_clauses[idx].append(c_idx_in_list)
        
        # WalkSAT with incremental violation tracking
        # State:
        # assign: list of bools
        # clause_true_counts: list of ints (0-3)
        # violated_clauses: set of clause indices
        
        def init_state(assign):
            clause_true_counts = [0] * len(clauses)
            violated_clauses = set()
            for c_idx, clause in enumerate(clauses):
                cnt = 0
                for idx in clause:
                    if assign[idx]:
                        cnt += 1
                clause_true_counts[c_idx] = cnt
                if cnt == 3:
                    violated_clauses.add(c_idx)
            return clause_true_counts, violated_clauses

        def flip_var(idx, assign, clause_true_counts, violated_clauses):
            old_val = assign[idx]
            assign[idx] = not old_val
            
            for c_idx in var_to_clauses[idx]:
                old_cnt = clause_true_counts[c_idx]
                if assign[idx]:
                    new_cnt = old_cnt + 1
                else:
                    new_cnt = old_cnt - 1
                
                clause_true_counts[c_idx] = new_cnt
                
                # Update violated set
                if old_cnt == 2 and new_cnt == 3:
                    violated_clauses.add(c_idx)
                elif old_cnt == 3 and new_cnt == 2:
                    violated_clauses.remove(c_idx)
                # Cases: 1->2, 0->1, 2->1, 1->0 do not change violation status (not 3)
                # 3->4 impossible.
            
            return violated_clauses

        best_size = 0
        best_assign = None
        
        max_restarts = 100
        steps_per_restart = 10000
        
        for r in range(max_restarts):
            if time.time() - start_time > DEADLINE - 2:
                break
                
            # Init: Random assignment, 50% True
            assign = [rng.random() < 0.5 for _ in range(n)]
            clause_true_counts, violated_clauses = init_state(assign)
            cur_size = sum(assign)
            
            if not violated_clauses and cur_size > best_size:
                best_size = cur_size
                best_assign = assign[:]
            
            for step in range(steps_per_restart):
                if time.time() - start_time > DEADLINE - 2:
                    break
                    
                if not violated_clauses:
                    # Valid state. Try to increase size.
                    # Find False variables
                    # Optimization: Keep a list of False vars? 
                    # For now, scan. n=240 is small.
                    false_vars = [i for i in range(n) if not assign[i]]
                    if not false_vars:
                        break
                    idx = rng.choice(false_vars)
                    
                    # Try flip to True
                    old_viol_len = len(violated_clauses)
                    violated_clauses = flip_var(idx, assign, clause_true_counts, violated_clauses)
                    
                    if len(violated_clauses) == 0:
                        # Accepted
                        cur_size += 1
                        if cur_size > best_size:
                            best_size = cur_size
                            best_assign = assign[:]
                    else:
                        # Rejected
                        violated_clauses = flip_var(idx, assign, clause_true_counts, violated_clauses)
                        
                else:
                    # Invalid state. Pick random violated clause.
                    c_idx = rng.choice(list(violated_clauses))
                    clause = clauses[c_idx]
                    
                    # Pick a variable in the clause that is True to flip to False
                    # All vars in a violated clause are True.
                    # We want to flip one to False.
                    # Heuristic: Flip the one that appears in the most violated clauses?
                    # For now, random.
                    idx = rng.choice(clause)
                    
                    violated_clauses = flip_var(idx, assign, clause_true_counts, violated_clauses)
                    cur_size -= 1 # We flipped True to False
                    
                    # If we fixed it, we are in valid state. 
                    # If we created new violations, we are still invalid.
                    
        if best_assign is None:
            # Fallback: Greedy
            S = []
            S_set = set()
            for x in elements:
                can_add = True
                for a in S:
                    if x - a in S_set or a + x in S_set:
                        can_add = False
                        break
                if 2*x in S_set:
                    can_add = False
                if can_add:
                    S.append(x)
                    S_set.add(x)
            return S

        return [elements[i] for i in range(n) if best_assign[i]]

    # Generate instances
    instances = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        instances.append(instance(size, seed))
    
    results = []
    for i, elems in enumerate(instances):
        if time.time() - start_time > DEADLINE:
            results.append([x for x in elems if x % 2 != 0])
            continue
        
        remaining = DEADLINE - (time.time() - start_time)
        if i == 0:
            time_budget = remaining * 0.2
        elif i == 1:
            time_budget = remaining * 0.3
        else:
            time_budget = remaining * 0.5
        
        results.append(solve_instance(elems, seed_val=801+i, time_budget=time_budget))
        
    return results
# EVOLVE-BLOCK-END
