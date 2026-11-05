import numpy as np
import time
from itertools import combinations
import sys

def construct():
    start_time = time.time()
    # Safety margin: stop at 145s to allow for return overhead
    deadline = start_time + 145.0
    
    def build_structure(n):
        """
        Builds the hypergraph structure for n vertices.
        Returns:
        - num_triples
        - k4s: list of tuples of triple indices for each K4
        - k5s: list of tuples of triple indices for each K5
        - k4s_containing: for each triple, list of K4 indices containing it
        - k5s_containing: for each triple, list of K5 indices containing it
        """
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = []
        k4s_containing = [[] for _ in range(num_triples)]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            idx_tuple = (t1, t2, t3, t4)
            k4s.append(idx_tuple)
            for t in idx_tuple:
                k4s_containing[t].append(len(k4s) - 1)
                
        k5s = []
        k5s_containing = [[] for _ in range(num_triples)]
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            idx_tuple = tuple(idxs)
            k5s.append(idx_tuple)
            for t in idx_tuple:
                k5s_containing[t].append(len(k5s) - 1)
                
        return num_triples, k4s, k5s, k4s_containing, k5s_containing

    def local_search(n, time_limit, seed=0):
        """
        Runs local search to find a valid coloring.
        Returns (colors, is_valid)
        """
        np.random.seed(seed)
        num_triples, k4s, k5s, k4s_containing, k5s_containing = build_structure(n)
        
        # Initialize colors randomly
        colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
        
        # Initialize counts
        # red_count[i] = number of red triples in K4 i
        # blue_count[i] = number of blue triples in K5 i
        red_counts = np.zeros(len(k4s), dtype=np.int16)
        blue_counts = np.zeros(len(k5s), dtype=np.int16)
        
        # Calculate initial counts
        for i, k4 in enumerate(k4s):
            # k4 is a tuple of triple indices
            # colors are 0 (red) or 1 (blue)
            # We want count of 0s
            # sum(1 - colors[k4])
            red_counts[i] = np.sum(1 - colors[list(k4)])
            
        for i, k5 in enumerate(k5s):
            # We want count of 1s
            blue_counts[i] = np.sum(colors[list(k5)])
            
        # Calculate initial violations
        # Violation if red_count == 4 or blue_count == 10
        violations = np.sum(red_counts == 4) + np.sum(blue_counts == 10)
        
        best_violations = violations
        best_colors = colors.copy()
        
        temp = 1.0
        cool_rate = 0.9999
        min_temp = 0.01
        tabu_list = {}
        tabu_tenure = 100
        
        step = 0
        
        while time.time() < time_limit:
            if violations == 0:
                return colors, True
                
            # Pick a random triple
            t = np.random.randint(num_triples)
            
            # Check tabu
            if t in tabu_list and step - tabu_list[t] < tabu_tenure:
                continue
                
            old_color = colors[t]
            new_color = 1 - old_color
            
            delta_violations = 0
            
            # Update K4s containing t
            for k4_idx in k4s_containing[t]:
                old_rc = red_counts[k4_idx]
                if old_color == 0: # Was red, becoming blue
                    # Red count decreases by 1
                    new_rc = old_rc - 1
                    if old_rc == 4:
                        delta_violations -= 1 # Violation removed
                    # If new_rc == 4 is impossible since it was 4 and decreased
                else: # Was blue, becoming red
                    # Red count increases by 1
                    new_rc = old_rc + 1
                    if new_rc == 4:
                        delta_violations += 1 # Violation created
                    # If old_rc == 4 is impossible since it was < 4 (blue was 1)
                
                red_counts[k4_idx] = new_rc
                
            # Update K5s containing t
            for k5_idx in k5s_containing[t]:
                old_bc = blue_counts[k5_idx]
                if old_color == 1: # Was blue, becoming red
                    # Blue count decreases by 1
                    new_bc = old_bc - 1
                    if old_bc == 10:
                        delta_violations -= 1 # Violation removed
                else: # Was red, becoming blue
                    # Blue count increases by 1
                    new_bc = old_bc + 1
                    if new_bc == 10:
                        delta_violations += 1 # Violation created
                
                blue_counts[k5_idx] = new_bc
            
            new_violations = violations + delta_violations
            
            # Acceptance criterion
            accepted = False
            if new_violations < violations:
                accepted = True
            elif new_violations == violations:
                accepted = np.random.rand() < 0.5
            else:
                if temp > min_temp:
                    # Accept with probability exp(-delta / temp)
                    # delta is positive
                    prob = np.exp(-new_violations / temp)
                    accepted = np.random.rand() < prob
            
            if accepted:
                colors[t] = new_color
                violations = new_violations
                tabu_list[t] = step
                
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors.copy()
                    if best_violations == 0:
                        return best_colors, True
                
            else:
                # Revert counts if not accepted? 
                # Actually, we updated the counts in place. We must revert them if not accepted.
                # This is inefficient. Better to calculate delta first, then update.
                # Let's refactor to calculate delta first.
                pass
            
            # Cooling
            temp *= cool_rate
            step += 1
            
            # If stuck, reheat?
            if step % 10000 == 0 and temp < min_temp:
                temp = 0.5
        
        return best_colors, (best_violations == 0)

    def local_search_optimized(n, time_limit, seed=0):
        """
        Optimized version that calculates delta before updating state.
        """
        np.random.seed(seed)
        num_triples, k4s, k5s, k4s_containing, k5s_containing = build_structure(n)
        
        # Heuristic Initialization: Try a few patterns
        # Pattern 1: Random
        # Pattern 2: Based on sum mod 3
        # Let's stick to random for now, but maybe try a structured one first?
        # A good structured init: c(a,b,c) = 0 if (a+b+c)%2==0 else 1
        # But this might have many violations. Random is safer for diversity.
        
        colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
        
        red_counts = np.zeros(len(k4s), dtype=np.int16)
        blue_counts = np.zeros(len(k5s), dtype=np.int16)
        
        for i, k4 in enumerate(k4s):
            red_counts[i] = np.sum(1 - colors[list(k4)])
        for i, k5 in enumerate(k5s):
            blue_counts[i] = np.sum(colors[list(k5)])
        
        violations = np.sum(red_counts == 4) + np.sum(blue_counts == 10)
        
        best_violations = violations
        best_colors = colors.copy()
        
        temp = 1.0
        cool_rate = 0.99995
        min_temp = 0.01
        tabu_list = {}
        tabu_tenure = 50
        step = 0
        last_improve = time.time()
        
        while time.time() < time_limit:
            if violations == 0:
                return colors, True
            
            # Pick a random triple
            t = np.random.randint(num_triples)
            
            # Check tabu
            if t in tabu_list and step - tabu_list[t] < tabu_tenure:
                continue
                
            old_color = colors[t]
            new_color = 1 - old_color
            
            delta_violations = 0
            
            # Calculate delta for K4s
            for k4_idx in k4s_containing[t]:
                old_rc = red_counts[k4_idx]
                if old_color == 0: # Red -> Blue
                    # rc decreases
                    if old_rc == 4:
                        delta_violations -= 1
                else: # Blue -> Red
                    # rc increases
                    if old_rc == 3: # Becomes 4
                        delta_violations += 1
                        
            # Calculate delta for K5s
            for k5_idx in k5s_containing[t]:
                old_bc = blue_counts[k5_idx]
                if old_color == 1: # Blue -> Red
                    # bc decreases
                    if old_bc == 10:
                        delta_violations -= 1
                else: # Red -> Blue
                    # bc increases
                    if old_bc == 9: # Becomes 10
                        delta_violations += 1
                        
            new_violations = violations + delta_violations
            
            # Acceptance
            accepted = False
            if new_violations < violations:
                accepted = True
            elif new_violations == violations:
                accepted = np.random.rand() < 0.3
            else:
                if temp > min_temp:
                    prob = np.exp(-new_violations / temp)
                    accepted = np.random.rand() < prob
            
            if accepted:
                # Apply change
                colors[t] = new_color
                violations = new_violations
                tabu_list[t] = step
                
                # Update counts
                for k4_idx in k4s_containing[t]:
                    if old_color == 0:
                        red_counts[k4_idx] -= 1
                    else:
                        red_counts[k4_idx] += 1
                        
                for k5_idx in k5s_containing[t]:
                    if old_color == 1:
                        blue_counts[k5_idx] -= 1
                    else:
                        blue_counts[k5_idx] += 1
                
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors.copy()
                    last_improve = time.time()
                    if best_violations == 0:
                        return best_colors, True
            
            temp *= cool_rate
            step += 1
            
            # Reheat if stuck
            if time.time() - last_improve > 5.0:
                temp = 0.5
                # Random restart of some variables?
                # Or just reheat.
                
        return best_colors, (best_violations == 0)

    best_n = 9
    # Fallback trivial coloring for n=9
    best_result_colors = [0 if (a+b+c)%3==0 else 1 for a,b,c in combinations(range(9),3)]
    
    # Try to find a solution for high n
    target_ns = [34, 33, 32, 31, 30]
    
    # Allocate time
    # 145s total.
    # Give 60s to n=34.
    # If fails, give 30s to n=33.
    # Etc.
    
    time_budgets = {
        34: 60.0,
        33: 30.0,
        32: 20.0,
        31: 15.0,
        30: 10.0
    }
    
    for n in target_ns:
        if time.time() > deadline - 5:
            break
            
        budget = time_budgets.get(n, 10.0)
        # Ensure we don't exceed deadline
        end_time = min(time.time() + budget, deadline - 2)
        
        if time.time() >= end_time:
            continue
            
        # Try multiple seeds
        seeds = [0, 1, 42, 123, 999]
        
        for seed in seeds:
            if time.time() >= end_time:
                break
                
            # Give each seed a portion of the budget
            seed_budget = (end_time - time.time()) / (len(seeds) - seeds.index(seed))
            if seed_budget < 1.0:
                break
                
            try:
                colors, is_valid = local_search_optimized(n, time.time() + seed_budget, seed=seed)
                if is_valid:
                    best_n = n
                    best_result_colors = colors.tolist()
                    if n == 34:
                        return {'n': best_n, 'colors': best_result_colors}
                    break # Found valid for this n, try higher? No, we iterate high to low. So break inner loop.
            except Exception as e:
                pass

    return {'n': best_n, 'colors': best_result_colors}
