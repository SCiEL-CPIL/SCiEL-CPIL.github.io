import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        """Verify a coloring is valid (no red K4, no blue K5)."""
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def build_structures(target_n):
        """Precompute all structural mappings for a given n."""
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        all_quads = list(combinations(range(target_n), 4))
        quad_triples_idx = []
        for a, b, c, d in all_quads:
            quad_triples_idx.append([
                triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]
            ])
        num_quads = len(quad_triples_idx)

        all_quints = list(combinations(range(target_n), 5))
        quint_triples_idx = []
        for q5 in all_quints:
            quint_triples_idx.append([triple_to_idx[t] for t in combinations(q5, 3)])
        num_quints = len(quint_triples_idx)

        # Inverse mappings: for each triple, which quads/quints contain it
        triple_in_quads = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quad_triples_idx):
            for pos, tidx in enumerate(qt):
                triple_in_quads[tidx].append((qi, pos))

        triple_in_quints = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quint_triples_idx):
            for pos, tidx in enumerate(qt):
                triple_in_quints[tidx].append((qi, pos))

        return {
            'triple_list': triple_list,
            'num_triples': num_triples,
            'quad_triples_idx': quad_triples_idx,
            'quint_triples_idx': quint_triples_idx,
            'num_quads': num_quads,
            'num_quints': num_quints,
            'triple_in_quads': triple_in_quads,
            'triple_in_quints': triple_in_quints,
        }

    def solve_csp(target_n, structs, deadline, max_restarts=10):
        """
        Solve using Constraint Satisfaction with Arc Consistency and Backtracking.
        """
        num_triples = structs['num_triples']
        quad_triples_idx = structs['quad_triples_idx']
        quint_triples_idx = structs['quint_triples_idx']
        triple_in_quads = structs['triple_in_quads']
        triple_in_quints = structs['triple_in_quints']
        
        # Precompute neighbors for propagation
        # For each triple, we need to know which quads and quints it belongs to.
        # We also need to be able to quickly check the status of other triples in those structures.
        
        # State: 
        # colors: array of -1 (unassigned), 0 (red), 1 (blue)
        # domains: set of possible values for each unassigned variable (implicit via colors)
        
        def propagate(colors, queue):
            """
            AC-3 style propagation.
            If a constraint becomes violated or forces a value, we update and add neighbors to queue.
            Returns False if contradiction (empty domain).
            """
            while queue:
                t_idx = queue.pop()
                if colors[t_idx] != -1:
                    continue
                
                # Check if t_idx is forced by any quad
                forced_val = -1
                is_forced = False
                
                # Check Quads (Red K4 constraint: not all 4 red)
                # If 3 are red, 4th must be blue.
                # If 3 are blue, no restriction on 4th from Red K4 perspective alone? 
                # Wait, Red K4 is "all 4 red". So if we have 3 red, the 4th MUST be blue.
                # If we have 1 red, 2 blue, 1 unassigned? No immediate force.
                
                # Check Quads for Red K4
                for (qi, pos) in triple_in_quads[t_idx]:
                    qt = quad_triples_idx[qi]
                    others = [qt[p] for p in range(4) if p != pos]
                    red_count = 0
                    blue_count = 0
                    unassigned = 0
                    for o in others:
                        if colors[o] == 0: red_count += 1
                        elif colors[o] == 1: blue_count += 1
                        else: unassigned += 1
                    
                    if red_count == 3:
                        # 3 reds, t_idx must be blue
                        if forced_val == -1:
                            forced_val = 1
                            is_forced = True
                        elif forced_val != 1:
                            return False
                    # Note: If red_count < 3, t_idx can be red or blue (unless forced by other constraints)
                    # Actually, if red_count == 3, t_idx CANNOT be red. So domain shrinks to {1}.
                    # If red_count < 3, t_idx CAN be red.
                    
                # Check Quints (Blue K5 constraint: not all 10 blue)
                # If 9 are blue, 10th must be red.
                for (qi, pos) in triple_in_quints[t_idx]:
                    qt = quint_triples_idx[qi]
                    others = [qt[p] for p in range(10) if p != pos]
                    blue_count = 0
                    red_count = 0
                    unassigned = 0
                    for o in others:
                        if colors[o] == 1: blue_count += 1
                        elif colors[o] == 0: red_count += 1
                        else: unassigned += 1
                    
                    if blue_count == 9:
                        # 9 blues, t_idx must be red
                        if forced_val == -1:
                            forced_val = 0
                            is_forced = True
                        elif forced_val != 0:
                            return False
                    # If blue_count < 9, t_idx CAN be blue.

                # Determine if domain is empty or forced
                # If is_forced and forced_val is set, check if it's consistent with current "allowed" values?
                # Actually, the logic above identifies if t_idx is forced to a specific value by ANY constraint.
                # If multiple constraints force different values -> Contradiction.
                # If at least one constraint forces a value, t_idx is forced to that value.
                
                if is_forced:
                    # Check if this value is already excluded by other logic?
                    # In this simple model, we just assign it.
                    # But what if it's not forced? We don't prune domain here explicitly in AC-3 for binary vars 
                    # unless we track domains. For binary vars, "not forced to 0" means "can be 0 or 1".
                    # "Forced to 1" means "must be 1".
                    # If we have "must be 1" and "must be 0" -> Fail.
                    # If we have "must be 1", we assign 1.
                    
                    colors[t_idx] = forced_val
                    
                    # Add neighbors to queue
                    # Neighbors in quads
                    for (qi, pos) in triple_in_quads[t_idx]:
                        qt = quad_triples_idx[qi]
                        for p in range(4):
                            if p != pos:
                                queue.append(qt[p])
                    # Neighbors in quints
                    for (qi, pos) in triple_in_quints[t_idx]:
                        qt = quint_triples_idx[qi]
                        for p in range(10):
                            if p != pos:
                                queue.append(qt[p])
                
                # If not forced, we don't change colors[t_idx], but we might have pruned its domain?
                # For binary CSP, if no constraint forces it to 0 or 1, it remains unassigned.
                # However, we might be able to prune if a constraint says "t_idx cannot be 0".
                # "Cannot be 0" happens if setting it to 0 causes a violation? No, that's forward checking.
                # AC-3 for "Not All Red" (Red K4):
                # Constraint: NOT (C1=0 AND C2=0 AND C3=0 AND C4=0)
                # If C1,C2,C3 are 0, C4 cannot be 0. So C4 must be 1.
                # This is exactly what the "red_count == 3" check does.
                
                # What about "Cannot be 1"?
                # Blue K5: NOT (all 10 are 1).
                # If 9 are 1, the 10th cannot be 1. So it must be 0.
                # This is exactly what the "blue_count == 9" check does.
                
                # Are there other prunings?
                # If I set t_idx to 0, does it force a neighbor to be impossible? That's forward checking, not AC-3.
                # AC-3 only propagates immediate consequences.
                
            return True

        def backtrack_search(colors, depth, deadline):
            """
            Backtracking search with MRV.
            """
            if time.time() > deadline:
                return None
                
            # Find unassigned variables
            unassigned = [i for i in range(num_triples) if colors[i] == -1]
            
            if not unassigned:
                # Check if solution is valid (should be, given propagation)
                # Verify no Red K4
                for qt in quad_triples_idx:
                    if all(colors[t] == 0 for t in qt):
                        return None
                # Verify no Blue K5
                for qt in quint_triples_idx:
                    if all(colors[t] == 1 for t in qt):
                        return None
                return colors.copy()
            
            # MRV Heuristic: Pick variable with fewest legal values?
            # In binary, it's 1 or 2.
            # We can estimate "tightness" by counting how many constraints are "almost satisfied"
            # i.e., how many quads have 3 reds (forcing blue) or quints have 9 blues (forcing red).
            # Variables involved in such "almost forced" situations are more critical.
            
            # Simple MRV: Just pick the first one? Or random?
            # Let's try picking one involved in the most "tight" constraints.
            # For each unassigned var, count "pressure".
            # Pressure = number of quads where it's the only non-red? (i.e. 3 reds, 1 unassigned)
            #          + number of quints where it's the only non-blue? (i.e. 9 blues, 1 unassigned)
            
            # This calculation is expensive if done every node.
            # Let's just use random or fixed order for now, or a simple degree heuristic.
            # Degree: number of neighbors.
            
            # Let's try random selection from unassigned to avoid bias, 
            # but prioritize those that are "forced" by propagation? 
            # Wait, propagation already handles forced values. 
            # So all unassigned vars are NOT forced.
            
            # Let's pick the variable with the highest "constraint degree" 
            # (number of quads + quints it belongs to).
            
            # Precompute degrees?
            # triple_in_quads[t] length + triple_in_quints[t] length.
            
            # For speed, let's just pick the first unassigned one? 
            # Or random? Random might help escape local minima.
            # Let's use a simple heuristic: pick the one with max degree.
            
            # To avoid O(N) scan every time, we can just pick random unassigned.
            # Given N is small (max ~ 34 choose 3 = 5984), scanning is fine.
            
            # Let's try: Pick variable with highest "potential to fail".
            # If a variable is in a quad with 3 reds, it's forced (handled by prop).
            # So all unassigned are "free".
            
            # Let's just pick the first unassigned index.
            var = unassigned[0]
            
            # Try values 0 and 1
            # Order: Try value that leads to more propagation? 
            # Or random?
            # Let's try 0 then 1.
            
            for val in [0, 1]:
                new_colors = colors.copy()
                new_colors[var] = val
                
                # Propagate
                queue = []
                # Add neighbors of var
                for (qi, pos) in triple_in_quads[var]:
                    qt = quad_triples_idx[qi]
                    for p in range(4):
                        if p != pos:
                            queue.append(qt[p])
                for (qi, pos) in triple_in_quints[var]:
                    qt = quint_triples_idx[qi]
                    for p in range(10):
                        if p != pos:
                            queue.append(qt[p])
                
                if propagate(new_colors, queue):
                    result = backtrack_search(new_colors, depth + 1, deadline)
                    if result is not None:
                        return result
                        
            return None

        # Initial randomization / Restart strategy
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            # Initialize with random colors or all -1?
            # All -1 is pure CSP.
            # Random partial assignment might help.
            # Let's start with all -1.
            
            colors = np.full(num_triples, -1, dtype=np.int8)
            
            # Randomly assign some variables to break symmetry?
            # Or just let CSP handle it.
            
            # To speed up, we can pre-assign some variables randomly.
            # Let's assign 10% of variables randomly.
            num_pre_assign = num_triples // 10
            indices = random.sample(range(num_triples), num_pre_assign)
            for idx in indices:
                colors[idx] = random.randint(0, 1)
            
            # Propagate initial assignments
            queue = []
            for idx in indices:
                for (qi, pos) in triple_in_quads[idx]:
                    qt = quad_triples_idx[qi]
                    for p in range(4):
                        if p != pos:
                            queue.append(qt[p])
                for (qi, pos) in triple_in_quints[idx]:
                    qt = quint_triples_idx[qi]
                    for p in range(10):
                        if p != pos:
                            queue.append(qt[p])
            
            if propagate(colors, queue):
                result = backtrack_search(colors, 0, deadline)
                if result is not None:
                    return result.tolist()
            
        return None

    # Main loop: try increasing n with CSP solver
    for target_n in range(10, 35):
        if time.time() > deadline - 10:
            break

        structs = build_structures(target_n)
        
        result = solve_csp(target_n, structs, deadline, max_restarts=20)
        
        if result is not None:
            if is_valid(target_n, result):
                best_n = target_n
                best_colors = result
            else:
                break
        else:
            # If CSP fails, maybe try a simpler heuristic? 
            # Or just break if n is large enough?
            # Given the time, if CSP fails, it's likely hard.
            # We can try one quick local search restart?
            # No, let's stick to the structural change. 
            # If CSP fails for n, it might fail for n+1 too? Not necessarily.
            # But if we run out of time, we break.
            if target_n >= 25:
                 # If we've tried up to 25 and failed, and time is running out, stop.
                 # Actually, let's just continue if time permits.
                 pass

    return {'n': best_n, 'colors': best_colors}
