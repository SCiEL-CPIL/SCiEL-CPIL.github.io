import time
import sys
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, use 145s to be safe
    deadline = start_time + 145.0

    def verify(colors, n):
        """
        Verify if a coloring is valid.
        colors: list of 0/1 for triples in lex order.
        Returns True if valid (no red K4, no blue K5).
        """
        triple_index = {}
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triple_index[(a,b,c)] = idx
                    idx += 1
        
        # Check Red K4
        for quad in combinations(range(n), 4):
            all_red = True
            for triple in combinations(quad, 3):
                if colors[triple_index[triple]] != 0:
                    all_red = False
                    break
            if all_red:
                return False
        
        # Check Blue K5
        for quint in combinations(range(n), 5):
            all_blue = True
            for triple in combinations(quint, 3):
                if colors[triple_index[triple]] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
                
        return True

    def solve_backtracking(n, time_limit):
        """
        Deterministic backtracking solver with forward checking.
        """
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # Precompute for each triple, the 4-sets and 5-sets it belongs to
        # This allows fast checking of constraints when a triple is assigned.
        
        # For Red K4: If we assign Red to triple t, we need to ensure no 4-set 
        # containing t becomes all-Red.
        # A 4-set contains t if the 4th vertex is in the complement of t.
        
        # For Blue K5: If we assign Blue to triple t, we need to ensure no 5-set
        # containing t becomes all-Blue.
        
        # To make this efficient, we maintain:
        # red_counts[quad]: count of red triples in this 4-set. If it reaches 4, violation.
        # blue_counts[quint]: count of blue triples in this 5-set. If it reaches 10, violation.
        
        # However, maintaining counts for ALL quads/quints might be memory heavy?
        # n=34: C(34,4) ~ 46k. C(34,5) ~ 278k. 
        # Arrays of size 46k and 278k are fine.
        
        # But we need to map from triple index to the specific quads/quints it affects.
        
        # Let's build the incidence structure.
        # quads = list of all 4-sets
        # quints = list of all 5-sets
        
        # This construction takes time.
        # C(34,4) is small. C(34,5) is moderate.
        
        # Actually, a smarter approach for backtracking:
        # When assigning color to triple t:
        # 1. Check if any 4-set containing t would become all-Red (if color=0).
        #    This requires knowing the colors of the other 3 triples in that 4-set.
        #    If all other 3 are Red (0), then assigning Red to t creates a Red K4.
        # 2. Check if any 5-set containing t would become all-Blue (if color=1).
        #    This requires knowing the colors of the other 9 triples in that 5-set.
        #    If all other 9 are Blue (1), then assigning Blue to t creates a Blue K5.
        
        # We can precompute for each triple t:
        # red_conflicts[t] = list of 4-sets containing t.
        # blue_conflicts[t] = list of 5-sets containing t.
        
        # Then, during assignment, we check these lists.
        # For speed, we can maintain counts of "remaining unassigned" or "assigned to opposite color"?
        # No, just checking the specific neighbors is fast enough if we do it carefully.
        
        # Let's precompute the neighbors.
        
        quads = list(combinations(range(n), 4))
        quints = list(combinations(range(n), 5))
        
        quad_index = {q: i for i, q in enumerate(quads)}
        quint_index = {q: i for i, q in enumerate(quints)}
        
        # For each triple, store the quads and quints it belongs to.
        # And for each quad/quint, store the triples in it.
        
        triple_to_quads = [[] for _ in range(num_triples)]
        triple_to_quints = [[] for _ in range(num_triples)]
        
        quad_triples = [[] for _ in range(len(quads))]
        quint_triples = [[] for _ in range(len(quints))]
        
        for q in quads:
            qi = quad_index[q]
            for t in combinations(q, 3):
                ti = triple_index[t]
                triple_to_quads[ti].append(qi)
                quad_triples[qi].append(ti)
                
        for q in quints:
            qi = quint_index[q]
            for t in combinations(q, 3):
                ti = triple_index[t]
                triple_to_quints[ti].append(qi)
                quint_triples[qi].append(ti)
        
        # State
        colors = [-1] * num_triples  # -1 unassigned, 0 Red, 1 Blue
        
        # We need to track for each quad: how many are Red?
        # If we assign Red to t, red_count[quad]++.
        # If red_count[quad] == 4, violation.
        # Similarly for Blue.
        
        red_count = [0] * len(quads)
        blue_count = [0] * len(quints)
        
        # Order of assignment:
        # We can try to assign triples in an order that maximizes constraints.
        # Or just lexicographic.
        # Lexicographic is simple.
        
        # To help backtracking, we can use a heuristic:
        # Always try the color that is "less constrained" or "more likely"?
        # Or just try Red then Blue.
        
        # Let's use a recursive function with a time check.
        
        def bt(idx):
            if time.time() > time_limit:
                return False
            if idx == num_triples:
                return True
                
            t = triples[idx]
            ti = idx # since we iterate in lex order, idx is the triple index
            
            # Try Red (0)
            can_red = True
            # Check red constraints: any quad containing t becomes all red?
            for qi in triple_to_quads[ti]:
                if red_count[qi] == 3: # 3 other triples in this quad are already Red
                    can_red = False
                    break
            
            if can_red:
                # Assign Red
                colors[ti] = 0
                for qi in triple_to_quads[ti]:
                    red_count[qi] += 1
                    
                if bt(idx + 1):
                    return True
                
                # Unassign
                for qi in triple_to_quads[ti]:
                    red_count[qi] -= 1
                colors[ti] = -1
                
            # Try Blue (1)
            can_blue = True
            # Check blue constraints: any quint containing t becomes all blue?
            for qi in triple_to_quints[ti]:
                if blue_count[qi] == 9: # 9 other triples in this quint are already Blue
                    can_blue = False
                    break
            
            if can_blue:
                # Assign Blue
                colors[ti] = 1
                for qi in triple_to_quints[ti]:
                    blue_count[qi] += 1
                    
                if bt(idx + 1):
                    return True
                
                # Unassign
                for qi in triple_to_quints[ti]:
                    blue_count[qi] -= 1
                colors[ti] = -1
                
            return False
        
        # The naive backtracking might be too slow for n=34.
        # We need a better heuristic or a different order.
        # The "first fail" heuristic: pick the variable with the smallest domain.
        # Here domain is always 2 initially, but shrinks as constraints are applied.
        # Implementing dynamic variable ordering is complex.
        
        # Alternative: Use the fact that the problem is symmetric?
        # Or use a randomized restart?
        
        # Given the 145s budget, let's try a few things:
        # 1. Try n=34 with a limited time (e.g. 10s). If it fails, try n=33, etc.
        # 2. The backtracking might get stuck. 
        #    We can add a "restart" mechanism or randomize the order of colors (Red/Blue) 
        #    for each triple to escape local symmetries?
        
        # Let's modify bt to accept a random seed for tie-breaking or color order?
        # No, let's keep it deterministic first. If it's too slow, we'll see.
        
        # Actually, for n=34, the number of triples is C(34,3) = 5984.
        # Depth 5984 is huge. Backtracking will likely explode.
        # We MUST have a stronger pruning or a different strategy.
        
        # What if we use the SAT solver approach from inspiration 3?
        # DPLL with unit propagation.
        # But implementing a fast DPLL in Python within 145s for this size is tricky.
        
        # Let's look at the "error_repair" idea.
        # Maybe we can use a hybrid:
        # 1. Start with a random coloring.
        # 2. Find violations.
        # 3. Use backtracking ONLY on the variables involved in violations to repair them?
        #    This is "local backtracking".
        
        # Or, use the deterministic backtracking but with a "fail-fast" and "restart" strategy.
        
        # Let's try a different structural change:
        # **Greedy with Conflict-Driven Backjumping**?
        
        # Actually, let's look at the known solution.
        # The published construction has 34 points.
        # It is likely a specific algebraic construction.
        # If we can find it, we don't need to search.
        # But we don't have it.
        
        # Let's try to improve the SA? No, the directive says "STRUCTURAL change".
        
        # How about **Genetic Algorithm**?
        # Represent coloring as a bit string.
        # Fitness: -1 * (num_red_k4 + num_blue_k5).
        # Crossover: Uniform.
        # Mutation: Flip bits.
        # Selection: Tournament.
        # This is also local search-ish but with population.
        
        # Let's try **Simulated Annealing with Multi-Start** but with a better energy function?
        # The current SA uses a simple count.
        
        # What if we use **Min-Conflict Heuristic**?
        # 1. Randomly assign colors.
        # 2. Find all violated constraints (Red K4s, Blue K5s).
        # 3. For each violated constraint, pick a random triple in it.
        # 4. Flip that triple to the color that minimizes the number of violated constraints involving it.
        # 5. Repeat.
        
        # This is a very different approach from SA. It's a greedy local search that directly targets violations.
        
        def solve_min_conflict(n, time_limit):
            triples = list(combinations(range(n), 3))
            num_triples = len(triples)
            triple_index = {t: i for i, t in enumerate(triples)}
            
            quads = list(combinations(range(n), 4))
            quints = list(combinations(range(n), 5))
            
            # Precompute incidence
            # For each quad, list of 3 triples.
            # For each quint, list of 10 triples.
            
            quad_triples = []
            for q in quads:
                quad_triples.append([triple_index[t] for t in combinations(q, 3)])
                
            quint_triples = []
            for q in quints:
                quint_triples.append([triple_index[t] for t in combinations(q, 3)])
                
            # For each triple, list of quads and quints it belongs to.
            triple_to_quads = [[] for _ in range(num_triples)]
            triple_to_quints = [[] for _ in range(num_triples)]
            
            for qi, ts in enumerate(quad_triples):
                for t in ts:
                    triple_to_quads[t].append(qi)
                    
            for qi, ts in enumerate(quint_triples):
                for t in ts:
                    triple_to_quints[t].append(qi)
                    
            # Initialize random coloring
            colors = np.random.randint(0, 2, size=num_triples)
            
            # Count violations
            # red_violations[qi] = 1 if all 3 triples in quad qi are Red
            # blue_violations[qi] = 1 if all 10 triples in quint qi are Blue
            
            def get_violations():
                rv = np.zeros(len(quads), dtype=np.int8)
                bv = np.zeros(len(quints), dtype=np.int8)
                
                for qi, ts in enumerate(quad_triples):
                    if colors[ts[0]] == 0 and colors[ts[1]] == 0 and colors[ts[2]] == 0:
                        rv[qi] = 1
                        
                for qi, ts in enumerate(quint_triples):
                    all_blue = True
                    for t in ts:
                        if colors[t] != 1:
                            all_blue = False
                            break
                    if all_blue:
                        bv[qi] = 1
                        
                return rv, bv
                
            rv, bv = get_violations()
            total_violations = np.sum(rv) + np.sum(bv)
            
            if total_violations == 0:
                return colors.tolist()
                
            # Min-Conflict Loop
            # We maintain the list of violated quads and quints.
            violated_quads = np.where(rv > 0)[0]
            violated_quints = np.where(bv > 0)[0]
            
            # To speed up, we can maintain counts of "how many red" in each quad?
            # If we flip a triple, we update the counts of affected quads/quints.
            
            red_count_quad = np.zeros(len(quads), dtype=np.int8)
            blue_count_quint = np.zeros(len(quints), dtype=np.int8)
            
            # Initialize counts
            for qi, ts in enumerate(quad_triples):
                red_count_quad[qi] = np.sum(colors[ts] == 0)
            for qi, ts in enumerate(quint_triples):
                blue_count_quint[qi] = np.sum(colors[ts] == 1)
                
            # Re-evaluate violations based on counts
            rv = (red_count_quad == 3).astype(np.int8)
            bv = (blue_count_quint == 10).astype(np.int8)
            
            total_violations = np.sum(rv) + np.sum(bv)
            if total_violations == 0:
                return colors.tolist()
                
            # Loop
            while time.time() < time_limit:
                if total_violations == 0:
                    return colors.tolist()
                    
                # Pick a random violated constraint
                # We can pick from the union of violated quads and quints.
                # To make it efficient, we can maintain a list of violated indices.
                
                # Simple approach:
                # Pick a random violated quad or quint.
                # If no violated quads, pick from quints.
                # If no violated quints, pick from quads.
                # If both, pick randomly.
                
                if len(violated_quads) > 0 and (len(violated_quints) == 0 or np.random.rand() < 0.5):
                    # Pick violated quad
                    vi = np.random.choice(violated_quads)
                    ts = quad_triples[vi]
                    # Pick a random triple in this quad to flip
                    t_idx = np.random.choice(ts)
                    constraint_type = 'quad'
                    constraint_idx = vi
                else:
                    # Pick violated quint
                    vi = np.random.choice(violated_quints)
                    ts = quint_triples[vi]
                    # Pick a random triple in this quint to flip
                    t_idx = np.random.choice(ts)
                    constraint_type = 'quint'
                    constraint_idx = vi
                    
                # Determine best new color for t_idx
                # We want to minimize the number of violated constraints involving t_idx.
                
                old_color = colors[t_idx]
                
                # Calculate cost for Red (0)
                cost_red = 0
                # Check quads
                for qi in triple_to_quads[t_idx]:
                    # If we set t_idx to Red, what is the new red_count?
                    # Current red_count includes t_idx if it was Red.
                    # Let's just calculate the resulting violation status.
                    # If t_idx is Red, red_count[qi] stays same or increases?
                    # Easier: calculate hypothetical counts.
                    
                    # This is getting complicated to do incrementally without careful bookkeeping.
                    # Let's just evaluate the full impact of flipping t_idx to 0 and 1.
                    
                    # Actually, we can just flip it, check the change in total violations, 
                    # and revert if it's worse?
                    # That's O(1) if we have the counts.
                    
                    pass
                
                # Let's use a simpler Min-Conflict:
                # Flip the triple to the color that is NOT the current color.
                # This is "random walk" on the violated set.
                # Or, try both colors and pick the one with lower local cost.
                
                # Local cost: Number of violated constraints involving t_idx.
                
                def local_cost(color_val):
                    cost = 0
                    # Check quads
                    for qi in triple_to_quads[t_idx]:
                        # Hypothetical red count if t_idx is color_val
                        # If color_val == 0, red count is red_count_quad[qi] + (1 if old_color != 0 else 0)
                        # Wait, red_count_quad[qi] is the CURRENT count.
                        # If we change t_idx to 0:
                        #   If old_color == 0, count unchanged.
                        #   If old_color == 1, count increases by 1.
                        
                        # Let's just compute the new count.
                        if color_val == 0:
                            new_count = red_count_quad[qi] + (1 if old_color != 0 else 0)
                        else:
                            new_count = red_count_quad[qi] - (1 if old_color == 0 else 0)
                            
                        if new_count == 3:
                            cost += 1
                            
                    # Check quints
                    for qi in triple_to_quints[t_idx]:
                        if color_val == 1:
                            new_count = blue_count_quint[qi] + (1 if old_color != 1 else 0)
                        else:
                            new_count = blue_count_quint[qi] - (1 if old_color == 1 else 0)
                            
                        if new_count == 10:
                            cost += 1
                            
                    return cost
                    
                cost_0 = local_cost(0)
                cost_1 = local_cost(1)
                
                if cost_0 < cost_1:
                    new_color = 0
                elif cost_1 < cost_0:
                    new_color = 1
                else:
                    # Tie: random
                    new_color = np.random.randint(0, 2)
                    
                if new_color == old_color:
                    # If the current color is already optimal locally, we are stuck?
                    # In Min-Conflict, if we are in a local minimum where flipping doesn't help,
                    # we need a random restart or a random move.
                    # Let's just flip to the other color to escape?
                    # Or pick a random triple from the violated constraint, not necessarily t_idx?
                    # We already picked t_idx randomly.
                    # If both colors have same cost, and it's equal to current cost, we are stuck.
                    # Let's just flip to 1 - old_color.
                    new_color = 1 - old_color
                    
                # Apply flip
                if new_color != old_color:
                    colors[t_idx] = new_color
                    
                    # Update counts
                    if old_color == 0: # Was Red, now Blue (or vice versa)
                        # Red count decreases for quads
                        for qi in triple_to_quads[t_idx]:
                            red_count_quad[qi] -= 1
                        # Blue count increases for quints
                        for qi in triple_to_quints[t_idx]:
                            blue_count_quint[qi] += 1
                    else: # Was Blue, now Red
                        for qi in triple_to_quads[t_idx]:
                            red_count_quad[qi] += 1
                        for qi in triple_to_quints[t_idx]:
                            blue_count_quint[qi] -= 1
                            
                    # Update violations
                    # We only need to update violations for the affected quads and quints.
                    # But we need to update the global lists `violated_quads` and `violated_quints`.
                    # This is expensive if we do it from scratch.
                    
                    # Let's just recompute the total violations and the lists every step?
                    # No, that's O(N).
                    
                    # We can maintain the lists incrementally.
                    # But it's complex.
                    
                    # Given the time limit, let's just recompute the total violations 
                    # and pick a new random violated constraint in the next iteration.
                    # We don't strictly need to maintain the list of ALL violated constraints 
                    # if we can pick one efficiently.
                    
                    # Actually, we can just recompute `rv` and `bv` arrays?
                    # That's O(C(n,4) + C(n,5)).
                    # For n=34, ~300k operations. 
                    # If we do this every step, and we have ~100k steps in 145s...
                    # 300k * 100k = 30 billion. Too slow.
                    
                    # We MUST update incrementally.
                    
                    # Let's update the `rv` and `bv` arrays for the affected indices.
                    
                    # For quads affected by t_idx:
                    for qi in triple_to_quads[t_idx]:
                        # Check if it was violated before
                        was_violated = (red_count_quad[qi] == 3) # BEFORE update?
                        # We updated red_count_quad already.
                        # So we need to know the OLD state.
                        
                        # This is getting messy.
                        
                    # Let's stick to a simpler approach:
                    # Just recompute the total violations count, but not the lists.
                    # And pick a random triple from the WHOLE set if we don't have a list?
                    # No, that's too random.
                    
                    # Alternative: Keep the lists, but update them lazily?
                    # Or just recompute the lists every K steps?
                    
                    # Let's try a different Min-Conflict variant:
                    # 1. Find all violated constraints.
                    # 2. For each, pick a variable.
                    # 3. Flip it.
                    # 4. Repeat.
                    
                    # To find all violated constraints efficiently, we need the counts.
                    # We have the counts.
                    # We can just scan the counts arrays to find violations?
                    # That's O(N).
                    
                    # If we scan O(N) every step, and take O(1) to flip,
                    # total time is O(N * Steps).
                    # If Steps is 1000, and N is 300k, that's 300M ops.
                    # In Python, 300M ops is ~30s.
                    # So we can do ~1000 steps in 30s.
                    # Is 1000 steps enough to solve n=34?
                    # Probably not.
                    
                    # We need to be smarter.
                    
                    # Let's go back to **Backtracking** but with **Randomized Restarts**.
                    # The deterministic backtracking might get stuck.
                    # But if we randomize the ORDER of triples, we might find a solution faster.
                    
                    # Let's implement a randomized backtracking.
                    # 1. Shuffle the order of triples.
                    # 2. Run backtracking with this order.
                    # 3. If it fails (or times out), restart with a new shuffle.
                    
                    # This is a structural change from the current SA.
                    
                    pass # Placeholder for the logic we are discarding

        # Let's implement the Randomized Backtracking.
        
        def solve_randomized_backtracking(n, time_limit):
            triples = list(combinations(range(n), 3))
            num_triples = len(triples)
            triple_index = {t: i for i, t in enumerate(triples)}
            
            quads = list(combinations(range(n), 4))
            quints = list(combinations(range(n), 5))
            
            quad_index = {q: i for i, q in enumerate(quads)}
            quint_index = {q: i for i, q in enumerate(quints)}
            
            triple_to_quads = [[] for _ in range(num_triples)]
            triple_to_quints = [[] for _ in range(num_triples)]
            
            quad_triples = [[] for _ in range(len(quads))]
            quint_triples = [[] for _ in range(len(quints))]
            
            for q in quads:
                qi = quad_index[q]
                for t in combinations(q, 3):
                    ti = triple_index[t]
                    triple_to_quads[ti].append(qi)
                    quad_triples[qi].append(ti)
                    
            for q in quints:
                qi = quint_index[q]
                for t in combinations(q, 3):
                    ti = triple_index[t]
                    triple_to_quints[ti].append(qi)
                    quint_triples[qi].append(ti)
                    
            # Order of triples to assign
            # We will shuffle this order for each restart.
            
            def bt(order, colors, red_count, blue_count, idx):
                if time.time() > time_limit:
                    return False
                if idx == num_triples:
                    return True
                    
                ti = order[idx]
                
                # Try Red
                can_red = True
                for qi in triple_to_quads[ti]:
                    if red_count[qi] == 3:
                        can_red = False
                        break
                        
                if can_red:
                    colors[ti] = 0
                    for qi in triple_to_quads[ti]:
                        red_count[qi] += 1
                        
                    if bt(order, colors, red_count, blue_count, idx + 1):
                        return True
                        
                    for qi in triple_to_quads[ti]:
                        red_count[qi] -= 1
                    colors[ti] = -1
                    
                # Try Blue
                can_blue = True
                for qi in triple_to_quints[ti]:
                    if blue_count[qi] == 9:
                        can_blue = False
                        break
                        
                if can_blue:
                    colors[ti] = 1
                    for qi in triple_to_quints[ti]:
                        blue_count[qi] += 1
                        
                    if bt(order, colors, red_count, blue_count, idx + 1):
                        return True
                        
                    for qi in triple_to_quints[ti]:
                        blue_count[qi] -= 1
                    colors[ti] = -1
                    
                return False
                
            # Try multiple restarts
            while time.time() < time_limit:
                order = list(range(num_triples))
                random.shuffle(order)
                
                colors = [-1] * num_triples
                red_count = [0] * len(quads)
                blue_count = [0] * len(quints)
                
                # We need to pass time_limit to bt.
                # But bt is recursive.
                
                if bt(order, colors, red_count, blue_count, 0):
                    return colors
                    
            return None

    # Try n from 34 downwards
    n_targets = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]
    
    # Time allocation
    # Give more time to higher n.
    # 34: 20s
    # 33: 15s
    # 32: 10s
    # 31: 10s
    # 30: 10s
    # 29: 10s
    # 28: 10s
    # 27: 10s
    # 26: 5s
    # 25: 5s
    # 24: 5s
    # 23: 5s
    # 22: 5s
    # 21: 5s
    # 20: 5s
    # 19-4: 1s each (16s)
    # Total: 20+15+10*6+5*6+16 = 20+15+60+30+16 = 141s. Safe.
    
    time_alloc = {
        34: 20.0,
        33: 15.0,
        32: 10.0,
        31: 10.0,
        30: 10.0,
        29: 10.0,
        28: 10.0,
        27: 10.0,
        26: 5.0,
        25: 5.0,
        24: 5.0,
        23: 5.0,
        22: 5.0,
        21: 5.0,
        20: 5.0,
    }
    
    for n in n_targets:
        if time.time() > deadline - 2.0:
            break
            
        t_limit = time_alloc.get(n, 1.0)
        time_limit = time.time() + t_limit
        if time_limit > deadline - 2.0:
            time_limit = deadline - 2.0
            
        res = solve_randomized_backtracking(n, time_limit)
        if res is not None:
            return {'n': n, 'colors': res}
            
    # Fallback: n=4
    return {'n': 4, 'colors': [0, 1, 1, 1]}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
