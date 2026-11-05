import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def solve_backtracking(n, time_limit, seed=None):
        """
        Backtracking with forward checking.
        Variables: triples (a,b,c).
        Domain: {0, 1}.
        Constraints: No red K4, no blue K5.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}

        # Precompute for each triple, which 4-sets and 5-sets it belongs to
        # And for each 4-set/5-set, which triples it contains
        four_sets = list(combinations(range(n), 4))
        five_sets = list(combinations(range(n), 5))
        
        # Map set index to list of triple indices
        set_4_triples = []
        for s4 in four_sets:
            ts = []
            for t in combinations(s4, 3):
                ts.append(triple_index[t])
            set_4_triples.append(ts)
            
        set_5_triples = []
        for s5 in five_sets:
            ts = []
            for t in combinations(s5, 3):
                ts.append(triple_index[t])
            set_5_triples.append(ts)

        # Map triple index to list of 4-set indices and 5-set indices
        triple_to_4sets = [[] for _ in range(num_triples)]
        for i, ts in enumerate(set_4_triples):
            for t_idx in ts:
                triple_to_4sets[t_idx].append(i)
                
        triple_to_5sets = [[] for _ in range(num_triples)]
        for i, ts in enumerate(set_5_triples):
            for t_idx in ts:
                triple_to_5sets[t_idx].append(i)

        # State:
        # colors: array of -1 (unassigned), 0, 1
        colors = np.full(num_triples, -1, dtype=np.int8)
        
        # Counts:
        # red_count_4[i]: number of red triples assigned in 4-set i
        # blue_count_5[j]: number of blue triples assigned in 5-set j
        # unassigned_4[i]: number of unassigned triples in 4-set i
        # unassigned_5[j]: number of unassigned triples in 5-set i
        
        red_count_4 = np.zeros(len(four_sets), dtype=np.int8)
        blue_count_5 = np.zeros(len(five_sets), dtype=np.int8)
        unassigned_4 = np.full(len(four_sets), 4, dtype=np.int8)
        unassigned_5 = np.full(len(five_sets), 10, dtype=np.int8)

        # MRV Heuristic:
        # We want to pick a variable (triple) that is involved in the most "tight" constraints.
        # A constraint is tight if it is close to being violated.
        # For a 4-set: if red_count == 3 and unassigned == 1, the remaining MUST be Blue.
        # If red_count == 3 and unassigned == 0 -> Violation (shouldn't happen if we check).
        # If red_count == 4 -> Violation.
        
        # We will maintain a list of "critical" triples or just iterate through all unassigned
        # and pick the one with highest "pressure".
        
        # To make it fast, we can just iterate through unassigned triples in a random order
        # and check if they are forced? Or use a simple priority queue?
        # Given n<=34, num_triples ~ 5984. Iterating all is O(N).
        # Let's try a simple approach: pick the first unassigned triple that is involved in
        # a 4-set with red_count==3 or a 5-set with blue_count==9.
        # If none, pick random.
        
        # Actually, a better MRV for CSP is "Least Constraining Value" or "Most Constrained Variable".
        # Most Constrained Variable: The variable that appears in the most constraints that are nearly violated.
        
        def get_pressure(t_idx):
            p = 0
            for s4 in triple_to_4sets[t_idx]:
                if red_count_4[s4] == 3 and unassigned_4[s4] == 1:
                    p += 100 # Forced
                elif red_count_4[s4] == 3:
                    p += 10
            for s5 in triple_to_5sets[t_idx]:
                if blue_count_5[s5] == 9 and unassigned_5[s5] == 1:
                    p += 100 # Forced
                elif blue_count_5[s5] == 9:
                    p += 10
            return p

        # Backtracking function
        # We use recursion. Depth is num_triples.
        # Python recursion limit might be an issue? 5984 is too deep for default 1000.
        # We must implement iterative backtracking or increase limit.
        # sys.setrecursionlimit(10000)
        import sys
        sys.setrecursionlimit(10000)

        # We will use a stack for iterative backtracking to avoid recursion depth issues and allow
        # easier time checking.
        
        # Stack entry: (t_idx, color, state_snapshot?)
        # Saving state snapshot is expensive.
        # Instead, we use a trail (undo log).
        
        trail = []
        
        # We need to pick the next variable dynamically.
        # This makes iterative backtracking harder because the "next" variable depends on current state.
        # Standard iterative BT with dynamic variable ordering requires re-picking at each step.
        
        # Let's try a recursive approach with a global time check and early exit.
        # If it hits recursion limit, we fail.
        # To mitigate, we can use a "randomized restart" strategy.
        # Each restart, we pick a random order of triples? No, that's too slow to generate.
        # Instead, we pick the next variable randomly among the "critical" ones, or randomly if none.
        
        def backtrack(depth):
            if time.time() > time_limit:
                return False, "timeout"
            
            if depth == num_triples:
                return True, "success"
            
            # Find candidate variables
            # We look for unassigned triples.
            # To speed up, we can maintain a list of unassigned indices?
            # Or just iterate 0..num_triples-1 and check colors[i] == -1.
            
            # Heuristic:
            # 1. If any triple is forced (due to 4-set red_count=3, unassigned=1 or 5-set blue_count=9, unassigned=1),
            #    we MUST assign it the forced value.
            # 2. Otherwise, pick a random unassigned triple (or one with high pressure).
            
            forced_t = -1
            forced_val = -1
            
            candidates = []
            
            for t_idx in range(num_triples):
                if colors[t_idx] != -1:
                    continue
                
                is_forced = False
                f_val = -1
                
                # Check 4-sets
                for s4 in triple_to_4sets[t_idx]:
                    if unassigned_4[s4] == 1:
                        if red_count_4[s4] == 3:
                            # Must be Blue
                            f_val = 1
                            is_forced = True
                            break
                        # If red_count < 3, it can be Red or Blue?
                        # If red_count == 3, it MUST be Blue to avoid Red K4.
                        # If red_count < 3, it CAN be Red.
                    elif red_count_4[s4] == 4:
                        # Should have been caught earlier?
                        # If we assign Red to a 4-set that already has 3 Reds, we make it 4.
                        # So if red_count_4[s4] == 3, we cannot assign Red.
                        # This is a domain wipeout if we consider constraints.
                        # In forward checking, we check if assigning a value leads to immediate violation.
                        pass

                # Check 5-sets
                if not is_forced:
                    for s5 in triple_to_5sets[t_idx]:
                        if unassigned_5[s5] == 1:
                            if blue_count_5[s5] == 9:
                                # Must be Red
                                f_val = 0
                                is_forced = True
                                break
                        pass

                if is_forced:
                    # Check if this forced value conflicts with other constraints?
                    # e.g. if forced Blue, but some 5-set has blue_count=9? No, we just checked.
                    # But what if assigning Blue creates a Blue K5 elsewhere?
                    # We check consistency after assignment.
                    forced_t = t_idx
                    forced_val = f_val
                    break
                else:
                    # Calculate pressure for ranking
                    p = 0
                    for s4 in triple_to_4sets[t_idx]:
                        if red_count_4[s4] == 3: p += 5
                    for s5 in triple_to_5sets[t_idx]:
                        if blue_count_5[s5] == 9: p += 5
                    candidates.append((p, t_idx))
            
            if forced_t != -1:
                t_idx = forced_t
                val = forced_val
                # Try only this value
                if not assign(t_idx, val):
                    return False, "conflict"
                success, status = backtrack(depth + 1)
                if success:
                    return True, status
                unassign(t_idx)
                return False, status # Propagate failure? Or try other values?
                # If forced, there are no other values. So if it fails, this branch is dead.
                
            else:
                # Pick a candidate
                if not candidates:
                    # No unassigned? Should be caught by depth check.
                    return True, "success"
                
                # Sort by pressure descending, then randomize top few?
                # Or just pick random from high pressure?
                # Let's pick the one with max pressure, with some randomness.
                # Sort is O(N log N). N=6000. 6000*log(6000) ~ 60k ops.
                # Doing this at every step (6000 steps) is 360M ops. Might be slow in Python.
                
                # Optimization: Just pick a random unassigned triple?
                # Random walk with backtracking might be slow.
                # Let's try picking the first unassigned triple in a shuffled order?
                # No, we need to react to state.
                
                # Let's try: Pick a random unassigned triple.
                # This is O(N) to find one.
                t_idx = random.choice([t for _, t in candidates])
                
                # Try values in random order
                vals = [0, 1]
                random.shuffle(vals)
                
                for val in vals:
                    # Check if val is allowed (forward checking)
                    if is_consistent(t_idx, val):
                        if not assign(t_idx, val):
                            return False, "conflict"
                        
                        success, status = backtrack(depth + 1)
                        if success:
                            return True, status
                        
                        unassign(t_idx)
                        
                        if time.time() > time_limit:
                            return False, "timeout"
                            
                return False, "conflict"

        def is_consistent(t_idx, val):
            # Check if assigning val to t_idx creates an immediate violation
            # in any 4-set or 5-set containing t_idx.
            
            if val == 0: # Red
                for s4 in triple_to_4sets[t_idx]:
                    if red_count_4[s4] == 3:
                        # Would become 4
                        return False
            else: # Blue
                for s5 in triple_to_5sets[t_idx]:
                    if blue_count_5[s5] == 9:
                        # Would become 10
                        return False
            return True

        def assign(t_idx, val):
            # Apply assignment and update counts
            trail.append(t_idx)
            colors[t_idx] = val
            
            if val == 0:
                for s4 in triple_to_4sets[t_idx]:
                    red_count_4[s4] += 1
                    unassigned_4[s4] -= 1
                    if red_count_4[s4] == 4:
                        return False # Conflict
            else:
                for s5 in triple_to_5sets[t_idx]:
                    blue_count_5[s5] += 1
                    unassigned_5[s5] -= 1
                    if blue_count_5[s5] == 10:
                        return False # Conflict
            return True

        def unassign(t_idx):
            val = colors[t_idx]
            colors[t_idx] = -1
            trail.pop()
            
            if val == 0:
                for s4 in triple_to_4sets[t_idx]:
                    red_count_4[s4] -= 1
                    unassigned_4[s4] += 1
            else:
                for s5 in triple_to_5sets[t_idx]:
                    blue_count_5[s5] -= 1
                    unassigned_5[s5] += 1

        # Start
        success, status = backtrack(0)
        if success:
            return colors.tolist()
        return None

    # Try n from 34 downwards
    n_targets = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]
    
    # Time allocation
    # Backtracking can be very fast if it finds a solution, or very slow if it hits a dead end.
    # Randomized restarts help.
    # Let's give 5s to 34, 5s to 33, etc.
    
    for n in n_targets:
        if time.time() > deadline - 2.0:
            break
            
        t_limit = 5.0
        if n > 30:
            t_limit = 10.0
            
        time_limit = time.time() + t_limit
        if time_limit > deadline - 2.0:
            time_limit = deadline - 2.0
            
        # Try a few random restarts
        found = False
        restarts = 5
        for r in range(restarts):
            res = solve_backtracking(n, time_limit, seed=r)
            if res is not None:
                return {'n': n, 'colors': res}
            if time.time() > time_limit:
                break
                
    # Fallback: n=4
    return {'n': 4, 'colors': [0, 1, 1, 1]}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
