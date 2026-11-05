import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def verify(n, colors):
        """
        Verify if a coloring is valid.
        Returns True if no Red K4 and no Blue K5.
        """
        # Map triple to index
        triples = list(combinations(range(n), 3))
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # Check Red K4 (4 vertices, 4 triples all red)
        for s4 in combinations(range(n), 4):
            count_red = 0
            for t in combinations(s4, 3):
                if colors[triple_index[t]] == 0:
                    count_red += 1
            if count_red == 4:
                return False
        
        # Check Blue K5 (5 vertices, 10 triples all blue)
        for s5 in combinations(range(n), 5):
            count_blue = 0
            for t in combinations(s5, 3):
                if colors[triple_index[t]] == 1:
                    count_blue += 1
            if count_blue == 10:
                return False
                
        return True

    def solve_csp(n, time_limit):
        """
        Solve the CSP for a given n using backtracking with forward checking.
        Variables: Triples in lexicographic order.
        Values: 0 (Red), 1 (Blue).
        Constraints:
        1. No 4-set has 4 Red triples.
        2. No 5-set has 10 Blue triples.
        """
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # Precompute incident sets
        # For each triple, which 4-sets and 5-sets does it belong to?
        four_sets = list(combinations(range(n), 4))
        five_sets = list(combinations(range(n), 5))
        
        triple_to_4sets = [[] for _ in range(num_triples)]
        triple_to_5sets = [[] for _ in range(num_triples)]
        
        # Map 4-set index to its 4 triples
        s4_triples = []
        for s4 in four_sets:
            ts = [triple_index[t] for t in combinations(s4, 3)]
            s4_triples.append(ts)
            for t_idx in ts:
                triple_to_4sets[t_idx].append(len(s4_triples) - 1)
                
        # Map 5-set index to its 10 triples
        s5_triples = []
        for s5 in five_sets:
            ts = [triple_index[t] for t in combinations(s5, 3)]
            s5_triples.append(ts)
            for t_idx in ts:
                triple_to_5sets[t_idx].append(len(s5_triples) - 1)
                
        # State
        colors = [-1] * num_triples
        # red_count_4[s4_idx]: number of red triples in this 4-set
        red_count_4 = [0] * len(four_sets)
        # blue_count_5[s5_idx]: number of blue triples in this 5-set
        blue_count_5 = [0] * len(five_sets)
        
        # We will assign triples in an order that might help pruning.
        # Lexicographic is fine, but we can try to order by "constraint tightness" later if needed.
        # For now, standard order.
        
        # Variable ordering: We can use dynamic variable ordering (MRV) but that's complex.
        # Let's stick to static order but with good value ordering.
        
        # To speed up, we can use a heuristic: 
        # If a 4-set has 3 red triples, the 4th MUST be blue.
        # If a 5-set has 9 blue triples, the 10th MUST be red.
        # This is unit propagation.
        
        def propagate():
            """
            Apply unit propagation. Returns False if contradiction.
            Modifies colors, red_count_4, blue_count_5 in place.
            """
            changed = True
            while changed:
                changed = False
                for t_idx in range(num_triples):
                    if colors[t_idx] != -1:
                        continue
                    
                    # Check if forced by 4-sets
                    forced_val = -1
                    for s4_idx in triple_to_4sets[t_idx]:
                        if red_count_4[s4_idx] == 3:
                            # Must be Blue to avoid 4 reds
                            forced_val = 1
                            break
                    if forced_val == -1:
                        # Check if forced by 5-sets
                        for s5_idx in triple_to_5sets[t_idx]:
                            if blue_count_5[s5_idx] == 9:
                                # Must be Red to avoid 10 blues
                                forced_val = 0
                                break
                    
                    if forced_val != -1:
                        # Assign
                        colors[t_idx] = forced_val
                        changed = True
                        # Update counts
                        if forced_val == 0: # Red
                            for s4_idx in triple_to_4sets[t_idx]:
                                red_count_4[s4_idx] += 1
                                if red_count_4[s4_idx] > 4:
                                    return False
                        else: # Blue
                            for s5_idx in triple_to_5sets[t_idx]:
                                blue_count_5[s5_idx] += 1
                                if blue_count_5[s5_idx] > 10:
                                    return False
            return True

        def backtrack(idx):
            """
            Recursive backtracking.
            idx: index of the next triple to assign.
            """
            if time.time() > time_limit:
                return False, "timeout"
            
            # Find next unassigned variable
            next_idx = -1
            for i in range(idx, num_triples):
                if colors[i] == -1:
                    next_idx = i
                    break
                    
            if next_idx == -1:
                # All assigned
                return True, None
                
            # Try values. 
            # Heuristic: Try the value that is "safer" or more constrained?
            # Usually, trying the value that satisfies more constraints first is good.
            # Or just try 0 then 1.
            
            # Let's try to prioritize the value that doesn't immediately create a near-violation.
            # But simple 0, 1 is often fine with propagation.
            
            for val in [0, 1]:
                # Check if val is allowed
                allowed = True
                if val == 0: # Red
                    for s4_idx in triple_to_4sets[next_idx]:
                        if red_count_4[s4_idx] == 4:
                            allowed = False
                            break
                else: # Blue
                    for s5_idx in triple_to_5sets[next_idx]:
                        if blue_count_5[s5_idx] == 10:
                            allowed = False
                            break
                
                if not allowed:
                    continue
                
                # Save state
                old_color = colors[next_idx]
                colors[next_idx] = val
                
                old_r4 = red_count_4.copy()
                old_b5 = blue_count_5.copy()
                
                # Apply
                if val == 0:
                    for s4_idx in triple_to_4sets[next_idx]:
                        red_count_4[s4_idx] += 1
                else:
                    for s5_idx in triple_to_5sets[next_idx]:
                        blue_count_5[s5_idx] += 1
                        
                # Propagate
                if propagate():
                    success, reason = backtrack(next_idx + 1)
                    if success:
                        return True, None
                    if reason == "timeout":
                        # Restore state before returning timeout? 
                        # Actually, if we timeout, we should just propagate the timeout up.
                        # But we need to clean up if we want to try other branches?
                        # No, if we timeout, we stop the whole search.
                        # So we can just return.
                        # However, we modified state. 
                        # Since we are returning False with "timeout", the parent will also return.
                        # We don't strictly need to restore if we are aborting the whole run.
                        # But to be clean:
                        colors[next_idx] = old_color
                        red_count_4 = old_r4
                        blue_count_5 = old_b5
                        return False, "timeout"
                
                # Restore state
                colors[next_idx] = old_color
                red_count_4 = old_r4
                blue_count_5 = old_b5
                
            return False, None

        # Initialize
        if not propagate():
            return None
            
        success, reason = backtrack(0)
        if success:
            return colors
        else:
            return None

    # Try n from 34 downwards
    # We use CSP backtracking which is deterministic but can be exponential.
    # We give a time limit per n.
    
    # The published construction has 34 points.
    # We try to find a valid coloring for n=34, then 33, etc.
    
    # Note: Backtracking on n=34 might be too slow for pure Python if the search space is large.
    # However, forward checking and unit propagation are very strong.
    # If n=34 is too hard, we fall back to smaller n.
    
    # To improve performance, we can add a small amount of randomness to variable ordering
    # or value ordering to avoid getting stuck in a specific bad path, but pure deterministic
    # with good propagation is often sufficient for small n.
    
    # Let's add a simple randomization to value ordering to help escape symmetries?
    # Actually, let's just run the deterministic CSP.
    
    # We will try n=34 with 30s, n=33 with 20s, etc.
    
    n_targets = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]
    
    # Time allocation
    time_alloc = {
        34: 30.0,
        33: 20.0,
        32: 15.0,
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
        if time.time() > deadline:
            break
            
        t_limit = time_alloc.get(n, 2.0)
        time_limit = time.time() + t_limit
        if time_limit > deadline:
            time_limit = deadline
            
        # CSP solver
        res = solve_csp(n, time_limit)
        
        if res is not None:
            # Verify just in case (should be fast for n<=34)
            if verify(n, res):
                return {'n': n, 'colors': res}
            else:
                # Should not happen if CSP is correct
                pass
                
    # Fallback if CSP fails for all (should not happen for n=4)
    n = 4
    colors = [0, 0, 0, 1]
    return {'n': n, 'colors': colors}

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(res['n'])
