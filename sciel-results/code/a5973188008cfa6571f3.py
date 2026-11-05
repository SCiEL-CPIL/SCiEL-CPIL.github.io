import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s buffer for safety

    # We aim for the maximum known valid n, which is 34.
    # The seed program likely returns n=9 or similar.
    # We will attempt to find a valid coloring for n=34.
    # If that fails, we fall back to n=33, then n=32, etc.
    # However, the problem statement says "The published construction has 34 points".
    # This suggests n=34 is achievable.
    
    # To make this robust and fast, we use a highly optimized WalkSAT/Local Search.
    # We represent the coloring as a bitarray or list of integers.
    # We precompute the incidence structure efficiently.
    
    target_n = 34
    
    def try_solve(n):
        """
        Attempts to find a valid coloring for n vertices.
        Returns None if failed or time is up.
        """
        if time.time() - start_time > time_limit - 5.0:
            return None
            
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Red constraints: No 4 vertices have all 4 triples red (0).
        # So for every 4-set, at least one triple is blue (1).
        # Clause: OR(triple_i == 1 for i in 4-set)
        red_clauses = []
        for s in combinations(range(n), 4):
            idxs = [triple_to_idx[(s[0], s[1], s[2])], 
                    triple_to_idx[(s[0], s[1], s[3])],
                    triple_to_idx[(s[0], s[2], s[3])],
                    triple_to_idx[(s[1], s[2], s[3])]]
            red_clauses.append(idxs)
            
        # Blue constraints: No 5 vertices have all 10 triples blue (1).
        # So for every 5-set, at least one triple is red (0).
        # Clause: OR(triple_i == 0 for i in 5-set)
        blue_clauses = []
        for s in combinations(range(n), 5):
            idxs = [triple_to_idx[(a, b, c)] for a, b, c in combinations(s, 3)]
            blue_clauses.append(idxs)
            
        num_red = len(red_clauses)
        num_blue = len(blue_clauses)
        total_clauses = num_red + num_blue
        
        # Precompute variable-to-clause incidence
        # var_to_red[v] = list of red clause indices containing v
        # var_to_blue[v] = list of blue clause indices containing v
        var_to_red = [[] for _ in range(num_triples)]
        var_to_blue = [[] for _ in range(num_triples)]
        
        for i, idxs in enumerate(red_clauses):
            for v in idxs:
                var_to_red[v].append(i)
                
        for i, idxs in enumerate(blue_clauses):
            for v in idxs:
                var_to_blue[v].append(i)
                
        # Initialize random coloring
        random.seed(42 + n)
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Initialize violation counts
        # red_counts[i] = number of 1s in red_clauses[i]. Violated if 0.
        # blue_counts[i] = number of 0s in blue_clauses[i]. Violated if 0.
        red_counts = [0] * num_red
        blue_counts = [0] * num_blue
        
        for i, idxs in enumerate(red_clauses):
            for v in idxs:
                if colors[v] == 1:
                    red_counts[i] += 1
                    
        for i, idxs in enumerate(blue_clauses):
            for v in idxs:
                if colors[v] == 0:
                    blue_counts[i] += 1
                    
        viol_red = sum(1 for c in red_counts if c == 0)
        viol_blue = sum(1 for c in blue_counts if c == 0)
        total_viol = viol_red + viol_blue
        
        if total_viol == 0:
            return colors
            
        # WalkSAT with delta evaluation
        # We maintain a list of violated clauses for fast selection?
        # For n=34, total_clauses is ~ 324k.
        # Scanning all clauses every step is O(324k).
        # With ~100k steps, that's 3e10 ops. Too slow in Python.
        
        # Optimization: Maintain a set of violated clause indices.
        # When a clause becomes satisfied, remove it. When it becomes violated, add it.
        
        violated_red = set(i for i in range(num_red) if red_counts[i] == 0)
        violated_blue = set(i for i in range(num_blue) if blue_counts[i] == 0)
        
        p_flip_random = 0.3
        max_iters = 200000
        best_viol = total_viol
        best_colors = colors.copy()
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            if total_viol == 0:
                return colors
                
            if total_viol < best_viol:
                best_viol = total_viol
                best_colors = colors.copy()
                
            # Pick a random violated clause
            # We need a random element from the union of violated_red and violated_blue.
            # Since we don't have a single set, we can pick a random type.
            if random.random() < 0.5:
                if not violated_red:
                    continue # All red satisfied, must be blue violated
                clause_idx = random.choice(list(violated_red))
                ctype = 0
            else:
                if not violated_blue:
                    continue # All blue satisfied, must be red violated
                clause_idx = random.choice(list(violated_blue))
                ctype = 1
                
            if ctype == 0:
                idxs = red_clauses[clause_idx]
            else:
                idxs = blue_clauses[clause_idx]
                
            # Choose variable to flip
            # Option 1: Random variable in clause
            # Option 2: Variable that minimizes delta in violations
            
            # Calculate delta for each variable in the clause
            # Delta = (new_viol - old_viol)
            
            # To speed up, we can just try random first, then delta if random fails often?
            # Or always use delta? Delta is O(clause_size * avg_incidence).
            # Clause size is 4 (red) or 10 (blue).
            # Avg incidence: 
            # Red: Each triple is in (n-2) red clauses. n=34 -> 32.
            # Blue: Each triple is in C(n-3, 2) blue clauses. C(31, 2) = 465.
            # So delta calculation is roughly 10 * 465 = 4650 ops.
            # 200k iters * 4650 = 9e8 ops. This is feasible in Python if optimized.
            
            best_v = -1
            best_delta = float('inf')
            
            # To make it faster, we can sample a few variables instead of all?
            # Let's try all first. If too slow, we'll see.
            
            for v in idxs:
                old_val = colors[v]
                new_val = 1 - old_val
                
                delta = 0
                
                # Update red clauses
                if new_val == 1:
                    # v becomes 1
                    # Red clauses containing v: count increases.
                    # If count was 0, it becomes 1 (satisfied). Delta -1.
                    for rc in var_to_red[v]:
                        if red_counts[rc] == 0:
                            delta -= 1
                else:
                    # v becomes 0
                    # Red clauses containing v: count decreases.
                    # If count was 1, it becomes 0 (violated). Delta +1.
                    for rc in var_to_red[v]:
                        if red_counts[rc] == 1:
                            delta += 1
                            
                # Update blue clauses
                if new_val == 0:
                    # v becomes 0
                    # Blue clauses containing v: count increases.
                    # If count was 0, it becomes 1 (satisfied). Delta -1.
                    for bc in var_to_blue[v]:
                        if blue_counts[bc] == 0:
                            delta -= 1
                else:
                    # v becomes 1
                    # Blue clauses containing v: count decreases.
                    # If count was 1, it becomes 0 (violated). Delta +1.
                    for bc in var_to_blue[v]:
                        if blue_counts[bc] == 1:
                            delta += 1
                            
                if delta < best_delta:
                    best_delta = delta
                    best_v = v
                    
            # Sometimes random is better to escape local minima
            if random.random() < p_flip_random:
                v = random.choice(idxs)
            else:
                v = best_v
                
            # Flip v
            old_val = colors[v]
            colors[v] = 1 - old_val
            new_val = colors[v]
            
            # Update counts and violated sets
            if new_val == 1:
                # Red clauses: count +1
                for rc in var_to_red[v]:
                    if red_counts[rc] == 0:
                        total_viol -= 1
                        violated_red.discard(rc)
                    red_counts[rc] += 1
                    
                # Blue clauses: count -1
                for bc in var_to_blue[v]:
                    if blue_counts[bc] == 1:
                        total_viol += 1
                        violated_blue.add(bc)
                    blue_counts[bc] -= 1
            else:
                # Red clauses: count -1
                for rc in var_to_red[v]:
                    if red_counts[rc] == 1:
                        total_viol += 1
                        violated_red.add(rc)
                    red_counts[rc] -= 1
                    
                # Blue clauses: count +1
                for bc in var_to_blue[v]:
                    if blue_counts[bc] == 0:
                        total_viol -= 1
                        violated_blue.discard(bc)
                    blue_counts[bc] += 1
                        
        # Return best found if it's valid
        if best_viol == 0:
            return best_colors
        return None

    # Try n=34
    result = try_solve(34)
    if result is not None:
        return {'n': 34, 'colors': result}
        
    # Try n=33
    if time.time() - start_time < time_limit - 10.0:
        result = try_solve(33)
        if result is not None:
            return {'n': 33, 'colors': result}
            
    # Try n=32
    if time.time() - start_time < time_limit - 15.0:
        result = try_solve(32)
        if result is not None:
            return {'n': 32, 'colors': result}

    # Fallback: Return a valid small n
    # n=9 is valid with the seed logic
    n = 9
    colors = [0 if (a+b+c)%3==0 else 1 for a,b,c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
