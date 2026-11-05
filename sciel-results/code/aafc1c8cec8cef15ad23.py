import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s buffer for overhead and verification
    
    # We aim for n=34, which is the known lower bound for R(4,5;3) > 34.
    # If we succeed, score is 34/9 ~ 3.7.
    # If we can find n=35, score is 35/9 ~ 3.8.
    # Let's try to find a valid coloring for n=34 first.
    
    def get_conflicts(n, colors):
        """
        Returns a list of conflicts.
        Red conflict: 4 vertices, all 4 triples red (0).
        Blue conflict: 5 vertices, all 10 triples blue (1).
        """
        # Map triple index to color
        # Triple (a,b,c) index in lexicographic order
        # We need a fast way to get color of triple (a,b,c)
        # Precompute index map? 
        # n=34, C(34,3) = 5984. 
        # We can store colors in a dict or list.
        # List is faster. Index calculation:
        # It's easier to just generate triples and check.
        
        # For efficiency in local search, we should maintain counts.
        # But for initial check, let's just do a full scan.
        
        red_conflicts = []
        blue_conflicts = []
        
        # Check Red K4
        # For every 4-set, check if all 4 triples are red.
        # Number of 4-sets in 34: C(34,4) = 46376.
        # For each, check 4 triples.
        
        # Check Blue K5
        # For every 5-set, check if all 10 triples are blue.
        # Number of 5-sets in 34: C(34,5) = 278256.
        # For each, check 10 triples.
        
        # This is O(C(n,4) + C(n,5)) which is fine for n=34.
        
        # To speed up, we can precompute for each vertex pair (a,b) the list of c's?
        # Or just iterate.
        
        # Let's use a helper to get triple color
        # We need the index of (a,b,c) in the lexicographic list of combinations.
        # This is complex to calculate on the fly.
        # Instead, let's store colors in a dictionary keyed by (a,b,c) for O(1) lookup.
        # Or, since we are going to do local search, we need to update efficiently.
        
        # Let's re-structure:
        # We will maintain `colors` as a list of length C(n,3).
        # We need a fast way to get the index.
        
        # Alternative: Use a 3D array or dictionary for colors.
        # colors[a][b][c]
        # This makes updates O(1) and lookups O(1).
        
        pass

    # Better Approach for Local Search:
    # Use a dictionary `col` where `col[(a,b,c)]` is 0 or 1.
    # Maintain sets of conflicts.
    
    def try_find_coloring(n, max_time):
        # Initialize
        random.seed(42) # Fixed seed for reproducibility in evaluation? 
        # Actually, random restarts are better.
        
        best_conflicts = float('inf')
        best_colors = None
        
        for attempt in range(50): # Max 50 restarts
            if time.time() - start_time > max_time:
                break
                
            # Initialize colors randomly
            col = {}
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        col[(a,b,c)] = random.randint(0, 1)
            
            # Compute initial conflicts
            red_conflicts = []
            blue_conflicts = []
            
            # Red K4
            for quad in combinations(range(n), 4):
                a,b,c,d = quad
                if col[(a,b,c)] == 0 and col[(a,b,d)] == 0 and col[(a,c,d)] == 0 and col[(b,c,d)] == 0:
                    red_conflicts.append(quad)
                    
            # Blue K5
            for quint in combinations(range(n), 5):
                v1,v2,v3,v4,v5 = quint
                triples = [
                    (v1,v2,v3), (v1,v2,v4), (v1,v2,v5),
                    (v1,v3,v4), (v1,v3,v5), (v1,v4,v5),
                    (v2,v3,v4), (v2,v3,v5), (v2,v4,v5),
                    (v3,v4,v5)
                ]
                if all(col[t] == 1 for t in triples):
                    blue_conflicts.append(quint)
            
            total_conflicts = len(red_conflicts) + len(blue_conflicts)
            if total_conflicts < best_conflicts:
                best_conflicts = total_conflicts
                best_colors = dict(col)
                
            if total_conflicts == 0:
                return col
            
            # Local Search
            # We will try to flip vertices in random conflicts.
            # To be efficient, we can just pick a random conflict and try flipping one of its vertices.
            
            for step in range(10000):
                if time.time() - start_time > max_time:
                    return best_colors
                
                if not red_conflicts and not blue_conflicts:
                    return col
                    
                # Pick a random conflict
                # If red conflicts exist, pick one, else pick blue
                if red_conflicts and (random.random() < 0.5 or not blue_conflicts):
                    conflict = random.choice(red_conflicts)
                    # Try flipping a vertex in this red K4
                    # If we flip a vertex, the K4 is broken.
                    # But it might create new conflicts.
                    # We need to evaluate the change.
                    
                    # For simplicity, we just try flipping each vertex in the conflict
                    # and see if total conflicts decreases.
                    # This is expensive if we recompute all conflicts.
                    
                    # Heuristic: Just flip a random vertex in the conflict and hope for the best?
                    # Or do a proper delta evaluation.
                    
                    # Let's do a simple delta evaluation for the specific conflict.
                    # Flipping a vertex in a Red K4 breaks that K4.
                    # It might form new Red K4s or Blue K5s involving that vertex.
                    
                    # This is complex. Let's use a simpler "random walk with hill climbing on a sampled metric".
                    
                    # Actually, for n=34, a full re-evaluation of conflicts takes:
                    # C(34,4) + C(34,5) ~ 300k checks.
                    # Each check is O(1) lookups.
                    # 300k checks is fast in Python? Maybe 0.1s.
                    # So we can afford a full re-evaluation every few steps.
                    
                    # Let's just pick a random vertex from the conflict, flip it, 
                    # and re-evaluate total conflicts.
                    # If it improves, keep it. If not, revert.
                    # To avoid getting stuck, we can also accept neutral moves.
                    
                    v = random.choice(conflict)
                    
                    # Flip v
                    # We need to update col for all triples containing v.
                    # And re-evaluate conflicts.
                    
                    # To make re-evaluation faster, we can just recompute the list of conflicts from scratch.
                    
                    # Save old state
                    old_triples = []
                    for a in range(n):
                        for b in range(a+1, n):
                            for c in range(b+1, n):
                                if v in (a,b,c):
                                    old_triples.append((a,b,c))
                                    
                    for t in old_triples:
                        col[t] = 1 - col[t]
                        
                    # Recompute conflicts
                    new_red = []
                    new_blue = []
                    for quad in combinations(range(n), 4):
                        a,b,c,d = quad
                        if col[(a,b,c)] == 0 and col[(a,b,d)] == 0 and col[(a,c,d)] == 0 and col[(b,c,d)] == 0:
                            new_red.append(quad)
                            
                    for quint in combinations(range(n), 5):
                        v1,v2,v3,v4,v5 = quint
                        triples = [
                            (v1,v2,v3), (v1,v2,v4), (v1,v2,v5),
                            (v1,v3,v4), (v1,v3,v5), (v1,v4,v5),
                            (v2,v3,v4), (v2,v3,v5), (v2,v4,v5),
                            (v3,v4,v5)
                        ]
                        if all(col[t] == 1 for t in triples):
                            new_blue.append(quint)
                            
                    new_total = len(new_red) + len(new_blue)
                    old_total = total_conflicts
                    
                    if new_total <= old_total:
                        # Accept
                        red_conflicts = new_red
                        blue_conflicts = new_blue
                        total_conflicts = new_total
                        if total_conflicts < best_conflicts:
                            best_conflicts = total_conflicts
                            best_colors = dict(col)
                        if total_conflicts == 0:
                            return col
                    else:
                        # Revert
                        for t in old_triples:
                            col[t] = 1 - col[t]
                            
                else:
                    if blue_conflicts:
                        conflict = random.choice(blue_conflicts)
                        v = random.choice(conflict)
                        
                        old_triples = []
                        for a in range(n):
                            for b in range(a+1, n):
                                for c in range(b+1, n):
                                    if v in (a,b,c):
                                        old_triples.append((a,b,c))
                                        
                        for t in old_triples:
                            col[t] = 1 - col[t]
                            
                        new_red = []
                        new_blue = []
                        for quad in combinations(range(n), 4):
                            a,b,c,d = quad
                            if col[(a,b,c)] == 0 and col[(a,b,d)] == 0 and col[(a,c,d)] == 0 and col[(b,c,d)] == 0:
                                new_red.append(quad)
                                
                        for quint in combinations(range(n), 5):
                            v1,v2,v3,v4,v5 = quint
                            triples = [
                                (v1,v2,v3), (v1,v2,v4), (v1,v2,v5),
                                (v1,v3,v4), (v1,v3,v5), (v1,v4,v5),
                                (v2,v3,v4), (v2,v3,v5), (v2,v4,v5),
                                (v3,v4,v5)
                            ]
                            if all(col[t] == 1 for t in triples):
                                new_blue.append(quint)
                                
                        new_total = len(new_red) + len(new_blue)
                        old_total = total_conflicts
                        
                        if new_total <= old_total:
                            red_conflicts = new_red
                            blue_conflicts = new_blue
                            total_conflicts = new_total
                            if total_conflicts < best_conflicts:
                                best_conflicts = total_conflicts
                                best_colors = dict(col)
                            if total_conflicts == 0:
                                return col
                        else:
                            for t in old_triples:
                                col[t] = 1 - col[t]
                            
        return best_colors

    # Try to find a valid coloring for n=34
    n_target = 34
    colors_dict = try_find_coloring(n_target, time_limit)
    
    if colors_dict is None:
        # Fallback to seed
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
        
    # Verify
    # Check if colors_dict is valid
    valid = True
    for quad in combinations(range(n_target), 4):
        a,b,c,d = quad
        if colors_dict[(a,b,c)] == 0 and colors_dict[(a,b,d)] == 0 and colors_dict[(a,c,d)] == 0 and colors_dict[(b,c,d)] == 0:
            valid = False
            break
    if valid:
        for quint in combinations(range(n_target), 5):
            v1,v2,v3,v4,v5 = quint
            triples = [
                (v1,v2,v3), (v1,v2,v4), (v1,v2,v5),
                (v1,v3,v4), (v1,v3,v5), (v1,v4,v5),
                (v2,v3,v4), (v2,v3,v5), (v2,v4,v5),
                (v3,v4,v5)
            ]
            if all(colors_dict[t] == 1 for t in triples):
                valid = False
                break
                
    if valid:
        # Convert dict to list in lexicographic order
        colors_list = []
        for a in range(n_target):
            for b in range(a+1, n_target):
                for c in range(b+1, n_target):
                    colors_list.append(colors_dict[(a,b,c)])
        return {'n': n_target, 'colors': colors_list}
    else:
        # Fallback to seed
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0
    
    def try_find_coloring(n, max_time):
        best_conflicts = float('inf')
        best_colors = None
        
        for attempt in range(100):
            if time.time() - start_time > max_time:
                break
                
            random.seed(attempt)
            col = {}
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        col[(a,b,c)] = random.randint(0, 1)
            
            red_conflicts = []
            blue_conflicts = []
            
            for quad in combinations(range(n), 4):
                a,b,c,d = quad
                if col[(a,b,c)] == 0 and col[(a,b,d)] == 0 and col[(a,c,d)] == 0 and col[(b,c,d)] == 0:
                    red_conflicts.append(quad)
                    
            for quint in combinations(range(n), 5):
                v1,v2,v3,v4,v5 = quint
                triples = [
                    (v1,v2,v3), (v1,v2,v4), (v1,v2,v5),
                    (v1,v3,v4), (v1,v3,v5), (v1,v4,v5),
                    (v2,v3,v4), (v2,v3,v5), (v2,v4,v5),
                    (v3,v4,v5)
                ]
                if all(col[t] == 1 for t in triples):
                    blue_conflicts.append(quint)
            
            total_conflicts = len(red_conflicts) + len(blue_conflicts)
            if total_conflicts < best_conflicts:
                best_conflicts = total_conflicts
                best_colors = dict(col)
                
            if total_conflicts == 0:
                return col
            
            for step in range(5000):
                if time.time() - start_time > max_time:
                    return best_colors
                
                if not red_conflicts and not blue_conflicts:
                    return col
                    
                if red_conflicts and (random.random() < 0.5 or not blue_conflicts):
                    conflict = random.choice(red_conflicts)
                    v = random.choice(conflict)
                    
                    old_triples = []
                    for a in range(n):
                        for b in range(a+1, n):
                            for c in range(b+1, n):
                                if v in (a,b,c):
                                    old_triples.append((a,b,c))
                                    
                    for t in old_triples:
                        col[t] = 1 - col[t]
                        
                    new_red = []
                    new_blue = []
                    for quad in combinations(range(n), 4):
                        a,b,c,d = quad
                        if col[(a,b,c)] == 0 and col[(a,b,d)] == 0 and col[(a,c,d)] == 0 and col[(b,c,d)] == 0:
                            new_red.append(quad)
                            
                    for quint in combinations(range(n), 5):
                        v1,v2,v3,v4,v5 = quint
                        triples = [
                            (v1,v2,v3), (v1,v2,v4), (v1,v2,v5),
                            (v1,v3,v4), (v1,v3,v5), (v1,v4,v5),
                            (v2,v3,v4), (v2,v3,v5), (v2,v4,v5),
                            (v3,v4,v5)
                        ]
                        if all(col[t] == 1 for t in triples):
                            new_blue.append(quint)
                            
                    new_total = len(new_red) + len(new_blue)
                    old_total = total_conflicts
                    
                    if new_total <= old_total:
                        red_conflicts = new_red
                        blue_conflicts = new_blue
                        total_conflicts = new_total
                        if total_conflicts < best_conflicts:
                            best_conflicts = total_conflicts
                            best_colors = dict(col)
                        if total_conflicts == 0:
                            return col
                    else:
                        for t in old_triples:
                            col[t] = 1 - col[t]
                            
                else:
                    if blue_conflicts:
                        conflict = random.choice(blue_conflicts)
                        v = random.choice(conflict)
                        
                        old_triples = []
                        for a in range(n):
                            for b in range(a+1, n):
                                for c in range(b+1, n):
                                    if v in (a,b,c):
                                        old_triples.append((a,b,c))
                                        
                        for t in old_triples:
                            col[t] = 1 - col[t]
                            
                        new_red = []
                        new_blue = []
                        for quad in combinations(range(n), 4):
                            a,b,c,d = quad
                            if col[(a,b,c)] == 0 and col[(a,b,d)] == 0 and col[(a,c,d)] == 0 and col[(b,c,d)] == 0:
                                new_red.append(quad)
                                
                        for quint in combinations(range(n), 5):
                            v1,v2,v3,v4,v5 = quint
                            triples = [
                                (v1,v2,v3), (v1,v2,v4), (v1,v2,v5),
                                (v1,v3,v4), (v1,v3,v5), (v1,v4,v5),
                                (v2,v3,v4), (v2,v3,v5), (v2,v4,v5),
                                (v3,v4,v5)
                            ]
                            if all(col[t] == 1 for t in triples):
                                new_blue.append(quint)
                                
                        new_total = len(new_red) + len(new_blue)
                        old_total = total_conflicts
                        
                        if new_total <= old_total:
                            red_conflicts = new_red
                            blue_conflicts = new_blue
                            total_conflicts = new_total
                            if total_conflicts < best_conflicts:
                                best_conflicts = total_conflicts
                                best_colors = dict(col)
                            if total_conflicts == 0:
                                return col
                        else:
                            for t in old_triples:
                                col[t] = 1 - col[t]
                            
        return best_colors

    n_target = 34
    colors_dict = try_find_coloring(n_target, time_limit)
    
    if colors_dict is None:
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
        
    valid = True
    for quad in combinations(range(n_target), 4):
        a,b,c,d = quad
        if colors_dict[(a,b,c)] == 0 and colors_dict[(a,b,d)] == 0 and colors_dict[(a,c,d)] == 0 and colors_dict[(b,c,d)] == 0:
            valid = False
            break
    if valid:
        for quint in combinations(range(n_target), 5):
            v1,v2,v3,v4,v5 = quint
            triples = [
                (v1,v2,v3), (v1,v2,v4), (v1,v2,v5),
                (v1,v3,v4), (v1,v3,v5), (v1,v4,v5),
                (v2,v3,v4), (v2,v3,v5), (v2,v4,v5),
                (v3,v4,v5)
            ]
            if all(colors_dict[t] == 1 for t in triples):
                valid = False
                break
                
    if valid:
        colors_list = []
        for a in range(n_target):
            for b in range(a+1, n_target):
                for c in range(b+1, n_target):
                    colors_list.append(colors_dict[(a,b,c)])
        return {'n': n_target, 'colors': colors_list}
    else:
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
# EVOLVE-BLOCK-END
