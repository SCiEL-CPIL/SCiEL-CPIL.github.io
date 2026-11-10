import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 115

    def solve_for_n(n_target):
        # We build up from n=1 to n_target
        # But we can just try to find a coloring for n_target directly with random restarts
        # Incremental is better for small n, but for large n, maybe direct is better?
        # Let's stick to incremental as it's more structured.
        
        # To speed up, we only run this if time permits.
        if time.time() - start_time > time_limit:
            return 0

        color_dict = {}
        
        for n in range(2, n_target + 1):
            if time.time() - start_time > time_limit:
                return n - 1
            
            # Add vertex n-1
            new_triples = []
            for a in range(n-1):
                for b in range(a+1, n-1):
                    new_triples.append((a, b, n-1))
            
            for t in new_triples:
                color_dict[t] = random.randint(0, 1)
                
            # Local Search
            # Precompute lists of K4s and K5s involving n-1
            # To avoid regenerating every time
            
            red_k4_candidates = []
            for a in range(n-1):
                for b in range(a+1, n-1):
                    for c in range(b+1, n-1):
                        red_k4_candidates.append((a, b, c))
                        
            blue_k5_candidates = []
            for a in range(n-1):
                for b in range(a+1, n-1):
                    for c in range(b+1, n-1):
                        for d in range(c+1, n-1):
                            blue_k5_candidates.append((a, b, c, d))

            def is_red_k4(a, b, c):
                return (color_dict[(a,b,c)] == 0 and
                        color_dict[(a,b,n-1)] == 0 and
                        color_dict[(a,c,n-1)] == 0 and
                        color_dict[(b,c,n-1)] == 0)

            def is_blue_k5(a, b, c, d):
                return (color_dict[(a,b,c)] == 1 and
                        color_dict[(a,b,d)] == 1 and
                        color_dict[(a,b,n-1)] == 1 and
                        color_dict[(a,c,d)] == 1 and
                        color_dict[(a,c,n-1)] == 1 and
                        color_dict[(a,d,n-1)] == 1 and
                        color_dict[(b,c,d)] == 1 and
                        color_dict[(b,c,n-1)] == 1 and
                        color_dict[(b,d,n-1)] == 1 and
                        color_dict[(c,d,n-1)] == 1)

            # Simple loop
            max_iter = 2000
            for i in range(max_iter):
                if time.time() - start_time > time_limit:
                    return n - 1
                
                # Find a violation
                violation = None
                # Check random subset of candidates to speed up?
                # Or just check all?
                # For n=34, 46k checks.
                # Let's check all.
                
                # To make it faster, we can pick a random candidate and check it.
                # If we find a violation, fix it.
                # If we check all and find none, we are done.
                
                # Let's try to find ONE violation.
                found = False
                # Shuffle candidates? No, just pick random indices.
                # We want to be sure we find a violation if one exists.
                # Checking all is safer.
                
                for a, b, c in red_k4_candidates:
                    if is_red_k4(a, b, c):
                        violation = ('red', a, b, c)
                        found = True
                        break
                        
                if not found:
                    for a, b, c, d in blue_k5_candidates:
                        if is_blue_k5(a, b, c, d):
                            violation = ('blue', a, b, c, d)
                            found = True
                            break
                
                if not found:
                    return n # Success for this n
                
                # Fix violation
                if violation[0] == 'red':
                    _, a, b, c = violation
                    # Triples: (a,b,c), (a,b,n-1), (a,c,n-1), (b,c,n-1)
                    t = random.choice([(a,b,c), (a,b,n-1), (a,c,n-1), (b,c,n-1)])
                else:
                    _, a, b, c, d = violation
                    tris = [
                        (a,b,c), (a,b,d), (a,b,n-1),
                        (a,c,d), (a,c,n-1),
                        (a,d,n-1),
                        (b,c,d), (b,c,n-1),
                        (b,d,n-1),
                        (c,d,n-1)
                    ]
                    t = random.choice(tris)
                    
                color_dict[t] = 1 - color_dict[t]

            # If we didn't find a solution in max_iter, fail
            return n - 1

        return n_target

    # Try targets
    for target in [34, 32, 30, 28, 25, 20, 15, 10]:
        if time.time() - start_time > time_limit - 5:
            break
        n = solve_for_n(target)
        if n > 9:
            colors = []
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        colors.append(color_dict[(a,b,c)])
            return {'n': n, 'colors': colors}

    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
