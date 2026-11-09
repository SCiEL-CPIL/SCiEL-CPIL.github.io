"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    # Target n values to try (descending for max)
    # We know R(4,5;3) > 34 from published construction
    # Try to find valid colorings for larger n, or at least n=34
    
    def count_violations(n, colors):
        """Count red K4s and blue K5s in a coloring."""
        red_k4 = 0
        blue_k5 = 0
        
        # Precompute triple indices
        from itertools import combinations
        
        # Build triple lookup: (a,b,c) -> index
        triple_idx = {}
        idx = 0
        triples_list = []
        for a, b, c in combinations(range(n), 3):
            triple_idx[(a, b, c)] = idx
            triples_list.append((a, b, c))
            idx += 1
        
        # Count red K4: 4 vertices, all 4 triples red (0)
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[triple_idx[(a,b,c)]] == 0 and 
                colors[triple_idx[(a,b,d)]] == 0 and
                colors[triple_idx[(a,c,d)]] == 0 and
                colors[triple_idx[(b,c,d)]] == 0):
                red_k4 += 1
        
        # Count blue K5: 5 vertices, all 10 triples blue (1)
        for quint in combinations(range(n), 5):
            all_blue = True
            for triple in combinations(quint, 3):
                if colors[triple_idx[triple]] == 1:
                    continue
                else:
                    all_blue = False
                    break
            if all_blue:
                blue_k5 += 1
        
        return red_k4, blue_k5
    
    def random_coloring(n, rng):
        """Generate a random coloring."""
        from itertools import combinations
        num_triples = len(list(combinations(range(n), 3)))
        return [rng.randint(0, 1) for _ in range(num_triples)]
    
    def local_search(n, colors, rng, time_limit):
        """Perform local search to reduce violations."""
        from itertools import combinations
        
        triple_idx = {}
        idx = 0
        triples_list = []
        for a, b, c in combinations(range(n), 3):
            triple_idx[(a, b, c)] = idx
            triples_list.append((a, b, c))
            idx += 1
        
        # Precompute for each triple, which K4s and K5s it belongs to
        # For efficiency, we'll use a simpler approach: random flip with acceptance
        
        best_colors = list(colors)
        best_violations = count_violations(n, best_colors)
        
        current_violations = best_violations[0] + best_violations[1]
        
        max_iterations = 500000
        for it in range(max_iterations):
            if time.time() > time_limit:
                break
            
            # Pick a random triple to flip
            i = rng.randint(0, len(colors) - 1)
            old_val = colors[i]
            colors[i] = 1 - old_val
            
            new_violations = count_violations(n, colors)
            new_total = new_violations[0] + new_violations[1]
            
            if new_total < current_violations or (new_total == current_violations and rng.random() < 0.5):
                current_violations = new_total
                if new_total < best_violations[0] + best_violations[1]:
                    best_colors = list(colors)
                    best_violations = new_violations
            else:
                colors[i] = old_val
        
        return best_colors, best_violations
    
    def try_n(n, rng, time_budget):
        """Try to find a valid coloring for given n."""
        # Start with random coloring
        colors = random_coloring(n, rng)
        
        # Run local search
        colors, violations = local_search(n, colors, rng, start_time + time_budget)
        
        if violations[0] == 0 and violations[1] == 0:
            return colors, True
        
        return colors, False
    
    # Try different n values
    # We want to maximize n, so try larger values first but with less time
    # Published construction has n=34, so aim for at least that
    
    results = {}
    
    # Try n=34 first (known to be achievable)
    time_per_n = 10.0
    for n in [34, 35, 36, 37, 38, 39, 40]:
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(42 + n)
        colors, valid = try_n(n, rng, time_per_n)
        
        if valid:
            results[n] = colors
            print(f"Found valid coloring for n={n}", file=sys.stderr)
            # If we found a valid one, we can stop or continue to find larger
            if n >= 40:
                break
        else:
            print(f"No valid coloring found for n={n}", file=sys.stderr)
    
    # If no valid coloring found, try n=34 with more time
    if not results:
        rng = random.Random(123)
        colors, valid = try_n(34, rng, 10.0)
        if valid:
            results[34] = colors
            print("Found valid coloring for n=34 on retry", file=sys.stderr)
    
    # If still nothing, try a simpler construction: use modular arithmetic
    if not results:
        # Try the modular construction from the seed
        for n in [34, 33, 32, 31, 30]:
            from itertools import combinations
            colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
            red_k4, blue_k5 = count_violations(n, colors)
            if red_k4 == 0 and blue_k5 == 0:
                results[n] = colors
                print(f"Modular construction works for n={n}", file=sys.stderr)
                break
    
    # If still nothing, try random restarts with more attempts
    if not results:
        for n in [34, 33]:
            if time.time() > deadline - 3.0:
                break
            for attempt in range(50):
                if time.time() > deadline - 2.0:
                    break
                rng = random.Random(attempt * 1000 + n)
                colors, valid = try_n(n, rng, 0.5)
                if valid:
                    results[n] = colors
                    print(f"Found valid coloring for n={n} on attempt {attempt}", file=sys.stderr)
                    break
            if n in results:
                break
    
    # Fallback: return the largest n we found, or a small valid one
    if results:
        best_n = max(results.keys())
        return {'n': best_n, 'colors': results[best_n]}
    
    # Ultimate fallback: return a trivial valid coloring
    # For n=3, no K4 or K5 possible, so any coloring works
    from itertools import combinations
    n = 3
    colors = [0, 0, 0]
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
