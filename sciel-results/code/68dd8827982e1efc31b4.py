"""Rewritten program: Local search solver for 3-color 4-AP avoidance."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    deadline = time.time() + 85  # Safety margin for 90s limit
    N = 128
    
    # Precompute all nonconstant 4-term progressions
    # a, a+d, a+2d, a+3d
    aps = []
    for a in range(N):
        for d in range(1, (N - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps.append((a, b, c, e))
            
    num_aps = len(aps)
    
    # Helper to count violations for a given coloring
    # A violation is a 4-AP with fewer than 3 distinct colors.
    def count_violations(colors):
        violations = 0
        for a, b, c, e in aps:
            # Check distinct colors
            # We can just check if the set size is < 3
            # To speed up, we can check specific patterns or just use set
            # Set creation is O(1) for small sets.
            if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                violations += 1
        return violations

    def local_search(k, max_iterations=500000, initial_temp=1.0):
        """
        Attempts to find a valid coloring with k colors.
        Returns the coloring if found, else None.
        """
        # Initialize with random colors
        colors = [random.randint(0, k - 1) for _ in range(N)]
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            return colors

        best_colors = colors[:]
        best_violations = current_violations
        
        # Simulated Annealing parameters
        temp = initial_temp
        cooling = 0.9999
        min_temp = 0.01
        
        for i in range(max_iterations):
            if time.time() > deadline:
                break
                
            if temp < min_temp:
                # Restart with a new random configuration if stuck
                colors = [random.randint(0, k - 1) for _ in range(N)]
                current_violations = count_violations(colors)
                if current_violations == 0:
                    return colors
                best_colors = colors[:]
                best_violations = current_violations
                temp = initial_temp

            # Pick a random position
            pos = random.randint(0, N - 1)
            old_color = colors[pos]
            
            # Pick a new color different from old
            new_color = random.randint(0, k - 1)
            if new_color == old_color:
                # Force a different color to ensure movement
                new_color = (new_color + 1) % k

            colors[pos] = new_color
            new_violations = count_violations(colors)
            
            delta = new_violations - current_violations
            
            if delta < 0 or (delta == 0 and random.random() < 0.5) or (random.random() < pow(2.718, -delta / temp)):
                # Accept move
                current_violations = new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    if current_violations == 0:
                        return colors
            else:
                # Reject move
                colors[pos] = old_color
                
            temp *= cooling
            
        if best_violations == 0:
            return best_colors
        return None

    # Try k=3 first
    best_solution = None
    for k in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
            
        # Multiple restarts for local search
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
            solution = local_search(k)
            if solution:
                # Verify
                if count_violations(solution) == 0:
                    best_solution = solution
                    found = True
                    break
        
        if found:
            break

    if best_solution is None:
        # Fallback: Greedy with backtracking for small N or just random high k
        # If local search fails, try a simple greedy that checks all constraints
        # This is slow but might work for k=8+
        # Or just return a valid high-color solution
        # A simple valid solution is assigning unique colors? No, N=128, labels 0..127.
        # If we use 128 colors, every 4-AP has 4 distinct colors (if positions are distinct).
        # But we want to minimize colors.
        # If we fail to find k<=6, we can try a constructive heuristic for higher k.
        
        # Let's try a simple greedy with lookahead for k=8
        colors = [-1] * N
        for i in range(N):
            valid_colors = [c for c in range(8)]
            # Check constraints with previous assignments
            # This is O(N^3) roughly, might be slow.
            # Instead, let's just use a known valid pattern if we have one, or random high k.
            # Random k=10 is very likely valid.
            pass
        
        # Emergency fallback: Random 10-coloring until valid
        for _ in range(1000):
            if time.time() > deadline:
                break
            colors = [random.randint(0, 9) for _ in range(N)]
            if count_violations(colors) == 0:
                best_solution = colors
                break
        
        if best_solution is None:
            # Absolute last resort: 128 unique colors
            best_solution = list(range(N))

    return best_solution
# EVOLVE-BLOCK-END
