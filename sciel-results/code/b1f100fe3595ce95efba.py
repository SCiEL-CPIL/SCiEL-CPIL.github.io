import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin

    def get_idx(i, j, n):
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def get_color(i, j, colors, n):
        if i > j:
            i, j = j, i
        return colors[get_idx(i, j, n)]

    def set_color(i, j, c, colors, n):
        if i > j:
            i, j = j, i
        colors[get_idx(i, j, n)] = c

    def count_violations(n, N, colors):
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_idx(i, j, n)]
                # N[c][i] is bitmask of neighbors of i in color c
                common = N[c][i] & N[c][j]
                if bin(common).count('1') >= 2:
                    violations += 1
        return violations

    def is_valid(n, N, colors):
        return count_violations(n, N, colors) == 0

    def local_search(n, initial_colors, time_budget):
        colors = initial_colors[:]
        n_edges = n * (n - 1) // 2
        
        # Build N
        N = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_idx(i, j, n)]
                N[c][i] |= (1 << j)
                N[c][j] |= (1 << i)
        
        # Initial check
        if is_valid(n, N, colors):
            return colors, True

        # Precompute edge list for random access? 
        # Or just generate random i, j.
        # Generating random i, j is O(1).
        
        end_time = start_time + time_budget
        no_improvement_count = 0
        
        while time.time() < end_time:
            # Pick a random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            c = colors[get_idx(i, j, n)]
            
            # Check if this edge is part of a violation
            # A violation exists if there are >= 2 common neighbors in color c
            common = N[c][i] & N[c][j]
            if bin(common).count('1') < 2:
                continue # No violation on this edge
            
            # Try to recolor
            recolor_success = False
            # Try other colors
            for new_c in random.sample(range(5), 5):
                if new_c == c:
                    continue
                
                # Check if new_c is safe
                # If we recolor (i,j) to new_c, does it create a C4?
                # It creates a C4 if i and j have >= 2 common neighbors in new_c
                new_common = N[new_c][i] & N[new_c][j]
                if bin(new_common).count('1') < 2:
                    # Apply change
                    # Remove from old color
                    N[c][i] &= ~(1 << j)
                    N[c][j] &= ~(1 << i)
                    
                    # Add to new color
                    N[new_c][i] |= (1 << j)
                    N[new_c][j] |= (1 << i)
                    
                    colors[get_idx(i, j, n)] = new_c
                    recolor_success = True
                    no_improvement_count = 0
                    break
            
            if not recolor_success:
                no_improvement_count += 1
                # If stuck for a long time, restart?
                # For now, just keep trying. The random picks might eventually find a path.
                # Or we can add a "kick" by randomly changing a color even if it creates a violation?
                # No, that increases violations.
                # Maybe just keep going.
            
            # Check if valid occasionally? 
            # It's expensive. 
            # But we can check if the specific edge we just fixed is still violating? 
            # No, other edges might be violating.
            # We should check global validity every K iterations.
            
            if no_improvement_count % 1000 == 0:
                if is_valid(n, N, colors):
                    return colors, True

        # Final check
        if is_valid(n, N, colors):
            return colors, True
        return colors, False

    best_n = 11
    # Seed a valid coloring for n=11 to be safe
    # The seed program used: [min(j - i, 11 - j + i) - 1 ...]
    # This is a circular coloring.
    # Let's verify if it's valid.
    # For n=11, colors are 0..4.
    # min(j-i, 11-(j-i)) - 1
    # Distances: 1->0, 2->1, 3->2, 4->3, 5->4.
    # This is a valid 5-coloring for C11? No, K11.
    # Let's just use this as a fallback.
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to improve
    # We have ~140s.
    # Try n=12 to 26.
    # Give each n a time slice.
    # If we find a valid n, we can try n+1.
    
    # Strategy: 
    # 1. Try n=12 with 5s. If success, best_n=12.
    # 2. Try n=13 with 5s. If success, best_n=13.
    # ...
    # 3. Try n=26 with remaining time.
    
    # Actually, it's better to try higher n first if we have time?
    # No, lower n is easier to find.
    # Let's allocate time based on difficulty.
    # n=12: 2s
    # n=13: 2s
    # ...
    # n=20: 5s
    # n=26: remaining
    
    # But if we fail 12, we might fail 13.
    # Let's just loop 12..26 and give each a chunk.
    # Total 140s.
    # 15 values (12..26).
    # ~9s each.
    
    for n in range(12, 27):
        if time.time() > deadline:
            break
        
        # Time budget for this n
        # If n is close to 26, give more time?
        # Or just equal time.
        time_budget = min(10.0, deadline - time.time())
        
        found_valid = False
        # Try multiple restarts within the budget
        restarts = 0
        while time.time() < deadline - 1.0: # Keep 1s for next loop/overhead
            # Initialize
            init_colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
            
            # Run local search with a small sub-budget
            # Let's give it 2 seconds per restart
            sub_budget = min(2.0, deadline - time.time() - 1.0)
            if sub_budget < 0.5:
                break
                
            colors, valid = local_search(n, init_colors, sub_budget)
            
            if valid:
                best_n = n
                best_colors = colors
                found_valid = True
                break # Found valid for this n, move to next n
            
            restarts += 1
            if time.time() > deadline - 5.0:
                break

        if found_valid:
            # If we found n, we can try n+1 in the next iteration of the loop
            # The loop continues automatically.
            pass

    return {'n': best_n, 'colors': best_colors}
