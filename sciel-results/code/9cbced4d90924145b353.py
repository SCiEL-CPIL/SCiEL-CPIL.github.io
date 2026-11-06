<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    # Precompute all 4-term arithmetic progressions
    # A 4-term AP is (a, a+d, a+2d, a+3d) for d>=1, a>=0, a+3d < n
    # Positions are 0-indexed: 0..127
    progressions = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(n - 3 * d):
            progressions.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each position, precompute which progressions it's part of and its index in them
    pos_in_prog = [[] for _ in range(n)]
    for pi, (p0, p1, p2, p3) in enumerate(progressions):
        pos_in_prog[p0].append((pi, 0))
        pos_in_prog[p1].append((pi, 1))
        pos_in_prog[p2].append((pi, 2))
        pos_in_prog[p3].append((pi, 3))
    
    def count_violations(colors):
        """Count 4-term APs that use fewer than 3 distinct colors."""
        violations = 0
        for (p0, p1, p2, p3) in progressions:
            c0, c1, c2, c3 = colors[p0], colors[p1], colors[p2], colors[p3]
            # Count distinct colors
            distinct = 1
            if c1 != c0: distinct += 1
            if c2 != c0 and c2 != c1: distinct += 1
            if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
            if distinct < 3:
                violations += 1
        return violations
    
    def count_distinct_colors(colors):
        return len(set(colors))
    
    # Start with greedy solution
    colors = []
    for x in range(n):
        best_color = 0
        for color in range(128):
            ok = True
            for (pi, idx) in pos_in_prog[x]:
                p0, p1, p2, p3 = progressions[pi]
                # Check if assigning this color would violate
                # We need at least 3 distinct colors in the AP
                # For now, just try the greedy approach
                pass
            # Simple greedy: try smallest color that doesn't immediately cause issues
            # Actually let's just use a simple initial coloring
            best_color = x % 4  # 4 colors initially
            break
        colors.append(best_color)
    
    # Better initial: use greedy from original
    colors = []
    for x in range(1, n + 1):
        for color in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                if x - 3 * d >= 1:
                    c0 = colors[x - 1 - 3 * d]
                    c1 = colors[x - 1 - 2 * d]
                    c2 = colors[x - 1 - d]
                    distinct = len({color, c0, c1, c2})
                    if distinct < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                break
        else:
            colors.append(0)
    
    initial_colors = list(colors)
    initial_distinct = count_distinct_colors(initial_colors)
    initial_violations = count_violations(initial_colors)
    
    best_colors = initial_colors[:]
    best_distinct = initial_distinct
    best_violations = initial_violations
    
    # Simulated annealing
    if initial_violations == 0:
        # Start from valid, try to reduce distinct colors
        current = initial_colors[:]
        current_distinct = initial_distinct
        current_violations = 0
    else:
        current = initial_colors[:]
        current_distinct = initial_distinct
        current_violations = initial_violations
    
    temperature = 5.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random position and new color
        pos = random.randint(0, n - 1)
        old_color = current[pos]
        new_color = random.randint(0, best_distinct - 1)
        
        if new_color == old_color:
            continue
        
        # Compute change in violations
        delta_v = 0
        for (pi, idx) in pos_in_prog[pos]:
            p0, p1, p2, p3 = progressions[pi]
            # Old distinct count
            oc = [current[p0], current[p1], current[p2], current[p3]]
            old_distinct = len(set(oc))
            # New distinct count
            nc = [current[p0], current[p1], current[p2], current[p3]]
            nc[idx] = new_color
            new_distinct = len(set(nc))
            
            old_bad = 1 if old_distinct < 3 else 0
            new_bad = 1 if new_distinct < 3 else 0
            delta_v += new_bad - old_bad
        
        current[pos] = new_color
        current_violations += delta_v
        
        # Compute change in distinct colors
        old_dc = count_distinct_colors(current)
        
        # Acceptance criterion
        if current_violations == 0:
            # Valid: minimize distinct colors
            if old_dc <= current_violations or True:  # just compare distinct
                pass
        
        # Energy: violations * 100 + distinct_colors
        old_energy = current_violations * 100 + old_dc
        new_energy = current_violations * 100 + len(set(current))
        
        if new_energy < old_energy or random.random() < pow(2.718281828, -(new_energy - old_energy) / temperature):
            if current_violations == 0:
                dc = len(set(current))
                if dc < best_distinct:
                    best_distinct = dc
                    best_colors = current[:]
                    best_violations = 0
            # Keep the move
            pass
        else:
            # Reject
            current[pos] = old_color
            current_violations -= delta_v
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
    
    # Final: ensure we return a valid coloring
    if best_violations == 0:
        return best_colors
    else:
        # Fallback to greedy
        colors = []
        for x in range(1, n + 1):
            for color in range(128):
                valid = True
                for d in range(1, (x - 1) // 3 + 1):
                    if x - 3 * d >= 1:
                        c0 = colors[x - 1 - 3 * d]
                        c1 = colors[x - 1 - 2 * d]
                        c2 = colors[x - 1 - d]
                        distinct = len({color, c0, c1, c2})
                        if distinct < 3:
                            valid = False
                            break
                if valid:
                    colors.append(color)
                    break
            else:
                colors.append(0)
        return colors
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, let me reconsider. The above approach has issues with the acceptance criterion logic. Let me write a cleaner version:

