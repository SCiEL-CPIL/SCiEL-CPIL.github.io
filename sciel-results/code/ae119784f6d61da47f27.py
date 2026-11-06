import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 80  # safety margin before 90s kill
    
    N = 128
    positions = list(range(1, N + 1))  # 1..128
    
    # Precompute all nonconstant 4-term arithmetic progressions
    # A 4-term AP is (a, a+d, a+2d, a+3d) with d >= 1 and all positions in [1, 128]
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            a3 = a + 3 * d
            if a3 <= N:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a3 - 1))  # 0-indexed
    aps = np.array(aps, dtype=np.int32)
    
    def count_colors(colors):
        return len(set(colors))
    
    def check_valid(colors):
        """Check if every nonconstant 4-term AP uses at least 3 distinct colors."""
        c = np.array(colors, dtype=np.int32)
        for i in range(len(aps)):
            p = aps[i]
            vals = c[p]
            if len(set(vals.tolist())) < 3:
                return False
        return True
    
    def count_violations(colors):
        """Count number of 4-term APs that use fewer than 3 distinct colors."""
        c = np.array(colors, dtype=np.int32)
        violations = 0
        for i in range(len(aps)):
            p = aps[i]
            vals = c[p]
            if len(set(vals.tolist())) < 3:
                violations += 1
        return violations
    
    # Phase 1: Greedy construction to get a valid coloring
    # Start with random assignment using k colors, then repair
    best_result = None
    best_colors_used = N + 1
    
    # Try to build with few colors
    for k in [3, 4, 5, 6, 7, 8]:
        if time.time() > DEADLINE:
            break
        
        # Multiple restarts for each k
        for restart in range(50):
            if time.time() > DEADLINE:
                break
            
            # Random initial coloring with k colors
            colors = [random.randint(0, k - 1) for _ in range(N)]
            
            # Local search: minimize violations
            for step in range(5000):
                if time.time() > DEADLINE:
                    break
                
                # Pick a random position to change
                pos = random.randint(0, N - 1)
                old_color = colors[pos]
                
                # Find the best new color that reduces violations
                best_new_color = old_color
                best_viol = count_violations(colors)
                
                for new_color in range(k):
                    if new_color == old_color:
                        continue
                    colors[pos] = new_color
                    v = count_violations(colors)
                    if v < best_viol:
                        best_viol = v
                        best_new_color = new_color
                    colors[pos] = old_color
                
                colors[pos] = best_new_color
                
                if best_viol == 0:
                    break
            
            if best_viol == 0:
                # Valid coloring found with k colors
                # Now try to merge colors to reduce count
                current_k = k
                colors_copy = colors[:]
                
                # Try merging colors pairwise
                for merge_iter in range(100):
                    if time.time() > DEADLINE:
                        break
                    
                    used_colors = sorted(set(colors_copy))
                    if len(used_colors) <= 3:
                        break
                    
                    merged = False
                    # Try all pairs of colors to merge
                    for i in range(len(used_colors)):
                        for j in range(i + 1, len(used_colors)):
                            ci, cj = used_colors[i], used_colors[j]
                            # Try merging cj into ci
                            test_colors = [ci if c == cj else c for c in colors_copy]
                            if count_violations(test_colors) == 0:
                                colors_copy = test_colors
                                merged = True
                                break
                        if merged:
                            break
                    
                    if not merged:
                        break
                
                final_colors = colors_copy
                final_k = count_colors(final_colors)
                if final_k < best_colors_used:
                    best_colors_used = final_k
                    best_result = final_colors[:]
                
                # If we achieved 3 colors, that's optimal (can't do better than 3)
                if best_colors_used <= 3:
                    break
        
        if best_colors_used <= 3:
            break
    
    # If no valid coloring found, use a fallback
    if best_result is None:
        # Use the greedy seed approach as fallback
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                        for d in range(1, (x-1)//3 + 1))
                    and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                            for d in range(1, (x-1)//2 + 1))):
                    colors.append(color)
                    break
        best_result = colors
        best_colors_used = count_colors(best_result)
    
    return best_result
