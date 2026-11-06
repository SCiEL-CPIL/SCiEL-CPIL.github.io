import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Fallback: n=11 (Known valid: K11 decomposed into 5 Hamiltonian cycles)
    # Cycle i uses edges (j, j+i) mod 11 for j=0..10. 
    # For n=11, distance d in 1..5.
    # Edge (i, j) with i<j. dist = min(j-i, 11-(j-i)).
    # Color = dist - 1.
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback

    def get_idx(i, j, n):
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    def check_validity(colors, n):
        adj_mask = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                mask_u = adj_mask[c][u]
                if mask_u == 0:
                    continue
                for v in range(u + 1, n):
                    mask_v = adj_mask[c][v]
                    if mask_v == 0:
                        continue
                    common = mask_u & mask_v
                    if common:
                        if common.bit_count() > 1:
                            return False
        return True

    def count_violations(colors, n):
        adj_mask = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                idx += 1
        
        violations = 0
        for c in range(5):
            for u in range(n):
                mask_u = adj_mask[c][u]
                if mask_u == 0:
                    continue
                for v in range(u + 1, n):
                    mask_v = adj_mask[c][v]
                    if mask_v == 0:
                        continue
                    common = mask_u & mask_v
                    if common:
                        cnt = common.bit_count()
                        if cnt > 1:
                            violations += cnt - 1
        return violations

    def repair_local_search(colors, n, max_time):
        current_violations = count_violations(colors, n)
        if current_violations == 0:
            return colors, True
        
        num_edges = n * (n - 1) // 2
        inv_map = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        temp = 5.0
        cooling = 0.99
        min_temp = 0.01
        
        best_colors = colors[:]
        best_violations = current_violations
        
        while temp > min_temp and time.time() < max_time:
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
            
            colors[e_idx] = new_color
            new_violations = count_violations(colors, n)
            
            delta = new_violations - current_violations
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_violations = new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
            else:
                colors[e_idx] = old_color
            
            temp *= cooling
        
        if best_violations == 0:
            return best_colors, True
        return best_colors, False

    # Try n=12 first with many restarts.
    # If we succeed, try n=13, etc.
    
    # Strategy for initialization:
    # Random
    # Structured (e.g., based on distance in a cycle or grid)
    
    for n in range(12, 15):  # Try up to 14. 15 is very hard for random+SA in this time.
        if time.time() > deadline - 20.0:
            break
        
        success = False
        num_restarts = 10
        time_per_restart = (deadline - 20.0 - time.time()) / (num_restarts * (15 - n))
        
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Initialization strategy
            if r % 2 == 0:
                # Random
                colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
            else:
                # Structured: Use a cyclic pattern if n is odd, or similar for even.
                # For even n, a pure cycle distance doesn't work as well for 5 colors.
                # Let's try a simple grid-like or random with bias.
                # Actually, just random is often better for SA if SA is strong enough.
                # But let's try a "near-identity" pattern: color = (i+j) % 5 or similar.
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        colors.append((i + j) % 5)
            
            t_limit = time.time() + max(1.0, time_per_restart)
            result_colors, is_valid = repair_local_search(colors, n, t_limit)
            
            if is_valid:
                best_n = n
                best_colors = result_colors
                success = True
                break
        
        if not success:
            # If we fail for n, it's likely to fail for n+1 in the same time budget
            # unless we have a lot of time.
            # Given the strict time limit, breaking is safer.
            break

    return {'n': best_n, 'colors': best_colors}
