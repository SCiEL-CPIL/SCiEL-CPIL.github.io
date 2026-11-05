import time
import random
import numpy as np

def construct():
    N = 128
    # Precompute all 4-term arithmetic progressions (0-indexed)
    aps = []
    for a in range(N):
        for d in range(1, (N - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < N:
                aps.append((a, b, c, e))
    
    def count_violations(colors):
        bad = 0
        for a, b, c, e in aps:
            cols = {colors[a], colors[b], colors[c], colors[e]}
            if len(cols) < 3:
                bad += 1
        return bad

    def solve_with_k(k, time_limit=10):
        start_time = time.time()
        # Random initialization
        colors = np.random.randint(0, k, N)
        current_bad = count_violations(colors)
        
        if current_bad == 0:
            return colors.tolist()
            
        best_colors = colors.copy()
        best_bad = current_bad
        
        # Simple Local Search
        while time.time() - start_time < time_limit:
            # Find a random bad AP
            # To be faster, we can just pick a random position and check its neighbors?
            # Or pick a random AP.
            if aps:
                idx = random.randint(0, len(aps) - 1)
                a, b, c, e = aps[idx]
                cols = {colors[a], colors[b], colors[c], colors[e]}
                
                if len(cols) >= 3:
                    continue # This AP is fine, try another (or skip)
                
                # This AP is bad. Try to fix it by changing one of its colors.
                # Pick a random position in the AP
                pos_idx = random.choice([a, b, c, e])
                old_color = colors[pos_idx]
                
                # Try changing to a random different color
                new_color = random.randint(0, k - 1)
                if new_color == old_color:
                    continue
                    
                colors[pos_idx] = new_color
                new_bad = count_violations(colors)
                
                if new_bad < best_bad:
                    best_bad = new_bad
                    best_colors = colors.copy()
                    if best_bad == 0:
                        return best_colors.tolist()
                else:
                    # Revert if it got worse (greedy hill climbing)
                    # Or accept with probability (Simulated Annealing)
                    # For simplicity, revert if worse, or keep if equal?
                    # Let's revert if worse to avoid wandering too much, 
                    # but allow equal moves to escape plateaus.
                    if new_bad > best_bad:
                        colors[pos_idx] = old_color
                    # If new_bad == best_bad, keep the move to explore
            else:
                break
                
        if best_bad == 0:
            return best_colors.tolist()
        return None

    # Try k=3
    for k in [3, 4, 5]:
        # Use a fraction of the total time for each k
        # Total budget ~ 80s. Give 20s to k=3, 20s to k=4, etc.
        result = solve_with_k(k, time_limit=20)
        if result is not None:
            return result
            
    # Fallback: Return a greedy result or random if all fail
    # This should not happen if k is large enough
    return list(np.random.randint(0, 6, N))
