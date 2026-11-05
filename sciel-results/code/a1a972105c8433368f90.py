import time
import numpy as np
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # 1. Precompute all 4-cycles
    all_cycles = []
    vertices = list(range(N))
    
    def get_idx(i, j):
        if i > j: i, j = j, i
        return i * N - i * (i + 1) // 2 + (j - i - 1)

    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        all_cycles.append((get_idx(a,b), get_idx(b,c), get_idx(c,d), get_idx(d,a)))
        # Cycle 2: a-b-d-c-a
        all_cycles.append((get_idx(a,b), get_idx(b,d), get_idx(d,c), get_idx(c,a)))
        # Cycle 3: a-c-b-d-a
        all_cycles.append((get_idx(a,c), get_idx(c,b), get_idx(b,d), get_idx(d,a)))
        
    num_cycles = len(all_cycles)
    cycles_arr = np.array(all_cycles, dtype=np.int32) # shape (360, 4)

    # 2. Initial State: All zeros (1 color, 0 violations)
    current_colors = np.zeros(NUM_EDGES, dtype=np.int32)
    best_colors = current_colors.copy()
    best_score = 1
    best_violations = 0 # Should be 0 for valid

    # Track best valid state
    # We need a fast way to check if a state is valid and its score
    
    def evaluate(colors):
        """
        Returns (num_unique_colors, num_violations)
        """
        # Unique colors
        num_unique = len(np.unique(colors))
        
        # Violations: cycles with 4 distinct colors
        # Get colors for all cycles
        c0 = colors[cycles_arr[:, 0]]
        c1 = colors[cycles_arr[:, 1]]
        c2 = colors[cycles_arr[:, 2]]
        c3 = colors[cycles_arr[:, 3]]
        
        # A cycle is a violation if all 4 are distinct
        # (c0!=c1) & (c0!=c2) & (c0!=c3) & (c1!=c2) & (c1!=c3) & (c2!=c3)
        distinct_mask = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        num_violations = np.sum(distinct_mask)
        
        return num_unique, num_violations

    # Initial evaluation
    curr_score, curr_violations = evaluate(current_colors)
    
    # Simulated Annealing Parameters
    # We want to maximize score, minimize violations.
    # Combined objective for acceptance: 
    # If valid (violations=0), maximize score.
    # If invalid, we might accept if score is higher or if temp is high, 
    # but we only update BEST if valid.
    
    T = 100.0
    cooling = 0.9999
    T_min = 0.1
    
    rng = random.Random(42)
    
    steps = 0
    while time.time() < deadline:
        # Generate a move: change a random subset of edges
        # Subset size varies: usually 1-5, occasionally larger "kicks"
        if rng.random() < 0.05:
            # Kick: change 10-20 edges
            num_changes = rng.randint(10, 20)
        else:
            num_changes = rng.randint(1, 5)
            
        # Pick random edges to change
        # Using rng.sample is O(k log n) or O(k) with random choices? 
        # For small k, just picking random indices is fine, duplicates are OK (just overwrite)
        # But better to pick distinct edges for a "cleaner" move
        # For speed, let's just pick random indices with replacement
        edges_to_change = [rng.randint(0, NUM_EDGES - 1) for _ in range(num_changes)]
        
        # Save old values
        old_vals = current_colors[edges_to_change].copy()
        
        # Assign new random colors
        # Bias towards unused colors? 
        # Simple: random 0..119
        new_vals = np.array([rng.randint(0, MAX_COLORS - 1) for _ in range(num_changes)], dtype=np.int32)
        
        # Apply
        current_colors[edges_to_change] = new_vals
        
        # Evaluate new state
        new_score, new_violations = evaluate(current_colors)
        
        # Update Best if valid and better
        if new_violations == 0 and new_score > best_score:
            best_score = new_score
            best_colors = current_colors.copy()
            
        # Acceptance Criteria
        # We want to stay in the neighborhood of good states.
        # If new state is valid and score >= old score: Accept
        # If new state is valid and score < old score: Accept with prob exp((score_diff)/T)
        # If new state is invalid: Accept with prob exp((score_diff - penalty)/T)
        
        # Let's define a "fitness" for acceptance
        # fitness = score - lambda * violations
        # Lambda should be high enough to strongly prefer valid states, 
        # but allow some invalid states to be explored if score is much higher.
        # However, since we only keep valid bests, exploring invalid states is only useful 
        # if they are "close" to valid states with higher scores.
        
        old_fitness = curr_score - 10.0 * curr_violations
        new_fitness = new_score - 10.0 * new_violations
        
        delta_fitness = new_fitness - old_fitness
        
        if delta_fitness >= 0:
            # Always accept improvement or neutral
            pass
        else:
            # Accept with probability exp(delta / T)
            if rng.random() < np.exp(delta_fitness / T):
                pass
            else:
                # Reject: revert
                current_colors[edges_to_change] = old_vals
                # curr_score, curr_violations remain unchanged
                # T cooling still happens
                T = max(T_min, T * cooling)
                steps += 1
                continue
        
        # Update current state metrics
        curr_score = new_score
        curr_violations = new_violations
        
        # Cooling
        T = max(T_min, T * cooling)
        
        # Optional: If we are stuck at a low score for a long time, restart?
        # Or just let SA handle it.
        
        steps += 1
        
        # Safety check: if steps is huge, maybe break? 
        # The time check handles this.

    # Final Verification
    final_score, final_violations = evaluate(best_colors)
    if final_violations != 0:
        # This shouldn't happen if we only update best when valid, 
        # but just in case, return a known valid state (all zeros)
        return np.zeros(NUM_EDGES, dtype=np.int32).tolist()
        
    return best_colors.tolist()

if __name__ == "__main__":
    colors = construct()
    print(f"Max colors: {len(set(colors))}")
