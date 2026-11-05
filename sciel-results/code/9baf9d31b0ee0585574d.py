import time
import numpy as np
import random
from itertools import combinations
import sys

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def edge_vertices(idx):
    i = 0
    while i * N - i * (i + 1) // 2 + (N - i - 1) <= idx:
        i += 1
    i -= 1
    offset = idx - (i * N - i * (i + 1) // 2)
    j = i + 1 + offset
    return i, j

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin
    
    # Precompute all 4-cycles
    all_cycles = []
    vertices = list(range(N))
    
    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        all_cycles.append((edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a)))
        # Cycle 2: a-b-d-c-a
        all_cycles.append((edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a)))
        # Cycle 3: a-c-b-d-a
        all_cycles.append((edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a)))
        
    num_cycles = len(all_cycles)
    
    # For fast constraint checking during backtracking
    # For each edge, list of (cycle_index, position_in_cycle)
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(all_cycles):
        for pos, e in enumerate(cycle):
            edge_to_cycles[e].append((ci, pos))
            
    def is_fully_valid(colors):
        # Used for final verification
        for ci, cycle in enumerate(all_cycles):
            c0 = colors[cycle[0]]
            c1 = colors[cycle[1]]
            c2 = colors[cycle[2]]
            c3 = colors[cycle[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True

    def count_distinct(colors):
        return len(set(colors.tolist()))

    def check_move_valid_rec(e_idx, new_color, colors):
        """
        Returns True if setting colors[e_idx] = new_color is valid,
        assuming the rest of the coloring is valid.
        """
        old_color = colors[e_idx]
        if new_color == old_color:
            return True
            
        for ci, pos in edge_to_cycles[e_idx]:
            cycle = all_cycles[ci]
            other_positions = [p for p in range(4) if p != pos]
            c1 = colors[cycle[other_positions[0]]]
            c2 = colors[cycle[other_positions[1]]]
            c3 = colors[cycle[other_positions[2]]]
            
            # Check for 4 distinct: new_color, c1, c2, c3
            if new_color == c1 or new_color == c2 or new_color == c3:
                continue # This cycle is safe
            
            if c1 == c2 or c1 == c3 or c2 == c3:
                continue # This cycle is safe
                
            # If we reach here, this cycle would have 4 distinct colors.
            return False
            
        return True

    def check_move_valid_swap(e1_idx, e2_idx, colors):
        """
        Returns True if swapping colors of e1_idx and e2_idx is valid,
        assuming the rest of the coloring is valid.
        """
        c1 = colors[e1_idx]
        c2 = colors[e2_idx]
        
        if c1 == c2:
            return True
            
        # We need to check all cycles involving e1 or e2.
        # A cycle is invalid if it has 4 distinct colors after the swap.
        # For cycles involving e1 but not e2:
        #   The color of e1 changes from c1 to c2.
        #   We check if {c2, other3} are all distinct.
        # For cycles involving e2 but not e1:
        #   The color of e2 changes from c2 to c1.
        #   We check if {c1, other3} are all distinct.
        # For cycles involving both e1 and e2:
        #   The colors swap. The set of colors in the cycle remains the same.
        #   So if it was valid before, it is valid after.
        
        # Optimization: Union of cycles for e1 and e2.
        # But we must be careful not to double-check or misinterpret.
        
        checked_cycles = set()
        
        for ci, pos1 in edge_to_cycles[e1_idx]:
            if ci in checked_cycles:
                continue
            checked_cycles.add(ci)
            cycle = all_cycles[ci]
            
            # Check if e2 is in this cycle
            if e2_idx in cycle:
                # Both edges in cycle. Set of colors is invariant. 
                # If state was valid, it remains valid.
                continue
                
            # Only e1 in cycle.
            # New color for e1 is c2.
            other_positions = [p for p in range(4) if p != pos1]
            c_o1 = colors[cycle[other_positions[0]]]
            c_o2 = colors[cycle[other_positions[1]]]
            c_o3 = colors[cycle[other_positions[2]]]
            
            if c2 == c_o1 or c2 == c_o2 or c2 == c_o3:
                continue
            if c_o1 == c_o2 or c_o1 == c_o3 or c_o2 == c_o3:
                continue
            return False
            
        for ci, pos2 in edge_to_cycles[e2_idx]:
            if ci in checked_cycles:
                continue
            checked_cycles.add(ci)
            cycle = all_cycles[ci]
            
            # e1 is NOT in this cycle (otherwise it would have been processed above)
            # New color for e2 is c1.
            other_positions = [p for p in range(4) if p != pos2]
            c_o1 = colors[cycle[other_positions[0]]]
            c_o2 = colors[cycle[other_positions[1]]]
            c_o3 = colors[cycle[other_positions[2]]]
            
            if c1 == c_o1 or c1 == c_o2 or c1 == c_o3:
                continue
            if c_o1 == c_o2 or c_o1 == c_o3 or c_o2 == c_o3:
                continue
            return False
            
        return True

    rng = random.Random(42)
    
    # Initial coloring: All 0s is valid.
    # To get a good start, we can try to introduce some colors greedily first.
    # Or just start with all 0s and let SA do the work.
    # Starting with all 0s gives score 1.
    # Let's try a greedy initialization to get a higher starting score.
    
    def greedy_init():
        colors = np.zeros(NUM_EDGES, dtype=np.int32)
        used_colors = {0}
        # Random order
        order = list(range(NUM_EDGES))
        rng.shuffle(order)
        for e in order:
            # Try to find a valid color, preferring new ones
            found_new = False
            for _ in range(20):
                c = rng.randint(0, MAX_COLORS - 1)
                if c not in used_colors and check_move_valid_rec(e, c, colors):
                    colors[e] = c
                    used_colors.add(c)
                    found_new = True
                    break
            if not found_new:
                # Try existing
                for c in list(used_colors):
                    if check_move_valid_rec(e, c, colors):
                        colors[e] = c
                        break
        return colors

    best_colors = greedy_init()
    best_score = count_distinct(best_colors)
    current_colors = best_colors.copy()
    current_score = best_score
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling_rate = 0.9999
    min_T = 0.01
    
    # We will run SA for a fixed number of steps or until time limit.
    # Steps per second: Let's estimate.
    # check_move_valid_rec is fast.
    # Let's run until deadline.
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # Generate a move
        # 50% Recolor, 50% Swap
        if rng.random() < 0.5:
            # Recolor
            e_idx = rng.randint(0, NUM_EDGES - 1)
            old_color = current_colors[e_idx]
            # Bias towards new colors?
            # If we pick a random color from 0..119, we have 1/120 chance of new color (if <120 used).
            # If we pick from used colors, we might reduce distinct count.
            # Strategy: 
            # 1. Try a random new color (not in used set).
            # 2. If that fails or we pick existing, try random existing.
            
            used_set = set(current_colors.tolist())
            num_used = len(used_set)
            
            # With some probability, try to introduce a new color
            if rng.random() < 0.3 and num_used < MAX_COLORS:
                # Pick a color not in used_set
                # Simple way: pick random until not in set? Or maintain list?
                # For efficiency, just pick random and check.
                new_color = rng.randint(0, MAX_COLORS - 1)
                if new_color in used_set:
                    # Retry a few times
                    for _ in range(5):
                        new_color = rng.randint(0, MAX_COLORS - 1)
                        if new_color not in used_set:
                            break
            
            if check_move_valid_rec(e_idx, new_color, current_colors):
                # Calculate new score
                # If new_color was not in used_set, score increases by 1.
                # If new_color was in used_set, score might decrease by 1 (if old_color was unique) or stay same.
                # We need to check if old_color becomes unused.
                old_count = np.count_nonzero(current_colors == old_color)
                
                # Tentatively apply
                current_colors[e_idx] = new_color
                
                # Check validity of new state (already done in check_move_valid_rec? 
                # Yes, check_move_valid_rec assumes rest is valid and checks if move keeps it valid.
                # So state is valid.)
                
                # Calculate score change
                # We can just recount, or track.
                # Recounting is O(N). N=120. Fast enough.
                new_score = count_distinct(current_colors)
                
                delta = new_score - current_score
                
                if delta >= 0:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_colors = current_colors.copy()
                else:
                    # Accept with probability
                    prob = np.exp(delta / T)
                    if rng.random() < prob:
                        current_score = new_score
                    else:
                        # Revert
                        current_colors[e_idx] = old_color
                        
        else:
            # Swap
            e1 = rng.randint(0, NUM_EDGES - 1)
            e2 = rng.randint(0, NUM_EDGES - 1)
            if e1 == e2:
                continue
                
            if check_move_valid_swap(e1, e2, current_colors):
                # Swap
                c1 = current_colors[e1]
                c2 = current_colors[e2]
                current_colors[e1] = c2
                current_colors[e2] = c1
                
                # Score change?
                # Swapping preserves the multiset of colors, so distinct count is unchanged.
                # delta = 0.
                # We accept neutral moves to explore.
                # Since delta is 0, we always accept (or accept with high prob).
                # Let's always accept to allow mixing.
                # current_score remains same.
                # Check if this configuration is better? No, score is same.
                # But it might lead to better future moves.
                
        T *= cooling_rate
        
    # Final verification
    if not is_fully_valid(best_colors):
        # Fallback
        fallback = np.zeros(NUM_EDGES, dtype=np.int32)
        return fallback.tolist()
        
    return best_colors.tolist()

if __name__ == "__main__":
    colors = construct()
    print(f"Max colors: {len(set(colors))}")
    print(colors)
