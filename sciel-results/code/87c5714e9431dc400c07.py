import numpy as np
import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge indices for quick access
# edge_idx[i, j] gives the index in the flat color array for edge (i, j)
edge_idx = np.zeros((N, N), dtype=np.int32)
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edge_idx[i, j] = idx
        edge_idx[j, i] = idx
        idx += 1

# Precompute all 4-cycles in K_16
# A 4-cycle is defined by 4 distinct vertices {a, b, c, d}.
# The three possible cycles on these vertices are:
# 1. a-b-c-d-a (edges: ab, bc, cd, da)
# 2. a-b-d-c-a (edges: ab, bd, dc, ca)
# 3. a-c-b-d-a (edges: ac, cb, bd, da)
# We store these as tuples of 4 edge indices.
cycles = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b-c-d
                cycles.append((edge_idx[a, b], edge_idx[b, c], edge_idx[c, d], edge_idx[d, a]))
                # Cycle 2: a-b-d-c
                cycles.append((edge_idx[a, b], edge_idx[b, d], edge_idx[d, c], edge_idx[c, a]))
                # Cycle 3: a-c-b-d
                cycles.append((edge_idx[a, c], edge_idx[c, b], edge_idx[b, d], edge_idx[d, a]))

# Convert to numpy array for vectorized operations
cycles_arr = np.array(cycles, dtype=np.int32)
NUM_CYCLES = len(cycles_arr)

def count_violations_and_colors(colors):
    """
    Vectorized check for rainbow 4-cycles and count of distinct colors.
    colors: np.array of shape (120,)
    Returns: (num_violations, num_distinct_colors)
    """
    # Get colors of all edges in all cycles
    # shape: (NUM_CYCLES, 4)
    cycle_colors = colors[cycles_arr]
    
    # Check if all 4 colors in a cycle are distinct
    # A cycle is a violation if min != max OR (max == min is false and ... wait)
    # Actually, 4 distinct colors means:
    # len(set(row)) == 4
    # Efficient vectorized way:
    # Sort the row. If sorted[row] has all unique elements, it's a violation.
    # Unique elements in a sorted row of length 4:
    # c0 != c1 AND c1 != c2 AND c2 != c3
    sorted_colors = np.sort(cycle_colors, axis=1)
    c0 = sorted_colors[:, 0]
    c1 = sorted_colors[:, 1]
    c2 = sorted_colors[:, 2]
    c3 = sorted_colors[:, 3]
    
    is_rainbow = (c0 != c1) & (c1 != c2) & (c2 != c3)
    num_violations = np.sum(is_rainbow)
    
    # Count distinct colors
    # Since colors are integers 0..119, we can use a histogram or np.unique
    # np.unique is generally fast for small arrays
    num_distinct = len(np.unique(colors))
    
    return int(num_violations), int(num_distinct)

def construct():
    start_time = time.time()
    time_limit = 50.0  # Safety margin for 60s limit
    
    # Initial state: Random assignment of colors
    # We want to maximize distinct colors, so start with a diverse random assignment
    # but ensure it's not too chaotic.
    # A good start might be to assign random colors from a subset, or just 0..119 randomly.
    # Let's try a structured start that is valid or near-valid, then perturb.
    # However, random start with SA is robust.
    
    best_colors = None
    best_score = -1
    best_violations = float('inf')
    
    # Run multiple restarts if time permits
    while time.time() - start_time < time_limit:
        # Initialize
        # Strategy: Start with a "block" coloring or random?
        # Random is fine for SA.
        colors = np.random.randint(0, 120, size=NUM_EDGES, dtype=np.int32)
        
        # Initial evaluation
        violations, distinct = count_violations_and_colors(colors)
        
        # If we start with violations, we need to fix them.
        # SA Objective: Minimize Violations, then Maximize Distinct.
        # Combined Score: -violations * 1000 + distinct
        
        current_violations = violations
        current_distinct = distinct
        current_score = -current_violations * 1000 + current_distinct
        
        if current_violations < best_violations or (current_violations == best_violations and current_distinct > best_score):
            best_colors = colors.copy()
            best_score = current_distinct
            best_violations = current_violations
            
        # If we found a valid solution, try to improve distinct colors
        # If not, try to reduce violations
        
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.999
        
        # Limit iterations per restart to avoid long single runs if stuck
        max_iters = 20000
        iters = 0
        
        while iters < max_iters:
            if time.time() - start_time > time_limit - 2.0:
                break
                
            # Pick a random edge to change
            edge = np.random.randint(0, NUM_EDGES)
            old_color = colors[edge]
            
            # Pick a new color
            # Bias towards using existing colors or new ones?
            # To maximize distinct, we should encourage new colors.
            # But new colors increase violation risk.
            # Let's pick uniformly from 0..119.
            new_color = np.random.randint(0, 120)
            
            if new_color == old_color:
                continue
                
            # Apply change
            colors[edge] = new_color
            
            # Evaluate delta
            # Recalculating full violations is O(NUM_CYCLES) = 2080 operations.
            # With NumPy, this is very fast.
            new_violations, new_distinct = count_violations_and_colors(colors)
            
            new_score = -new_violations * 1000 + new_distinct
            delta_score = new_score - current_score
            
            # Acceptance probability
            if delta_score > 0 or random.random() < np.exp(delta_score / T):
                # Accept
                current_violations = new_violations
                current_distinct = new_distinct
                current_score = new_score
                
                # Update best
                if current_violations < best_violations or (current_violations == best_violations and current_distinct > best_score):
                    best_colors = colors.copy()
                    best_score = current_distinct
                    best_violations = current_violations
                    
                    # If we have 0 violations, we can potentially cool faster
                    if best_violations == 0:
                         T *= 0.99 
                else:
                    # Reject
                    colors[edge] = old_color
                    
            else:
                # Reject
                colors[edge] = old_color
                
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
            iters += 1
            
            # Early termination if we've reached a "good" state and T is low
            # e.g., 0 violations and high distinct colors
            if best_violations == 0 and best_score >= 30:
                 # Check if we should restart to find better
                 if random.random() < 0.01:
                     break

    # Return the best valid solution found
    # If best_violations > 0, we must return something valid?
    # The task says "No four-cycle may have four distinct edge colors."
    # An invalid result scores 0.
    # We must ensure the returned result is valid.
    # If best_violations > 0, we need a fallback valid solution.
    
    if best_violations > 0:
        # Fallback: Use the simple valid solution color(i,j) = i
        # This guarantees 15 colors and 0 violations.
        fallback = np.zeros(NUM_EDGES, dtype=np.int32)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                fallback[idx] = i
                idx += 1
        return fallback.tolist()
    
    return best_colors.tolist()

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()
# EVOLVE-BLOCK-END

# To keep the structure clean and match the "EVOLVE-BLOCK" requirement strictly if needed,
# I will place the main logic inside construct or a helper called by construct.
# The prompt asks to keep function names and signatures.
# The seed had construct() inside the block.
# I will refactor to put everything inside construct or define helpers outside if allowed?
# "Keep the function names and signatures of the seed program."
# Seed only had construct.
# I can define helpers outside, but usually it's safer to keep it self-contained or just use global helpers.
# Let's move the logic into construct to be safe, or just define helpers above.
# The block usually contains the code to be evolved.
# I will put the helper functions in the global scope (outside block) and construct inside.

def _construct_impl():
    start_time = time.time()
    time_limit = 50.0  # Safety margin for 60s limit
    
    best_colors = None
    best_score = -1
    best_violations = float('inf')
    
    # Run multiple restarts if time permits
    while time.time() - start_time < time_limit:
        # Initialize
        colors = np.random.randint(0, 120, size=NUM_EDGES, dtype=np.int32)
        
        # Initial evaluation
        violations, distinct = count_violations_and_colors(colors)
        
        current_violations = violations
        current_distinct = distinct
        current_score = -current_violations * 1000 + current_distinct
        
        if current_violations < best_violations or (current_violations == best_violations and current_distinct > best_score):
            best_colors = colors.copy()
            best_score = current_distinct
            best_violations = current_violations
            
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.999
        
        max_iters = 20000
        iters = 0
        
        while iters < max_iters:
            if time.time() - start_time > time_limit - 2.0:
                break
                
            edge = np.random.randint(0, NUM_EDGES)
            old_color = colors[edge]
            new_color = np.random.randint(0, 120)
            
            if new_color == old_color:
                continue
                
            colors[edge] = new_color
            
            new_violations, new_distinct = count_violations_and_colors(colors)
            
            new_score = -new_violations * 1000 + new_distinct
            delta_score = new_score - current_score
            
            if delta_score > 0 or random.random() < np.exp(delta_score / T):
                current_violations = new_violations
                current_distinct = new_distinct
                current_score = new_score
                
                if current_violations < best_violations or (current_violations == best_violations and current_distinct > best_score):
                    best_colors = colors.copy()
                    best_score = current_distinct
                    best_violations = current_violations
                    
                    if best_violations == 0:
                         T *= 0.99 
            else:
                colors[edge] = old_color
                
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
            iters += 1
            
            if best_violations == 0 and best_score >= 30:
                 if random.random() < 0.01:
                     break

    if best_violations > 0:
        fallback = np.zeros(NUM_EDGES, dtype=np.int32)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                fallback[idx] = i
                idx += 1
        return fallback.tolist()
    
    return best_colors.tolist()

# The actual construct function required by the interface
def construct():
    return _construct_impl()
