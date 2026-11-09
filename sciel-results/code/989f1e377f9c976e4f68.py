import numpy as np
import random
import time

N = 64
TOTAL_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    # 1. Precompute distances
    xs = np.arange(N)
    ys = np.arange(N)
    XX, YY = np.meshgrid(xs, ys)
    coords = np.column_stack((XX.ravel(), YY.ravel()))
    
    # dist2[i, j] is squared distance between point i and point j
    # Using int32 to save memory and speed up
    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    dist2 = (dx**2 + dy**2).astype(np.int32)
    
    # 2. Initial Greedy Construction
    # Try to build a valid set greedily
    S = []
    S_set = set()
    unused = set(range(TOTAL_POINTS))
    
    random.seed(42)
    
    def get_conflicts(p, S_arr):
        if len(S_arr) == 0:
            return 0
        d_pu = dist2[p, S_arr]
        d_uu = dist2[S_arr[:, None], S_arr[None, :]]
        
        # Count 1: d(p, u_i) == d(p, u_j)
        c1 = len(d_pu) - len(np.unique(d_pu))
        
        # Count 2: d(p, u_i) == d(u_i, u_j)
        # d_pu is (K,), d_uu is (K,K)
        # d_pu[:, None] is (K,1), d_uu is (K,K) -> (K,K)
        c2 = np.sum(d_pu[:, None] == d_uu)
        
        # Count 3: d(p, u_j) == d(u_i, u_j)
        # d_pu[None, :] is (1,K), d_uu is (K,K) -> (K,K)
        c3 = np.sum(d_pu[None, :] == d_uu)
        
        return c1 + c2 + c3

    # Greedy phase
    while unused and time.time() < start_time + 10:
        p = random.choice(list(unused))
        S_arr = np.array(S)
        conflicts = get_conflicts(p, S_arr)
        if conflicts == 0:
            S.append(p)
            S_set.add(p)
            unused.remove(p)
            
    best_S = S[:]
    best_size = len(S)
    
    # 3. Simulated Annealing
    # We will try to grow the set or swap to allow growth
    # State: S (list), S_set (set), unused (set)
    # We track current_conflicts (should be 0 for valid states)
    
    T = 10.0
    cooling_rate = 0.999
    
    # To speed up, we can limit the number of SA steps
    max_steps = 50000
    step = 0
    
    current_S = S[:]
    current_S_set = S_set.copy()
    current_unused = unused.copy()
    
    # Ensure current state is valid
    current_conflicts = 0
    
    while time.time() < deadline and step < max_steps:
        step += 1
        
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.5 and current_unused:
            # Add a point
            p = random.choice(list(current_unused))
            S_arr = np.array(current_S)
            new_conflicts = get_conflicts(p, S_arr)
            delta_E = new_conflicts
            
            if delta_E < 0 or (delta_E > 0 and random.random() < np.exp(-delta_E / T)):
                # Accept
                current_S.append(p)
                current_S_set.add(p)
                current_unused.remove(p)
                current_conflicts += delta_E
                
                if current_conflicts == 0 and len(current_S) > best_size:
                    best_S = current_S[:]
                    best_size = len(current_S)
                    
        elif move_type < 0.8 and current_S:
            # Remove a point
            p = random.choice(current_S)
            S_arr = np.array(current_S)
            # Calculate conflicts removed by p
            # This is the same as get_conflicts(p, S \ {p})
            # But we need to be careful.
            # Let S' = S \ {p}
            # S_arr_without_p = S_arr[S_arr != p]
            # conflicts_removed = get_conflicts(p, S_arr_without_p)
            
            mask = S_arr != p
            S_arr_wo_p = S_arr[mask]
            conflicts_removed = get_conflicts(p, S_arr_wo_p)
            
            delta_E = -conflicts_removed
            
            if delta_E < 0 or (delta_E > 0 and random.random() < np.exp(-delta_E / T)):
                # Accept
                current_S.remove(p)
                current_S_set.remove(p)
                current_unused.add(p)
                current_conflicts += delta_E
                
                if current_conflicts == 0 and len(current_S) > best_size:
                    best_S = current_S[:]
                    best_size = len(current_S)
                    
        else:
            # Swap: Remove p_in, Add p_out
            if current_S and current_unused:
                p_in = random.choice(current_S)
                p_out = random.choice(list(current_unused))
                
                S_arr = np.array(current_S)
                mask = S_arr != p_in
                S_arr_wo_in = S_arr[mask]
                
                # Conflicts in S
                # Remove p_in: reduces conflicts by C_in
                C_in = get_conflicts(p_in, S_arr_wo_in)
                
                # Add p_out to S_wo_in: adds conflicts C_out
                C_out = get_conflicts(p_out, S_arr_wo_in)
                
                delta_E = C_out - C_in
                
                if delta_E < 0 or (delta_E > 0 and random.random() < np.exp(-delta_E / T)):
                    # Accept
                    current_S.remove(p_in)
                    current_S_set.remove(p_in)
                    current_unused.add(p_in)
                    
                    current_S.append(p_out)
                    current_S_set.add(p_out)
                    current_unused.remove(p_out)
                    
                    current_conflicts += delta_E
                    
                    if current_conflicts == 0 and len(current_S) > best_size:
                        best_S = current_S[:]
                        best_size = len(current_S)
                        
        # Cool down
        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    # Final check: ensure best_S is valid
    # The logic above only updates best_S if current_conflicts == 0
    # So best_S should be valid.
    
    return [(x // N, x % N) for x in best_S]

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We need to wrap it or just define it here.
# The prompt asks to keep function names.
# I will define construct() in the block.
# EVOLVE-BLOCK-END
