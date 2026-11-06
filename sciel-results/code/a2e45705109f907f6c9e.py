import numpy as np
from itertools import combinations
import random
import time

def construct():
    # Target n=34
    n = 34
    
    # Precompute all triples and their indices
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute adjacency lists for faster violation checking
    # red_adj[t] = list of other triples that share exactly 2 vertices with t
    # If t is Red, and all other triples in red_adj[t] are Red, we have a Red K4.
    # Actually, a Red K4 is formed by 4 vertices. The 4 triples are the 4 choose 3 subsets.
    # Let's precompute for each 4-set, the 4 triples it contains.
    
    # For efficiency, we will maintain:
    # red_count[v4] = number of Red triples in the 4-set v4. If 4, it's a violation.
    # blue_count[v5] = number of Blue triples in the 5-set v5. If 10, it's a violation.
    
    # However, updating these counts on every flip is O(size of neighborhood).
    # Let's stick to a vectorized check using NumPy for the global state, 
    # but we need to be smart about which flips to try.
    
    # Better strategy:
    # 1. Represent colors as a numpy array of ints (0=Red, 1=Blue).
    # 2. Precompute a matrix M_red where M_red[i, j] = 1 if triple i and triple j 
    #    can be part of the same Red K4 (i.e., they share 2 vertices).
    #    Actually, a Red K4 exists if there are 4 triples $t_1, t_2, t_3, t_4$ such that 
    #    they are all Red and they correspond to the 4 triples of a 4-set.
    #    This is a 4-clique in the "triple intersection graph" where edges mean sharing 2 vertices?
    #    No, it's simpler to just check all 4-sets.
    
    # Let's use a hybrid approach:
    # 1. Initialize with a known good structure or random.
    # 2. Use a local search loop.
    # 3. In each step, find a random violation (Red K4 or Blue K5).
    # 4. Try flipping one of the triples in that violation to see if it reduces the total violation count.
    #    To make this fast, we need a fast way to count violations.
    
    # Fast Violation Count:
    # We can precompute for each 4-set, the 4 triples it contains.
    # Let's create a list of all 4-sets and their corresponding triple indices.
    # k4_sets = [( (t1_idx, t2_idx, t3_idx, t4_idx), (v1,v2,v3,v4) ) ...]
    
    k4_triple_indices = []
    for verts in combinations(range(n), 4):
        t_indices = [triple_to_idx[t] for t in combinations(verts, 3)]
        k4_triple_indices.append(tuple(t_indices))
        
    k5_triple_indices = []
    for verts in combinations(range(n), 5):
        t_indices = [triple_to_idx[t] for t in combinations(verts, 3)]
        k5_triple_indices.append(tuple(t_indices))
        
    # Vectorized check for violations:
    # red_violations: count of k4s where all 4 triples are Red (0)
    # blue_violations: count of k5s where all 10 triples are Blue (1)
    
    def count_violations(colors):
        # colors is a numpy array of 0s and 1s
        # Red K4: sum of 4 triples == 0
        # Blue K5: sum of 10 triples == 0 (if we treat Blue as 0? No, Blue is 1)
        # Let's define:
        # Red is 0. Blue is 1.
        # Red K4 violation: all 4 are 0.
        # Blue K5 violation: all 10 are 1.
        
        # For Red K4s:
        # k4_matrix is shape (num_k4s, 4). 
        # red_k4_counts = sum(k4_matrix == 0, axis=1)
        # violations_red = sum(red_k4_counts == 4)
        
        # For Blue K5s:
        # k5_matrix is shape (num_k5s, 10).
        # blue_k5_counts = sum(k5_matrix == 1, axis=1)
        # violations_blue = sum(blue_k5_counts == 10)
        
        # To make this fast, we precompute the matrices.
        return 0 # Placeholder, will be implemented in the loop with precomputed arrays
        
    # Precompute matrices for vectorized operations
    k4_matrix = np.array(k4_triple_indices, dtype=np.int32)
    k5_matrix = np.array(k5_triple_indices, dtype=np.int32)
    
    def get_violation_counts(colors):
        # colors: np.array of shape (num_triples,)
        # Red K4s:
        # We want to check if colors[k4_matrix] are all 0.
        # red_vals = colors[k4_matrix] # shape (num_k4s, 4)
        # red_viol = np.sum(np.all(red_vals == 0, axis=1))
        
        # Blue K5s:
        # blue_vals = colors[k5_matrix] # shape (num_k5s, 10)
        # blue_viol = np.sum(np.all(blue_vals == 1, axis=1))
        
        # This is memory heavy for n=34.
        # num_k4s = C(34,4) = 46376. 46376 * 4 * 4 bytes = ~740 KB. OK.
        # num_k5s = C(34,5) = 278256. 278256 * 10 * 4 bytes = ~11 MB. OK.
        
        red_vals = colors[k4_matrix]
        red_viol = np.sum(np.all(red_vals == 0, axis=1))
        
        blue_vals = colors[k5_matrix]
        blue_viol = np.sum(np.all(blue_vals == 1, axis=1))
        
        return red_viol, blue_viol

    # Initialization:
    # Use a random seed for reproducibility but allow variation.
    # A good start is important.
    # Let's try a "structured" random: 
    # Start with all Blue (1), then randomly set some to Red?
    # Or start with a known algebraic construction for n=32 and extend?
    # The F_2^5 construction has no Red K4s but many Blue K5s.
    # Let's initialize based on F_2^5 for the first 32 vertices, and random for 32,33.
    
    colors = np.ones(num_triples, dtype=np.int8) # Start all Blue
    
    # Apply F_2^5 rule for vertices < 32
    for a in range(32):
        for b in range(a+1, 32):
            for c in range(b+1, 32):
                if (a ^ b ^ c) == 0:
                    idx = triple_to_idx[(a,b,c)]
                    colors[idx] = 0 # Red
    
    # Local Search
    # We will iterate, finding a random violation and trying to fix it.
    
    start_time = time.time()
    max_time = 110 # Leave 10s margin
    
    # Initial violation count
    red_viol, blue_viol = get_violation_counts(colors)
    total_viol = red_viol + blue_viol
    
    if total_viol == 0:
         return {'n': n, 'colors': colors.tolist()}
         
    # Precompute neighbors for faster local updates?
    # For now, we'll use a simple strategy:
    # 1. Pick a random violation (Red K4 or Blue K5).
    # 2. Pick a random triple in that violation.
    # 3. Flip it.
    # 4. Calculate new violation count.
    # 5. If better, keep. If worse, revert.
    # 6. If stuck, try random flips or accept worse moves (Simulated Annealing).
    
    # To speed up "Calculate new violation count", we can't just recompute the whole thing (11MB array op) 5000 times.
    # We need an incremental update.
    
    # Incremental Update Logic:
    # When we flip triple t from c to 1-c:
    # It affects all K4s containing t.
    # It affects all K5s containing t.
    
    # Let's precompute:
    # k4s_containing[t] = list of indices in k4_matrix that contain t.
    # k5s_containing[t] = list of indices in k5_matrix that contain t.
    
    # This is expensive to build but worth it.
    
    k4s_containing = [[] for _ in range(num_triples)]
    for i, t_indices in enumerate(k4_triple_indices):
        for t in t_indices:
            k4s_containing[t].append(i)
            
    k5s_containing = [[] for _ in range(num_triples)]
    for i, t_indices in enumerate(k5_triple_indices):
        for t in t_indices:
            k5s_containing[t].append(i)
            
    # Convert to numpy arrays for fast access? Or just lists.
    # Lists are fine.
    
    # Current state of violations per K4/K5
    # red_k4_state[i] = number of Red triples in k4 i. (0 to 4)
    # blue_k5_state[i] = number of Blue triples in k5 i. (0 to 10)
    
    # Initialize these states
    red_k4_state = np.zeros(len(k4_triple_indices), dtype=np.int32)
    blue_k5_state = np.zeros(len(k5_triple_indices), dtype=np.int32)
    
    # We can compute initial states using the vectorized method, then extract.
    # But we need the counts, not just the violation boolean.
    
    # Let's compute initial counts manually or via numpy
    # red_k4_state = np.sum(colors[k4_matrix] == 0, axis=1)
    # blue_k5_state = np.sum(colors[k5_matrix] == 1, axis=1)
    
    red_k4_state = np.sum(colors[k4_matrix] == 0, axis=1)
    blue_k5_state = np.sum(colors[k5_matrix] == 1, axis=1)
    
    # Count violations
    red_viol = np.sum(red_k4_state == 4)
    blue_viol = np.sum(blue_k5_state == 10)
    total_viol = red_viol + blue_viol
    
    if total_viol == 0:
        return {'n': n, 'colors': colors.tolist()}
        
    # Local Search Loop
    # We will use a simple hill climbing with random restarts or random moves.
    
    for step in range(50000): # Max steps
        if time.time() - start_time > max_time:
            break
            
        if total_viol == 0:
            break
            
        # Pick a random violation
        # If red_viol > 0, pick a random Red K4.
        # If blue_viol > 0, pick a random Blue K5.
        # Balance: pick based on ratio.
        
        if red_viol > 0 and (blue_viol == 0 or random.random() < red_viol / (red_viol + blue_viol)):
            # Pick a random Red K4
            # We need a list of indices where red_k4_state == 4
            # To avoid scanning the whole array every time, we can maintain a list or just sample.
            # Sampling:
            while True:
                idx = random.randint(0, len(k4_triple_indices) - 1)
                if red_k4_state[idx] == 4:
                    break
                    
            # This K4 has 4 triples. Pick one to flip.
            t_indices = k4_triple_indices[idx]
            t_idx = random.choice(t_indices)
            # Flip t_idx from 0 (Red) to 1 (Blue)
            
            old_color = colors[t_idx]
            if old_color == 0:
                # Flip to Blue
                colors[t_idx] = 1
                
                # Update states
                # For each K4 containing t_idx:
                # It was Red (0), now Blue (1).
                # red_k4_state decreases by 1.
                # Note: If red_k4_state was 4, it becomes 3 (violation removed).
                # If it was 1, it becomes 0.
                
                # We need to track changes in total_viol.
                # delta_red_viol = 0
                # for k4_idx in k4s_containing[t_idx]:
                #    old_count = red_k4_state[k4_idx]
                #    new_count = old_count - 1
                #    red_k4_state[k4_idx] = new_count
                #    if old_count == 4 and new_count == 3: delta_red_viol -= 1
                #    if old_count == 0 and new_count == 1: delta_red_viol += 1 # Wait, 0->1 doesn't create violation.
                #    Violation is count==4.
                #    So:
                #    if old_count == 4: delta_red_viol -= 1
                #    if new_count == 4: delta_red_viol += 1 (Impossible since we decrement)
                
                # Similarly for Blue K5s.
                # t_idx was Red (0), now Blue (1).
                # blue_k5_state increases by 1.
                # if old_count == 9 and new_count == 10: delta_blue_viol += 1
                # if old_count == 10 and new_count == 11: delta_blue_viol -= 1 (Impossible since we increment, and max is 10)
                
                # Let's implement this carefully.
                
                delta_red = 0
                for k4_idx in k4s_containing[t_idx]:
                    old_c = red_k4_state[k4_idx]
                    new_c = old_c - 1
                    red_k4_state[k4_idx] = new_c
                    if old_c == 4:
                        delta_red -= 1
                    # new_c can't be 4 if old_c was 4->3.
                    
                delta_blue = 0
                for k5_idx in k5s_containing[t_idx]:
                    old_c = blue_k5_state[k5_idx]
                    new_c = old_c + 1
                    blue_k5_state[k5_idx] = new_c
                    if old_c == 9 and new_c == 10:
                        delta_blue += 1
                    # old_c can't be 10 because we are adding to it.
                    
                total_viol += delta_red + delta_blue
                
                # Check if this move was good
                # If total_viol decreased or stayed same, keep it.
                # If increased, revert?
                # For now, let's keep it if it doesn't increase? 
                # Actually, we should probably only keep if it decreases.
                # But if we are stuck, we might need to accept worse moves.
                # Let's use a simple rule: Keep if total_viol < previous_total.
                
                # We didn't store previous_total.
                # Let's assume we want to minimize total_viol.
                # If total_viol increased, we should revert.
                
                # Wait, I modified colors and states already.
                # I need to check if total_viol is lower than before the move.
                # I don't have "before_total".
                
                # Let's restructure:
                # 1. Calculate delta.
                # 2. If delta < 0, apply.
                # 3. If delta >= 0, don't apply (or apply with prob exp(-delta/T)).
                
                # Let's just apply if delta < 0.
                # If delta >= 0, revert.
                
                if delta_red + delta_blue >= 0:
                    # Revert
                    colors[t_idx] = 0
                    for k4_idx in k4s_containing[t_idx]:
                        red_k4_state[k4_idx] += 1
                    for k5_idx in k5s_containing[t_idx]:
                        blue_k5_state[k5_idx] -= 1
                    # total_viol unchanged
                    
            # End while loop for finding a valid K4
        else:
            # Pick a random Blue K5
            while True:
                idx = random.randint(0, len(k5_triple_indices) - 1)
                if blue_k5_state[idx] == 10:
                    break
                    
            t_indices = k5_triple_indices[idx]
            t_idx = random.choice(t_indices)
            # Flip t_idx from 1 (Blue) to 0 (Red)
            
            old_color = colors[t_idx]
            if old_color == 1:
                # Flip to Red
                colors[t_idx] = 0
                
                delta_red = 0
                for k4_idx in k4s_containing[t_idx]:
                    old_c = red_k4_state[k4_idx]
                    new_c = old_c + 1
                    red_k4_state[k4_idx] = new_c
                    if old_c == 3 and new_c == 4:
                        delta_red += 1
                        
                delta_blue = 0
                for k5_idx in k5s_containing[t_idx]:
                    old_c = blue_k5_state[k5_idx]
                    new_c = old_c - 1
                    blue_k5_state[k5_idx] = new_c
                    if old_c == 10:
                        delta_blue -= 1
                        
                if delta_red + delta_blue >= 0:
                    # Revert
                    colors[t_idx] = 1
                    for k4_idx in k4s_containing[t_idx]:
                        red_k4_state[k4_idx] -= 1
                    for k5_idx in k5s_containing[t_idx]:
                        blue_k5_state[k5_idx] += 1
                else:
                    total_viol += delta_red + delta_blue

    # Final Check
    red_viol, blue_viol = get_violation_counts(colors)
    if red_viol == 0 and blue_viol == 0:
        return {'n': n, 'colors': colors.tolist()}
    else:
        # Fallback to a smaller valid n if 34 fails
        # Try n=33?
        # For now, return a known valid small n to ensure non-zero score.
        # n=9 is valid with the seed logic.
        n_seed = 9
        return {'n': n_seed, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_seed), 3)]}
