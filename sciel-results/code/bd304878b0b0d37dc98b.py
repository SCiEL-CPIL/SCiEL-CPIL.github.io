import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # We know n=26 is possible. Try to find it.
    # If we can't find it quickly, fall back to smaller n.
    target_ns = [26, 25, 24, 23, 22, 21, 20]
    
    best_result = None
    
    for n in target_ns:
        if time.time() > deadline:
            break
            
        # Allocate time for this n, but ensure we don't exceed global deadline
        time_spent = time.time() - start_time
        remaining = deadline - time_spent
        # Give more time to larger n initially
        if n == 26:
            time_limit = min(remaining, 60.0)
        elif n == 25:
            time_limit = min(remaining, 30.0)
        else:
            time_limit = min(remaining, 15.0)
            
        result = local_search_tabu(n, time_limit)
        if result is not None:
            best_result = result
            break
            
    if best_result is None:
        # Fallback: Return a trivially valid small n
        # n=11 coloring: K11 split into 5 Hamiltonian cycles by circular distance
        best_result = {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}
        
    return best_result

def local_search_tabu(n, time_limit=60):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute edge indices for quick lookup
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    def initialize():
        # Random initialization
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        neighbors = [[0] * n for _ in range(5)]
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return colors, neighbors, violations

    def calc_delta(neighbors, e_idx, old_color, new_color):
        """
        Calculate the change in violations when changing edge e_idx from old_color to new_color.
        """
        if old_color == new_color:
            return 0
            
        i, j = edges[e_idx]
        delta = 0
        
        # 1. Remove from old_color
        c = old_color
        # Removing j from N_c(i)
        # For each v in N_c(j) \ {i}, j was a common neighbor of i and v.
        # k_iv decreases by 1. Change in binom(k,2) is -(k-1).
        mask_j = neighbors[c][j]
        v = 0
        m = mask_j
        while m:
            if m & 1:
                if v != i:
                    k = (neighbors[c][i] & neighbors[c][v]).bit_count()
                    delta -= (k - 1)
            m >>= 1
            v += 1
            
        # Removing i from N_c(j)
        # For each v in N_c(i) \ {j}, i was a common neighbor of j and v.
        # k_jv decreases by 1. Change is -(k-1).
        mask_i = neighbors[c][i]
        v = 0
        m = mask_i
        while m:
            if m & 1:
                if v != j:
                    k = (neighbors[c][j] & neighbors[c][v]).bit_count()
                    delta -= (k - 1)
            m >>= 1
            v += 1
            
        # Apply removal to masks
        neighbors[c][i] &= ~(1 << j)
        neighbors[c][j] &= ~(1 << i)
        
        # 2. Add to new_color
        c = new_color
        # Adding j to N_c(i)
        # For each v in N_c(j) \ {i}, j becomes a common neighbor of i and v.
        # k_iv increases by 1. Change is k (current count before adding).
        mask_j = neighbors[c][j]
        v = 0
        m = mask_j
        while m:
            if m & 1:
                if v != i:
                    k = (neighbors[c][i] & neighbors[c][v]).bit_count()
                    delta += k
            m >>= 1
            v += 1
            
        # Adding i to N_c(j)
        # For each v in N_c(i) \ {j}, i becomes a common neighbor of j and v.
        # k_jv increases by 1. Change is k.
        mask_i = neighbors[c][i]
        v = 0
        m = mask_i
        while m:
            if m & 1:
                if v != j:
                    k = (neighbors[c][j] & neighbors[c][v]).bit_count()
                    delta += k
            m >>= 1
            v += 1
            
        # Apply addition to masks
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
        
        return delta

    def revert(neighbors, e_idx, old_color, new_color):
        """Revert the mask changes (colors array is handled outside)."""
        i, j = edges[e_idx]
        # Revert new_color addition
        c = new_color
        neighbors[c][i] &= ~(1 << j)
        neighbors[c][j] &= ~(1 << i)
        
        # Revert old_color removal (i.e., add back)
        c = old_color
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)

    start_time = time.time()
    best_violations = float('inf')
    best_colors = None
    
    # Tabu Search parameters
    tabu_size = 50
    max_no_improve = 1000
    restarts = 0
    
    while time.time() < start_time + time_limit:
        # Initialize
        colors, neighbors, current_violations = initialize()
        
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            if best_violations == 0:
                return {'n': n, 'colors': colors}
        
        # Tabu list: store (edge_idx, color) pairs that were recently changed
        # We store the edge index and the color it was CHANGED TO.
        # Tabu if we try to change it back to the color it was CHANGED FROM?
        # Standard Tabu: store the move made. A move is (e_idx, old_color, new_color).
        # Tabu if we try to make the reverse move (e_idx, new_color, old_color).
        tabu_list = [] # List of (e_idx, target_color) where target_color is the color we just moved TO.
        # Actually, simpler: store (e_idx, color) that is tabu for a certain duration.
        # If we change e_idx to color c, then e_idx is tabu for color c? No, that prevents staying.
        # We want to prevent reversing.
        # Let's store (e_idx, old_color). If we try to set e_idx to old_color, it's tabu.
        
        no_improve_count = 0
        local_best_violations = current_violations
        local_best_colors = colors[:]
        
        while no_improve_count < max_no_improve and time.time() < start_time + time_limit:
            # Select a move
            # Strategy: Try all possible moves for a random edge? Or pick a random edge and try all colors?
            # For efficiency, pick a random edge, try all 5 colors, pick the best non-tabu one.
            # If no non-tabu move is improving, pick the best tabu move (aspiration criterion).
            
            e_idx = random.randint(0, num_edges - 1)
            i, j = edges[e_idx]
            old_color = colors[e_idx]
            
            best_delta = float('inf')
            best_new_color = -1
            best_is_tabu = True
            best_tabu_move_delta = float('inf')
            best_tabu_move_color = -1
            
            for new_color in range(5):
                if new_color == old_color:
                    continue
                
                # Check if tabu
                is_tabu = False
                # Tabu if we recently changed this edge TO this color?
                # No, tabu if we recently changed this edge FROM this color?
                # Let's say: if we changed e_idx to new_color recently, we are tabu from changing it BACK to old_color?
                # Actually, the standard is: a move is tabu if it is the reverse of a recent move.
                # Recent move: e_idx changed from old_c to new_c.
                # Reverse move: e_idx changed from new_c to old_c.
                # So if we are currently at old_color, and we want to go to new_color,
                # is this move tabu? It is tabu if we recently went from new_color to old_color.
                
                # Let's store (e_idx, color_after_change) in tabu_list.
                # If (e_idx, new_color) is in tabu_list, it means we recently changed e_idx TO new_color.
                # That doesn't prevent us from changing it to new_color again.
                # We want to prevent cycling.
                # If we changed e_idx to new_color, we should avoid changing it BACK to old_color immediately.
                # So, if we are at old_color and want to go to new_color,
                # we check if we recently changed e_idx FROM new_color TO old_color.
                # This is getting complicated to track exactly.
                
                # Simpler Tabu:
                # Store (e_idx, new_color) in tabu_list.
                # A move (e_idx, old_color -> new_color) is tabu if (e_idx, old_color) is in tabu_list?
                # No.
                # Let's use: Tabu if we changed e_idx to new_color in the last K moves.
                # This prevents immediate reversal if we just changed it to new_color, 
                # but wait, if we just changed it to new_color, we are AT new_color.
                # We are proposing to change FROM old_color TO new_color.
                # If we just changed e_idx to new_color, we are not at old_color anymore.
                
                # Correct Tabu Logic:
                # Move: e_idx, c_from -> c_to.
                # Tabu if (e_idx, c_from) was the c_to of a recent move?
                # i.e., if we recently did e_idx: c_to -> c_from, then e_idx: c_from -> c_to is tabu.
                # So we store (e_idx, c_from) in tabu_list? No, we store the move.
                # Let's store (e_idx, c_from) as "do not change e_idx FROM c_from".
                # If we are at c_from and want to go to c_to, check if (e_idx, c_from) is tabu.
                
                # Implementation:
                # tabu_list contains (e_idx, color) pairs.
                # Meaning: "Do not change edge e_idx FROM color".
                # When we make a move e_idx: c_old -> c_new, we add (e_idx, c_old) to tabu_list.
                
                is_tabu = (e_idx, old_color) in tabu_list
                
                # Calculate delta
                # We need to be careful: calc_delta modifies neighbors.
                # We need to calculate delta without modifying, or modify and revert.
                # For efficiency in selection, we can calculate delta for each color.
                # But calc_delta modifies neighbors.
                # So we must revert after calculating delta for each candidate.
                
                # This is expensive. 5 candidates * (calc + revert) per edge.
                # Let's just pick one random new_color and evaluate it.
                # This is a "random walk" with tabu.
                
                pass # We'll use a simpler strategy below

            # Simpler Strategy: Random Edge, Random New Color, Check Tabu
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
                
            is_tabu = (e_idx, old_color) in tabu_list
            
            if is_tabu:
                # Only allow if it's an aspiration move (improves best)
                # We need delta to know if it improves.
                # Calculate delta
                delta = calc_delta(neighbors, e_idx, old_color, new_color)
                new_violations = current_violations + delta
                if new_violations < local_best_violations:
                    # Aspiration: allow tabu move
                    is_tabu = False
                else:
                    continue # Skip tabu move
            
            if not is_tabu:
                # Calculate delta
                delta = calc_delta(neighbors, e_idx, old_color, new_color)
                new_violations = current_violations + delta
                
                # Accept move
                # Update tabu list
                tabu_list.append((e_idx, old_color))
                if len(tabu_list) > tabu_size:
                    tabu_list.pop(0)
                
                # Update state
                colors[e_idx] = new_color
                current_violations = new_violations
                
                if current_violations < local_best_violations:
                    local_best_violations = current_violations
                    local_best_colors = colors[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return {'n': n, 'colors': colors}
        
        # Restart if stuck
        restarts += 1
        # If we found a good solution, keep it.
        # If we restart, we keep the global best.
        
    return {'n': n, 'colors': best_colors} if best_colors else None
