import time
import random
from itertools import combinations

def construct():
    _t0 = time.time()
    _DEADLINE = _t0 + 140.0  # Safety margin

    n = 16
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    m = len(edges)  # 120

    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx

    # Precompute K4s
    all_k4s = list(combinations(range(n), 4))
    k4_count = len(all_k4s)
    
    k4_edges = []
    for quad in all_k4s:
        e_idx = []
        for a, b in combinations(quad, 2):
            e_idx.append(edge_idx[(a, b)])
        k4_edges.append(e_idx)
        
    # For each edge, list of K4 indices containing it
    edge_k4s = [[] for _ in range(m)]
    for k4i, e_idx in enumerate(k4_edges):
        for ei in e_idx:
            edge_k4s[ei].append(k4i)

    def get_violations(colors, K):
        """Calculate total violations and distinct colors used."""
        # k4_color_counts[k4_idx] is a dict {color: count}
        # We want to minimize sum(max(0, 5 - len(keys with count>0)))
        total_viol = 0
        used_colors = set(colors)
        
        # For speed in SA, we maintain state. 
        # Here is a simple O(k4_count * 6) check for initial or final validation.
        for k4i in range(k4_count):
            cnt = {}
            for ei in k4_edges[k4i]:
                c = colors[ei]
                cnt[c] = cnt.get(c, 0) + 1
            distinct = len(cnt)
            if distinct < 5:
                total_viol += (5 - distinct)
                
        return total_viol, len(used_colors)

    def solve_with_K(K, time_budget):
        """Run Simulated Annealing to find valid coloring with K colors."""
        # Initialize
        colors = [random.randint(0, K - 1) for _ in range(m)]
        
        # State for fast delta calculation
        # k4_counts[k4_idx] is a list of length K, initialized to 0
        k4_counts = [[0] * K for _ in range(k4_count)]
        k4_distinct = [0] * k4_count
        
        for k4i in range(k4_count):
            for ei in k4_edges[k4i]:
                c = colors[ei]
                if k4_counts[k4i][c] == 0:
                    k4_distinct[k4i] += 1
                k4_counts[k4i][c] += 1
                
        total_viol = 0
        for k4i in range(k4_count):
            if k4_distinct[k4i] < 5:
                total_viol += (5 - k4_distinct[k4i])
                
        best_colors = colors[:]
        best_viol = total_viol
        best_used = len(set(colors))
        
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        
        t_start = time.time()
        while time.time() - t_start < time_budget and T > min_T:
            # Pick random edge
            ei = random.randint(0, m - 1)
            c_old = colors[ei]
            
            # Pick new color
            if K > 1:
                c_new = random.randint(0, K - 1)
                if c_new == c_old:
                    continue
            else:
                continue
            
            # Calculate delta violation
            delta_viol = 0
            affected_k4s = edge_k4s[ei]
            
            # Store old states to revert if rejected
            # We only need to revert if rejected. 
            # For each affected k4, we update counts.
            
            # Optimization: Only iterate affected k4s
            for k4i in affected_k4s:
                # Remove c_old
                old_distinct = k4_distinct[k4i]
                k4_counts[k4i][c_old] -= 1
                if k4_counts[k4i][c_old] == 0:
                    k4_distinct[k4i] -= 1
                    new_distinct = k4_distinct[k4i]
                else:
                    new_distinct = old_distinct
                
                old_contrib = 0 if old_distinct >= 5 else (5 - old_distinct)
                new_contrib = 0 if new_distinct >= 5 else (5 - new_distinct)
                
                # Add c_new
                if k4_counts[k4i][c_new] == 0:
                    k4_distinct[k4i] += 1
                    final_distinct = k4_distinct[k4i]
                else:
                    final_distinct = k4_distinct[k4i]
                    
                k4_counts[k4i][c_new] += 1
                
                final_contrib = 0 if final_distinct >= 5 else (5 - final_distinct)
                
                delta_viol += final_contrib - old_contrib
                
                # If rejected, we must revert. 
                # It's easier to just apply the change, check accept/reject, and revert if rejected.
                # But we need the final state for the next step if accepted.
                # So: Apply change to state, calc delta. If reject, revert state.
                
            # The above loop modifies state. Let's restructure.
            # 1. Apply change to state
            # 2. Calc delta based on before/after
            # This is tricky to do in one pass without storing 'before' states.
            
            # Let's do it this way:
            # Store old values for affected k4s
            old_states = []
            for k4i in affected_k4s:
                old_states.append((k4_counts[k4i][c_old], k4_counts[k4i][c_new], k4_distinct[k4i]))
                
            # Apply change
            for k4i in affected_k4s:
                k4_counts[k4i][c_old] -= 1
                if k4_counts[k4i][c_old] == 0:
                    k4_distinct[k4i] -= 1
                k4_counts[k4i][c_new] += 1
                if k4_counts[k4i][c_new] == 1: # was 0
                    k4_distinct[k4i] += 1
                    
            # Calc delta
            # We need old distinct for each k4.
            # We stored old counts. We can recompute old distinct? No, we stored distinct too.
            # Let's just compute delta_viol by comparing old_contrib and new_contrib.
            # We need to reconstruct old_contrib from old_states.
            
            delta_viol = 0
            for idx, k4i in enumerate(affected_k4s):
                old_c_old, old_c_new, old_dist = old_states[idx]
                old_contrib = 0 if old_dist >= 5 else (5 - old_dist)
                
                new_dist = k4_distinct[k4i]
                new_contrib = 0 if new_dist >= 5 else (5 - new_dist)
                
                delta_viol += new_contrib - old_contrib
                
            # Accept/Reject
            if delta_viol <= 0 or random.random() < pow(2.718281828, -delta_viol / T):
                total_viol += delta_viol
                colors[ei] = c_new
                
                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = colors[:]
                    best_used = len(set(colors))
                    if best_viol == 0:
                        # Found valid solution
                        # Check if we can reduce used colors? 
                        # The SA is constrained to K colors. 
                        # We return the best one found.
                        pass
            else:
                # Revert
                for idx, k4i in enumerate(affected_k4s):
                    old_c_old, old_c_new, old_dist = old_states[idx]
                    # Revert counts
                    k4_counts[k4i][c_new] -= 1
                    if k4_counts[k4i][c_new] == 0:
                        k4_distinct[k4i] -= 1
                    k4_counts[k4i][c_old] += 1
                    if k4_counts[k4i][c_old] == 1: # was 0
                        k4_distinct[k4i] += 1
                    # distinct is restored by logic above? 
                    # Let's just re-set distinct from counts to be safe? No, that's slow.
                    # The revert logic:
                    # c_new count decreased. If it became 0, distinct decreases.
                    # c_old count increased. If it was 0, distinct increases.
                    # This matches the forward logic reversed.
                    
                # Note: total_viol is not changed on reject.
            
            T *= cooling
            
        return best_colors, best_viol, best_used

    # Try K=13, 14, 15
    # Allocate time based on remaining time
    remaining = _DEADLINE - time.time()
    
    # Strategy: Try 15 first (easier), then 14, then 13?
    # Or 13 (hardest) first? 
    # If we find 13, it's the best. 
    # But 13 might be very hard. 
    # Let's try 15 with a large budget, 14 with medium, 13 with small.
    # Actually, if 15 is easy, we get 1.13. If we spend too much on 13 and fail, we might not have time for 15.
    # Better: Try 13 for a short time. If fail, try 14. If fail, try 15.
    # If 13 is impossible in short time, we lose that time.
    # Given bounds 13-15, 15 is likely the most robust to find.
    # Let's try 15 first to secure a score improvement.
    
    best_result = None
    best_score_val = 17 # seed
    
    # Try 15
    t_start = time.time()
    time_for_15 = min(remaining * 0.5, 60.0)
    if time_for_15 > 5:
        cols, viol, used = solve_with_K(15, time_for_15)
        if viol == 0:
            # Compress colors
            mapping = {}
            new_cols = []
            next_c = 0
            for c in cols:
                if c not in mapping:
                    mapping[c] = next_c
                    next_c += 1
                new_cols.append(mapping[c])
            best_result = new_cols
            best_score_val = next_c # used colors
            # If we found 15, we can try to improve to 14 or 13 with remaining time
            remaining = _DEADLINE - time.time()
            
    # Try 14 if time remains and we haven't found 13
    if best_score_val > 14 and time.time() < _DEADLINE - 10:
        time_for_14 = min(_DEADLINE - time.time() - 5, 40.0)
        if time_for_14 > 5:
            cols, viol, used = solve_with_K(14, time_for_14)
            if viol == 0:
                mapping = {}
                new_cols = []
                next_c = 0
                for c in cols:
                    if c not in mapping:
                        mapping[c] = next_c
                        next_c += 1
                    new_cols.append(mapping[c])
                if next_c < best_score_val:
                    best_result = new_cols
                    best_score_val = next_c
                remaining = _DEADLINE - time.time()
                
    # Try 13 if time remains
    if best_score_val > 13 and time.time() < _DEADLINE - 5:
        time_for_13 = _DEADLINE - time.time() - 2
        if time_for_13 > 5:
            cols, viol, used = solve_with_K(13, time_for_13)
            if viol == 0:
                mapping = {}
                new_cols = []
                next_c = 0
                for c in cols:
                    if c not in mapping:
                        mapping[c] = next_c
                        next_c += 1
                    new_cols.append(mapping[c])
                if next_c < best_score_val:
                    best_result = new_cols
                    best_score_val = next_c
                    
    if best_result is not None:
        return best_result
    else:
        # Fallback to seed if SA failed (should not happen if SA is correct, but just in case)
        return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
