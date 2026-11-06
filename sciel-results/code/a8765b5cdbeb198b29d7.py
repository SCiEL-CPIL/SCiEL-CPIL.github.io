import numpy as np
import time

def construct():
    N = 128
    # Precompute all 4-term APs
    # a, a+d, a+2d, a+3d <= N
    # 0-indexed: a, a+d, a+2d, a+3d < N
    # a >= 0, a+3d < N => a < N - 3d
    # d >= 1
    
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    # Map position to AP indices
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # Function to check if an AP is bad (uses < 3 distinct colors)
    def is_bad_ap(c0, c1, c2, c3):
        # Returns 1 if bad, 0 if good
        # Bad if distinct colors < 3
        # We can just use a set for simplicity in the inner loop if it's not too slow,
        # but we want speed.
        # Let's use the sorted logic or simple comparisons.
        # Since we do this many times, let's be efficient.
        # If any pair is different, we have at least 2.
        # We need to check if we have 3.
        # If c0==c1==c2==c3: 1 color (Bad)
        # If c0==c1==c2 and c2!=c3: 2 colors (Bad)
        # ...
        
        # Simple check:
        # Count distinct
        # This is 4 comparisons roughly.
        
        # Let's use a helper that is fast.
        # If c0 == c1:
        #   If c2 == c3:
        #     If c0 == c2: 1 color (Bad)
        #     Else: 2 colors (Bad)
        #   Else:
        #     If c0 == c2: 2 colors (Bad) # c0=c1=c2 != c3
        #     If c0 == c3: 2 colors (Bad) # c0=c1=c3 != c2
        #     Else: 3 colors (Good) # c0=c1, c2!=c3, c0!=c2, c0!=c3 => {c0, c2, c3}
        # Else:
        #   If c2 == c3:
        #     If c0 == c2: 2 colors (Bad) # c0=c2=c3 != c1
        #     If c1 == c2: 2 colors (Bad) # c1=c2=c3 != c0
        #     Else: 3 colors (Good) # {c0, c1, c2}
        #   Else:
        #     # c0 != c1, c2 != c3
        #     If c0 == c2:
        #       If c0 == c3: 2 colors (Bad) # c0=c2=c3 != c1
        #       Else: 3 colors (Good) # {c0, c1, c3}
        #     If c0 == c3:
        #       If c1 == c2: 2 colors (Bad) # c0=c3, c1=c2, c0!=c1
        #       Else: 3 colors (Good) # {c0, c1, c2}
        #     If c1 == c2:
        #       If c1 == c3: 2 colors (Bad) # c1=c2=c3 != c0
        #       Else: 3 colors (Good) # {c0, c1, c3}
        #     If c1 == c3:
        #       If c0 == c2: 2 colors (Bad) # c1=c3, c0=c2
        #       Else: 3 colors (Good) # {c0, c1, c2}
        #     Else: 4 colors (Good)
        
        # This is complex. Let's just use `len({c0, c1, c2, c3}) < 3`.
        # Python set creation is fast enough for ~100-200 checks per move.
        return 1 if len({c0, c1, c2, c3}) < 3 else 0

    def calculate_total_cost(colors):
        cost = 0
        for ap in aps:
            cost += is_bad_ap(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]])
        return cost

    # Search
    best_colors = None
    best_cost = float('inf')
    start_time = time.time()
    deadline = start_time + 45.0 # Leave 15s margin
    
    # Try k from 4 upwards
    for k in range(4, 15):
        if time.time() > deadline:
            break
            
        found = False
        # Run multiple restarts for this k
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize random
            colors = np.random.randint(0, k, N)
            
            # Calculate initial cost and bad_ap flags
            bad_flags = np.zeros(num_aps, dtype=bool)
            for idx, ap in enumerate(aps):
                bad_flags[idx] = (is_bad_ap(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]) == 1)
            current_cost = np.sum(bad_flags)
            
            if current_cost == 0:
                return colors.tolist()
                
            # Local Search (Hill Climbing with Random Restarts)
            # We use a simple stochastic local search
            # Temperature or probability of accepting worse moves?
            # Let's try pure hill climbing first, if stuck, restart.
            # Or Simulated Annealing.
            
            # Given the landscape, SA is better.
            T = 10.0
            T_min = 0.1
            cooling = 0.995
            
            for step in range(2000):
                if time.time() > deadline:
                    break
                    
                # Pick random position
                pos = np.random.randint(N)
                old_color = colors[pos]
                new_color = np.random.randint(k)
                if new_color == old_color:
                    continue
                    
                # Calculate delta cost
                delta = 0
                affected_aps = pos_to_aps[pos]
                for idx in affected_aps:
                    ap = aps[idx]
                    # Get current colors
                    c0 = colors[ap[0]]
                    c1 = colors[ap[1]]
                    c2 = colors[ap[2]]
                    c3 = colors[ap[3]]
                    
                    old_bad = 1 if is_bad_ap(c0, c1, c2, c3) == 1 else 0
                    
                    # Apply change temporarily
                    if ap[0] == pos:
                        c0 = new_color
                    elif ap[1] == pos:
                        c1 = new_color
                    elif ap[2] == pos:
                        c2 = new_color
                    elif ap[3] == pos:
                        c3 = new_color
                    
                    new_bad = 1 if is_bad_ap(c0, c1, c2, c3) == 1 else 0
                    
                    if old_bad == 0 and new_bad == 1:
                        delta += 1
                    elif old_bad == 1 and new_bad == 0:
                        delta -= 1
                    
                # Acceptance
                if delta <= 0:
                    colors[pos] = new_color
                    current_cost += delta
                    # Update bad_flags for affected APs
                    for idx in affected_aps:
                        ap = aps[idx]
                        bad_flags[idx] = (is_bad_ap(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]) == 1)
                else:
                    # Accept with probability exp(-delta / T)
                    if np.random.rand() < np.exp(-delta / T):
                        colors[pos] = new_color
                        current_cost += delta
                        for idx in affected_aps:
                            ap = aps[idx]
                            bad_flags[idx] = (is_bad_ap(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]) == 1)
                
                if current_cost == 0:
                    return colors.tolist()
                    
                T *= cooling
                if T < T_min:
                    break
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                
    # If we found a valid one, it would have returned.
    # Otherwise, return the best one found (which might be invalid, but we want to minimize colors? 
    # The task says "Every nonconstant... must use AT LEAST three". 
    # If we return an invalid one, the score is 0. 
    # We must return a VALID one. 
    # If we can't find a valid one with k < 15, we should try larger k until we find one.
    # But we have a time limit.
    
    # Fallback: Try to find ANY valid coloring with a larger k, e.g., 10 or 20.
    # If k=20, it's very likely to exist and easy to find.
    # Let's try k=10, 20, 30 in the remaining time.
    
    for k in [10, 20, 30, 50]:
        if time.time() > deadline:
            break
        for restart in range(10):
            if time.time() > deadline:
                break
            colors = np.random.randint(0, k, N)
            # Simple check
            cost = calculate_total_cost(colors)
            if cost == 0:
                return colors.tolist()
            # Quick local search
            T = 5.0
            for step in range(500):
                if time.time() > deadline:
                    break
                pos = np.random.randint(N)
                old_c = colors[pos]
                new_c = np.random.randint(k)
                if new_c == old_c: continue
                delta = 0
                for idx in pos_to_aps[pos]:
                    ap = aps[idx]
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    ob = 1 if is_bad_ap(c0, c1, c2, c3) else 0
                    if ap[0]==pos: c0=new_c
                    elif ap[1]==pos: c1=new_c
                    elif ap[2]==pos: c2=new_c
                    elif ap[3]==pos: c3=new_c
                    nb = 1 if is_bad_ap(c0, c1, c2, c3) else 0
                    if ob==0 and nb==1: delta+=1
                    elif ob==1 and nb==0: delta-=1
                if delta <= 0 or np.random.rand() < np.exp(-delta/T):
                    colors[pos] = new_c
                    cost += delta
                if cost == 0:
                    return colors.tolist()
                T *= 0.99

    # If still nothing, return a large-k random coloring (likely valid)
    # But we need it to be valid.
    # With k=128, it's almost certainly valid.
    colors = np.random.randint(0, 128, N)
    # Verify
    cost = calculate_total_cost(colors)
    if cost == 0:
        return colors.tolist()
    
    # If even that fails (unlikely), return the best effort
    if best_colors is not None:
        return best_colors.tolist()
    return colors.tolist()
