import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    N = 128
    
    # Pre-compute all 4-term APs and index them by position for fast updates
    aps = []
    ap_indices = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        for i in range(N - 3 * d):
            ap = (i, i + d, i + 2 * d, i + 3 * d)
            ap_idx = len(aps)
            aps.append(ap)
            for idx in ap:
                ap_indices[idx].append(ap_idx)
                
    num_aps = len(aps)
    
    def count_violations(colors):
        count = 0
        for ap in aps:
            if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                count += 1
        return count

    def run_seed():
        colors=[]
        for x in range(1,129):
            for color in range(128):
                if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                    and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                    colors.append(color);break
        return colors

    # Get initial seed solution
    seed_colors = run_seed()
    best_colors = seed_colors[:]
    best_k = len(set(seed_colors))
    
    # Local search function with incremental violation tracking
    def local_search(target_k, time_limit):
        colors = [random.randint(0, target_k - 1) for _ in range(N)]
        
        # Initialize violation flags
        violation_flags = [False] * num_aps
        num_violations = 0
        for ai, ap in enumerate(aps):
            if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                violation_flags[ai] = True
                num_violations += 1
        
        best_local_colors = colors[:]
        best_local_viol = num_violations
        steps_since_improve = 0
        max_steps_no_improve = 500
        
        while time.time() < time_limit:
            if num_violations == 0:
                return colors
            
            # Pick a random violated AP
            violated_indices = [i for i, v in enumerate(violation_flags) if v]
            if not violated_indices:
                return colors
            ai = random.choice(violated_indices)
            ap = aps[ai]
            
            # Pick a random element in the AP to change
            pos = random.randint(0, 3)
            idx = ap[pos]
            old_c = colors[idx]
            
            # Try all possible new colors
            best_new_c = old_c
            best_new_viol = num_violations
            best_new_flags = None
            
            for c in range(target_k):
                if c == old_c:
                    continue
                
                colors[idx] = c
                
                # Update violations for affected APs
                old_flags = []
                delta = 0
                for affected_ai in ap_indices[idx]:
                    old_flag = violation_flags[affected_ai]
                    old_flags.append(old_flag)
                    a = aps[affected_ai]
                    is_violated = len({colors[a[0]], colors[a[1]], colors[a[2]], colors[a[3]]}) < 3
                    if old_flag and not is_violated:
                        delta -= 1
                    elif not old_flag and is_violated:
                        delta += 1
                    violation_flags[affected_ai] = is_violated
                
                new_viol = num_violations + delta
                
                if new_viol < best_new_viol:
                    best_new_viol = new_viol
                    best_new_c = c
                    best_new_flags = violation_flags[:]
                
                # Revert flags for next iteration
                for i, affected_ai in enumerate(ap_indices[idx]):
                    violation_flags[affected_ai] = old_flags[i]
            
            # Apply best move
            if best_new_c != old_c:
                colors[idx] = best_new_c
                for affected_ai in ap_indices[idx]:
                    a = aps[affected_ai]
                    violation_flags[affected_ai] = len({colors[a[0]], colors[a[1]], colors[a[2]], colors[a[3]]}) < 3
                num_violations = sum(1 for v in violation_flags if v)
                
                if num_violations < best_local_viol:
                    best_local_viol = num_violations
                    best_local_colors = colors[:]
                    steps_since_improve = 0
                else:
                    steps_since_improve += 1
            else:
                steps_since_improve += 1
            
            if steps_since_improve > max_steps_no_improve:
                # Restart
                colors = [random.randint(0, target_k - 1) for _ in range(N)]
                for ai, ap in enumerate(aps):
                    violation_flags[ai] = len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3
                num_violations = sum(1 for v in violation_flags if v)
                if num_violations < best_local_viol:
                    best_local_viol = num_violations
                    best_local_colors = colors[:]
                steps_since_improve = 0
        
        return best_local_colors

    # Try to improve with fewer colors
    for target_k in range(3, best_k):
        if time.time() > deadline - 2:
            break
        
        time_limit = min(deadline - 1, time.time() + 5.0)
        result = local_search(target_k, time_limit)
        
        if result is not None and count_violations(result) == 0:
            if len(set(result)) < best_k:
                best_k = len(set(result))
                best_colors = result[:]
            else:
                best_colors = result[:]
            break
    
    # If no improvement found, return the seed or best found
    return best_colors
# EVOLVE-BLOCK-END
