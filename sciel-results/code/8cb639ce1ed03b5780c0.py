import random
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 80  # 80s safety margin before 90s kill
    
    n = 512
    
    # Precompute all 4-APs and element-to-AP mapping
    ap_list = []
    elem_to_aps = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            idx = len(ap_list)
            ap_list.append((a, b, c, e))
            elem_to_aps[a].append(idx)
            elem_to_aps[b].append(idx)
            elem_to_aps[c].append(idx)
            elem_to_aps[e].append(idx)
    
    num_aps = len(ap_list)
    
    def can_add(s_bitmask, x):
        """Check if x can be added to the set represented by s_bitmask without creating a 4-AP"""
        for ap_idx in elem_to_aps[x]:
            a, b, c, d = ap_list[ap_idx]
            others = [y for y in (a, b, c, d) if y != x]
            # Check if all others are in the set
            all_in = True
            for y in others:
                if not (s_bitmask >> y) & 1:
                    all_in = False
                    break
            if all_in:
                return False
        return True
    
    def run_trial(seed):
        random.seed(seed)
        
        # Phase 1: Greedy hitting set removal
        in_set = [True] * (n + 1)
        active_aps = set(range(num_aps))
        
        # Calculate initial degrees
        deg = [0] * (n + 1)
        for i in range(num_aps):
            a, b, c, d = ap_list[i]
            deg[a] += 1
            deg[b] += 1
            deg[c] += 1
            deg[d] += 1
        
        while active_aps:
            # Find element with max degree among those in_set
            max_deg = -1
            candidates = []
            for i in range(1, n + 1):
                if in_set[i]:
                    if deg[i] > max_deg:
                        max_deg = deg[i]
                        candidates = [i]
                    elif deg[i] == max_deg:
                        candidates.append(i)
                        
            if not candidates or max_deg == 0:
                break
                
            x = random.choice(candidates)
            in_set[x] = False
            
            # Update degrees for all APs containing x
            for ap_idx in elem_to_aps[x]:
                if ap_idx in active_aps:
                    active_aps.remove(ap_idx)
                    a, b, c, d = ap_list[ap_idx]
                    for y in (a, b, c, d):
                        if y != x and in_set[y]:
                            deg[y] -= 1
        
        S = set(i for i in range(1, n + 1) if in_set[i])
        
        # Phase 2: Add-back phase
        removed = [i for i in range(1, n + 1) if not in_set[i]]
        random.shuffle(removed)
        
        for x in removed:
            if can_add(S, x):
                S.add(x)
        
        # Phase 3: Local optimization via swaps
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            excluded = set(range(1, n + 1)) - S
            
            # Try removing each element and adding back multiple others
            for x in list(S):
                if time.time() > DEADLINE:
                    return S
                trial_set = set(S) - {x}
                added_count = 0
                for y in sorted(excluded):
                    if can_add(trial_set, y):
                        trial_set.add(y)
                        added_count += 1
                if added_count > 1:  # Net gain
                    S = trial_set
                    improved = True
                    break
        
        return S
    
    # Run multiple trials with different seeds
    best_set = set()
    
    # First, do a quick greedy forward construction as baseline
    selected = set()
    for x in range(1, n + 1):
        if can_add(selected, x):
            selected.add(x)
    best_set = set(selected)
    
    # Run hitting set trials
    num_trials = 50
    for trial in range(num_trials):
        if time.time() > DEADLINE:
            break
        s = run_trial(trial * 7 + 13)
        if len(s) > len(best_set):
            best_set = s
    
    # Final: try to add any remaining elements
    for x in range(1, n + 1):
        if x not in best_set and can_add(best_set, x):
            best_set.add(x)
    
    return sorted(best_set)
