import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 128
    
    # Precompute pairs by sum
    pairs_by_sum = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            s = a + b
            if s > N:
                break
            pairs_by_sum[s].append((a, b))
    
    # For each (a,b), precompute the list of (a+c, b+c) pairs for valid c>b
    # Store as two lists for faster checking
    ac_list = {}
    bc_list = {}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            s = a + b
            if s > N:
                break
            max_c = min(N - a, N - b)
            if max_c > b:
                acs = [a + c for c in range(b + 1, max_c + 1)]
                bcs = [b + c for c in range(b + 1, max_c + 1)]
                ac_list[(a, b)] = acs
                bc_list[(a, b)] = bcs
            else:
                ac_list[(a, b)] = []
                bc_list[(a, b)] = []
    
    def is_valid_mask(mask):
        for s in range(1, N + 1):
            if not (mask >> s) & 1:
                continue
            for (a, b) in pairs_by_sum[s]:
                acs = ac_list[(a, b)]
                if not acs:
                    continue
                bcs = bc_list[(a, b)]
                for i in range(len(acs)):
                    if (mask >> acs[i]) & 1 and (mask >> bcs[i]) & 1:
                        return False
        return True
    
    def is_valid_set(s_set):
        mask = 0
        for x in s_set:
            mask |= (1 << x)
        return is_valid_mask(mask)
    
    best_set = set(range(N // 2 + 1, N + 1))
    best_size = len(best_set)
    
    random.seed(42)
    
    def local_search(current_set, max_steps=50000):
        current_mask = 0
        for x in current_set:
            current_mask |= (1 << x)
        
        steps = 0
        stuck_count = 0
        while steps < max_steps and time.time() < deadline - 5:
            steps += 1
            improved = False
            
            candidates = [x for x in range(1, N + 1) if not (current_mask >> x) & 1]
            if not candidates:
                break
            
            candidates.sort(reverse=True)
            shuffle_end = max(1, len(candidates) // 3)
            random.shuffle(candidates[:shuffle_end])
            
            for elem in candidates:
                trial_mask = current_mask | (1 << elem)
                if is_valid_mask(trial_mask):
                    current_mask = trial_mask
                    improved = True
                    stuck_count = 0
                    break
            
            if improved:
                continue
            
            stuck_count += 1
            if stuck_count > 30:
                current_elements = [x for x in range(1, N + 1) if (current_mask >> x) & 1]
                if not current_elements:
                    break
                
                swap_found = False
                for to_remove in random.sample(current_elements, min(len(current_elements), 10)):
                    if swap_found:
                        break
                    base_mask = current_mask & ~(1 << to_remove)
                    cands = [x for x in range(1, N + 1) if x != to_remove and not (base_mask >> x) & 1]
                    cands.sort(reverse=True)
                    random.shuffle(cands[:max(1, len(cands) // 4)])
                    for elem in cands:
                        trial_mask = base_mask | (1 << elem)
                        if is_valid_mask(trial_mask):
                            current_mask = trial_mask
                            swap_found = True
                            stuck_count = 0
                            break
                
                if not swap_found:
                    to_remove = random.choice(current_elements)
                    base_mask = current_mask & ~(1 << to_remove)
                    cands = [x for x in range(1, N + 1) if x != to_remove and not (base_mask >> x) & 1]
                    cands.sort(reverse=True)
                    new_mask = base_mask
                    added = 0
                    for elem in cands:
                        if added == 2:
                            break
                        trial_mask = new_mask | (1 << elem)
                        if is_valid_mask(trial_mask):
                            new_mask = trial_mask
                            added += 1
                    if added >= 1:
                        current_mask = new_mask
                        stuck_count = 0
        
        result_set = set()
        for x in range(1, N + 1):
            if (current_mask >> x) & 1:
                result_set.add(x)
        return result_set

    num_restarts = 0
    while time.time() < deadline - 15:
        num_restarts += 1
        
        if num_restarts == 1:
            init = set(range(N // 2 + 1, N + 1))
        elif num_restarts == 2:
            init = set(range(N // 2 + 1, N + 1))
            for x in range(N // 2, 0, -1):
                trial = init | {x}
                if is_valid_set(trial):
                    init = trial
        elif num_restarts % 2 == 0:
            init = set()
            for x in range(N, 0, -1):
                if random.random() < 0.85:
                    trial = init | {x}
                    if is_valid_set(trial):
                        init = trial
                else:
                    y = random.randint(1, max(1, x - 20))
                    trial = init | {y}
                    if is_valid_set(trial):
                        init = trial
        else:
            if best_set:
                init = set(best_set)
                to_remove = random.sample(list(init), min(len(init), random.randint(1, 3)))
                for x in to_remove:
                    init.discard(x)
            else:
                init = set(range(N // 2 + 1, N + 1))
        
        improved_set = local_search(init)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
    
    for polish_pass in range(10):
        if time.time() > deadline - 2:
            break
        remaining = set(range(1, N + 1)) - best_set
        if not remaining:
            break
        added_any = False
        for elem in sorted(remaining, reverse=True):
            if time.time() > deadline - 2:
                break
            trial = best_set | {elem}
            if is_valid_set(trial):
                best_set = trial
                added_any = True
        if not added_any:
            break
    
    result = list(best_set)
    result = [int(x) for x in result]
    return result
