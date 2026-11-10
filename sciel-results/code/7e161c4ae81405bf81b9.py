import random
import time

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin (90s limit, leave 35s buffer)
    
    # Precompute all 4-APs
    aps = []
    # index: element -> list of AP indices
    elem_to_aps = [[] for _ in range(N + 1)]  # 1-indexed
    
    ap_id = 0
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps.append((a, b, c, e))
            for x in (a, b, c, e):
                elem_to_aps[x].append(ap_id)
            ap_id += 1
    
    total_aps = len(aps)
    
    best_size = 0
    best_set = set()
    
    def greedy_remove_and_fill():
        """Greedy removal to break all APs, then fill back."""
        S = set(range(1, N + 1))
        ap_count = [4] * total_aps
        elem_active_count = [0] * (N + 1)
        
        for ap_idx in range(total_aps):
            a, b, c, e = aps[ap_idx]
            elem_active_count[a] += 1
            elem_active_count[b] += 1
            elem_active_count[c] += 1
            elem_active_count[e] += 1
        
        # Greedy removal
        while True:
            max_count = 0
            candidate = -1
            # Use a random subset to speed up and add randomness
            sample_size = min(N, max(50, N // 4))
            items = random.sample(range(1, N + 1), sample_size)
            for x in items:
                if x in S and elem_active_count[x] > max_count:
                    max_count = elem_active_count[x]
                    candidate = x
            
            if max_count == 0:
                break
            
            S.remove(candidate)
            
            for ap_idx in elem_to_aps[candidate]:
                if ap_count[ap_idx] > 0:
                    ap_count[ap_idx] -= 1
                    if ap_count[ap_idx] == 3:
                        a, b, c, e = aps[ap_idx]
                        for y in (a, b, c, e):
                            if y != candidate:
                                elem_active_count[y] -= 1
        
        # Add-back phase
        candidates = [x for x in range(1, N + 1) if x not in S]
        random.shuffle(candidates)
        
        for x in candidates:
            can_add = True
            for ap_idx in elem_to_aps[x]:
                a, b, c, e = aps[ap_idx]
                others = [y for y in (a, b, c, e) if y != x]
                if all(y in S for y in others):
                    can_add = False
                    break
            if can_add:
                S.add(x)
        
        return S
    
    def local_search_2exchange(S, max_iters=200):
        """Try 2-exchange moves: remove one, add another."""
        S = set(S)
        improved = True
        iterations = 0
        while improved and iterations < max_iters and time.time() < deadline:
            improved = False
            iterations += 1
            # Try removing each element in S
            for x in list(S):
                S.remove(x)
                # Try adding each element not in S
                for y in range(1, N + 1):
                    if y in S:
                        continue
                    # Check if adding y is valid
                    can_add = True
                    for ap_idx in elem_to_aps[y]:
                        a, b, c, e = aps[ap_idx]
                        others = [z for z in (a, b, c, e) if z != y]
                        if all(z in S for z in others):
                            can_add = False
                            break
                    if can_add:
                        S.add(y)
                        improved = True
                        break
                if not improved:
                    S.add(x)  # Restore x if no improvement found
            if time.time() > deadline - 5:
                break
        return S
    
    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 10:
        num_restarts += 1
        S = greedy_remove_and_fill()
        
        # Local search to improve
        if time.time() < deadline - 5:
            S = local_search_2exchange(S, max_iters=100)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
        
        # If we've had a good result, spend remaining time on final polish
        if best_size >= 250 and time.time() < deadline - 3:
            break
    
    # Final polish: try to add any remaining elements
    if best_set and time.time() < deadline - 2:
        candidates = [x for x in range(1, N + 1) if x not in best_set]
        random.shuffle(candidates)
        for x in candidates:
            can_add = True
            for ap_idx in elem_to_aps[x]:
                a, b, c, e = aps[ap_idx]
                others = [y for y in (a, b, c, e) if y != x]
                if all(y in best_set for y in others):
                    can_add = False
                    break
            if can_add:
                best_set.add(x)
    
    return sorted(best_set)
