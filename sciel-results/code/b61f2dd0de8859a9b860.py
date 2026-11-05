import time
import random

def construct():
    N = 256
    start_time = time.time()
    time_limit = 50.0  # Safety margin for 60s limit
    
    # Precompute conflicts
    # conflicts[x] is a list of (a, b) such that {x, a, b} is a 3-AP
    conflicts = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        # x is smallest: x, x+d, x+2d
        for d in range(1, N):
            if x + 2*d <= N:
                conflicts[x].append((x+d, x+2*d))
        # x is middle: x-d, x, x+d
        for d in range(1, N):
            if x-d >= 1 and x+d <= N:
                conflicts[x].append((x-d, x+d))
        # x is largest: x-2d, x-d, x
        for d in range(1, N):
            if x-2*d >= 1:
                conflicts[x].append((x-2*d, x-d))
        
    def is_addable(x, in_set):
        for a, b in conflicts[x]:
            if in_set[a] and in_set[b]:
                return False
        return True

    def get_score(in_set):
        return sum(1.0 / i for i in range(1, N + 1) if in_set[i])

    best_set = None
    best_score = 0.0
    
    # Initialize with greedy
    def greedy_init(randomize=False):
        in_set = [False] * (N + 1)
        candidates = list(range(1, N + 1))
        if randomize:
            random.shuffle(candidates)
        for x in candidates:
            if is_addable(x, in_set):
                in_set[x] = True
        return in_set

    # Run multiple random restarts
    num_restarts = 50 # Increased restarts
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if i == 0:
            current_set = greedy_init(False)
        else:
            current_set = greedy_init(True)
            
        current_score = get_score(current_set)
        if current_score > best_score:
            best_score = current_score
            best_set = current_set[:]
            
        # Local Search
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            
            # 1. Try to add smallest available
            best_add_x = -1
            for x in range(1, N + 1):
                if not current_set[x] and is_addable(x, current_set):
                    best_add_x = x
                    break # Smallest x is best
            
            if best_add_x != -1:
                current_set[best_add_x] = True
                current_score += 1.0 / best_add_x
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
                continue # Restart loop to check for more adds or swaps
            
            # 2. Try to swap
            best_swap_gain = 0.0
            best_swap_x = -1
            best_swap_y = -1
            
            for x in range(1, N + 1):
                if current_set[x]:
                    continue
                
                # Check if x can be added by removing some y
                candidate_ys = set()
                has_active_conflict = False
                for a, b in conflicts[x]:
                    if current_set[a] and current_set[b]:
                        has_active_conflict = True
                        if not candidate_ys:
                            candidate_ys = {a, b}
                        else:
                            candidate_ys = candidate_ys.intersection({a, b})
                            
                if not has_active_conflict:
                    # Should have been added in step 1
                    continue

                if not candidate_ys:
                    # Conflicts don't share a common element, so no single removal works
                    continue

                for y in candidate_ys:
                    if not current_set[y]:
                        continue
                    gain = 1.0 / x - 1.0 / y
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_x = x
                        best_swap_y = y
            
            if best_swap_x != -1:
                current_set[best_swap_y] = False
                current_set[best_swap_x] = True
                current_score += best_swap_gain
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
            
    if best_set is None:
        # Fallback to simple greedy if something went wrong
        best_set = greedy_init(False)

    return [i for i in range(1, N + 1) if best_set[i]]
