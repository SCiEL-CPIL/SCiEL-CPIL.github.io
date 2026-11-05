import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    N = 1000
    best_set = None
    best_size = 0

    def can_add(x, chosen_set, counts):
        if counts.get(2 * x, 0) >= 2:
            return False
        for y in chosen_set:
            if y == x:
                continue
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        return True

    def add_element(x, chosen_set, counts):
        chosen_set.add(x)
        s = 2 * x
        counts[s] = counts.get(s, 0) + 1
        for y in chosen_set:
            if y == x:
                continue
            s = x + y
            counts[s] = counts.get(s, 0) + 1

    def remove_element(x, chosen_set, counts):
        chosen_set.remove(x)
        s = 2 * x
        counts[s] -= 1
        if counts[s] == 0:
            del counts[s]
        for y in chosen_set:
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]

    def greedy_with_scoring():
        """Greedy using scoring: pick element that minimizes future conflict potential."""
        chosen_set = set()
        counts = {}
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, chosen_set, counts):
                add_element(x, chosen_set, counts)

        return chosen_set, counts

    def greedy_sequential():
        """Greedy picking smallest available first (deterministic baseline)."""
        chosen_set = set()
        counts = {}
        
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if can_add(x, chosen_set, counts):
                add_element(x, chosen_set, counts)

        return chosen_set, counts

    def local_search(chosen_set, counts):
        """Enhanced local search with targeted swaps and candidate tracking."""
        chosen_set = set(chosen_set)
        counts = dict(counts)
        unchosen = set(range(1, N + 1)) - chosen_set
        unchosen_list = list(unchosen)
        
        improved = True
        passes = 0
        max_passes = 20

        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            unchosen_list = list(unchosen)
            random.shuffle(unchosen_list)

            # Phase 1: Try to add any candidate directly
            for x in unchosen_list:
                if time.time() > deadline:
                    break
                if can_add(x, chosen_set, counts):
                    add_element(x, chosen_set, counts)
                    unchosen.remove(x)
                    improved = True
                    continue

            # Phase 2: Try targeted swaps
            if not chosen_set or not unchosen:
                break
                
            unchosen_list = list(unchosen)
            random.shuffle(unchosen_list)
            chosen_list = list(chosen_set)
            random.shuffle(chosen_list)
            
            max_swap_attempts = min(len(unchosen_list) * len(chosen_list) // 50, 200)
            attempts = 0
            
            for x in unchosen_list:
                if time.time() > deadline:
                    break
                if attempts >= max_swap_attempts:
                    break
                    
                for y in chosen_list:
                    if time.time() > deadline:
                        break
                    if attempts >= max_swap_attempts:
                        break
                    attempts += 1
                    
                    # Try swapping: remove y, add x
                    remove_element(y, chosen_set, counts)
                    
                    if can_add(x, chosen_set, counts):
                        add_element(x, chosen_set, counts)
                        unchosen.remove(x)
                        unchosen.add(y)
                        improved = True
                        break
                    else:
                        # Put y back
                        add_element(y, chosen_set, counts)
                
                if improved:
                    # Restart the unchosen_list since it changed
                    unchosen_list = list(unchosen)
                    random.shuffle(unchosen_list)
                    break

        return chosen_set, counts

    def deep_local_search(chosen_set, counts):
        """More aggressive local search with multiple swap attempts per candidate."""
        chosen_set = set(chosen_set)
        counts = dict(counts)
        unchosen = set(range(1, N + 1)) - chosen_set
        
        improved = True
        passes = 0
        max_passes = 30

        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            
            unchosen_list = list(unchosen)
            random.shuffle(unchosen_list)
            chosen_list = list(chosen_set)
            random.shuffle(chosen_list)
            
            # Try to add any candidate
            for x in unchosen_list:
                if time.time() > deadline:
                    break
                if can_add(x, chosen_set, counts):
                    add_element(x, chosen_set, counts)
                    unchosen.remove(x)
                    improved = True
                    break

            if improved:
                continue
            
            # Try swaps: for each unchosen, try multiple chosen elements
            for x in unchosen_list:
                if time.time() > deadline:
                    break
                
                # Try swapping with a few random chosen elements
                swap_attempts = min(5, len(chosen_list))
                tried = set()
                
                for _ in range(swap_attempts):
                    if time.time() > deadline:
                        break
                    
                    y = random.choice(chosen_list)
                    if y in tried:
                        continue
                    tried.add(y)
                    
                    remove_element(y, chosen_set, counts)
                    
                    if can_add(x, chosen_set, counts):
                        add_element(x, chosen_set, counts)
                        unchosen.remove(x)
                        unchosen.add(y)
                        improved = True
                        break
                    else:
                        add_element(y, chosen_set, counts)
                
                if improved:
                    break

        return chosen_set, counts

    strategies = [
        greedy_with_scoring,
        greedy_sequential,
        lambda: (lambda cs, ct: (cs, ct))(greedy_with_scoring()),
    ]

    for restart in range(200):
        if time.time() > deadline:
            break

        # Vary the strategy
        if restart % 4 == 0:
            chosen_set, counts = greedy_sequential()
        elif restart % 4 == 1:
            chosen_set, counts = greedy_with_scoring()
        elif restart % 4 == 2:
            # Randomized greedy with different seed
            random.seed(restart * 13 + 7)
            chosen_set, counts = greedy_with_scoring()
        else:
            # Start from a small random subset and grow
            chosen_set = set()
            counts = {}
            initial_size = random.randint(10, 30)
            initial_candidates = random.sample(range(1, N + 1), initial_size)
            for x in initial_candidates:
                if can_add(x, chosen_set, counts):
                    add_element(x, chosen_set, counts)
            # Then do greedy to fill
            remaining = [x for x in range(1, N + 1) if x not in chosen_set]
            random.shuffle(remaining)
            for x in remaining:
                if time.time() > deadline:
                    break
                if can_add(x, chosen_set, counts):
                    add_element(x, chosen_set, counts)

        # Apply local search
        if time.time() < deadline - 5:
            chosen_set, counts = local_search(chosen_set, counts)
        
        # Apply deeper local search if time permits
        if time.time() < deadline - 10:
            chosen_set, counts = deep_local_search(chosen_set, counts)

        current_size = len(chosen_set)
        if current_size > best_size:
            best_size = current_size
            best_set = list(chosen_set)

    if best_set is None:
        best_set = []

    return best_set

random.seed(42)
