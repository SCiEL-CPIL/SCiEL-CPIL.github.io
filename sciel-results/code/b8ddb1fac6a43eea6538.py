import time
import random
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 85
    
    # Precompute gcd table for 1..128
    N = 128
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_bad_triple(a, b, c):
        return gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]

    def has_conflict(selected, x):
        # selected is a set or list
        # Check if adding x creates a bad triple
        for a in selected:
            for b in selected:
                if a >= b:
                    continue
                if is_bad_triple(a, b, x):
                    return True
        return False

    def is_valid_set(selected):
        # Check if the set itself is valid
        # This is O(|S|^3) but we can optimize
        # Actually, we build incrementally, so we assume valid.
        # For random restarts, we need to check.
        for a, b, c in combinations(selected, 3):
            if is_bad_triple(a, b, c):
                return False
        return True

    def local_search(selected, time_limit):
        selected = set(selected)
        best_size = len(selected)
        improved = True
        last_improvement = time.time()
        
        while time.time() < time_limit:
            improved = False
            # Try adding
            for x in range(1, N + 1):
                if x in selected:
                    continue
                if not has_conflict(selected, x):
                    selected.add(x)
                    improved = True
                    best_size = len(selected)
                    break # Restart loop to check new state
            
            if not improved:
                # Try swapping
                # Pick a random element to remove? Or try all?
                # Trying all is 65 * 63 ~ 4000 checks.
                # Each check has_conflict on 64 elements ~ 2000 pairs.
                # 4000 * 2000 = 8M ops. Might be slow in Python if done many times.
                # But we have 85s.
                
                # Strategy: Randomize removal to escape local optima
                # Or systematic: try all removals, pick best add
                best_swap = None
                best_score = -1
                
                # To speed up, we can limit the number of swap attempts per iteration
                # or use a heuristic.
                
                # Let's try a few random swaps
                for _ in range(10):
                    if time.time() > time_limit:
                        break
                    if len(selected) == 0:
                        break
                    remove = random.choice(list(selected))
                    candidate = selected - {remove}
                    
                    for add in range(1, N + 1):
                        if add in candidate:
                            continue
                        if not has_conflict(candidate, add):
                            new_set = candidate | {add}
                            # Heuristic: prefer sets with more "diverse" gcds?
                            # For now, just size.
                            if len(new_set) > len(selected):
                                selected = new_set
                                improved = True
                                best_size = len(selected)
                                break
                            elif len(new_set) == len(selected):
                                # Optional: check if this swap leads to better future potential
                                # For now, accept to explore
                                # But we need to be careful not to loop
                                pass
                    if improved:
                        break

            if not improved:
                # Stagnation: try a 2-swap? Or random restart?
                # If we are stuck at 65, we might just be at the optimum.
                # Let's try a random restart if we've been stuck for a while
                if time.time() - last_improvement > 5:
                    # Random restart
                    selected = set(random.sample(range(1, N + 1), 50))
                    # Clean up invalid triples?
                    # Greedy repair
                    valid_selected = []
                    for x in sorted(selected):
                        if not has_conflict(valid_selected, x):
                            valid_selected.append(x)
                    selected = set(valid_selected)
                    last_improvement = time.time()

            if improved:
                last_improvement = time.time()
            
            # Safety break if no improvement for a long time and time is running out
            if time.time() > time_limit - 5:
                break

        return list(selected)

    best_set = []
    
    # Initial Strategy: Evens + 1
    s1 = list(range(2, N + 1, 2)) + [1]
    # Validate s1
    # We know it's valid, but let's be sure
    if is_valid_set(s1):
        best_set = s1
    else:
        # Repair
        valid = []
        for x in s1:
            if not has_conflict(valid, x):
                valid.append(x)
        best_set = valid

    # Local search on s1
    if time.time() < deadline:
        s1_improved = local_search(best_set, deadline - 10)
        if len(s1_improved) > len(best_set):
            best_set = s1_improved

    # Try another start: Powers of 2 + some others?
    # Or just random restarts in the remaining time
    if time.time() < deadline - 10:
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            random.seed()
            # Start with a random set of size ~60
            initial_size = 60
            random_set = set(random.sample(range(1, N + 1), initial_size))
            # Repair
            valid = []
            for x in sorted(random_set):
                if not has_conflict(valid, x):
                    valid.append(x)
            
            # Local search
            s_rand = local_search(valid, min(deadline - 2, time.time() + 5))
            if len(s_rand) > len(best_set):
                best_set = s_rand

    return best_set
