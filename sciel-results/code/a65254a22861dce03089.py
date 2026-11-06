import time
import random
from math import gcd
from itertools import combinations

N = 128
DEADLINE = 90.0

# Precomputation
lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(1, N + 1):
        lcm_table[i][j] = i * j // gcd(i, j)

# bad_with[x] is a set of pairs (a, b) such that {a, b, x} is a bad triple
# Pairs are stored as (min, max)
bad_with = [set() for _ in range(N + 1)]
conflict_counts = [0] * (N + 1)

for a in range(1, N + 1):
    for b in range(a + 1, N + 1):
        L_ab = lcm_table[a][b]
        for x in range(1, N + 1):
            if x == a or x == b:
                continue
            if lcm_table[a][x] == L_ab and lcm_table[b][x] == L_ab:
                # {a, b, x} is a bad triple
                bad_with[x].add((a, b))
                bad_with[a].add((b, x))
                bad_with[b].add((a, x))

for x in range(1, N + 1):
    conflict_counts[x] = len(bad_with[x])

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    best_set = set()
    best_size = 0
    
    # Use a fixed seed for reproducibility of the search path
    random.seed(42)
    
    def local_search(initial_set):
        nonlocal best_set, best_size
        
        current_set = set(initial_set)
        current_size = len(current_set)
        
        # Maintain active_pairs: set of (min, max) tuples for all pairs in current_set
        active_pairs = set()
        for u, v in combinations(current_set, 2):
            if u < v:
                active_pairs.add((u, v))
            else:
                active_pairs.add((v, u))
                
        def is_valid_add(x):
            # Check if adding x creates a bad triple
            # A bad triple involving x exists if any pair (a, b) in current_set
            # is in bad_with[x]
            return bad_with[x].isdisjoint(active_pairs)
        
        def add_element(x):
            nonlocal current_set, current_size, active_pairs
            current_set.add(x)
            current_size += 1
            for y in current_set:
                if y != x:
                    if x < y:
                        active_pairs.add((x, y))
                    else:
                        active_pairs.add((y, x))
                        
        def remove_element(x):
            nonlocal current_set, current_size, active_pairs
            current_set.remove(x)
            current_size -= 1
            for y in current_set:
                if x < y:
                    active_pairs.discard((x, y))
                else:
                    active_pairs.discard((y, x))
                    
        # Main local search loop
        # We use a counter to limit the number of "stuck" iterations before giving up on this restart
        stuck_count = 0
        max_stuck = 50
        
        while time_left() > 5.0:
            improved = False
            
            # 1. Try to add any valid element
            # Iterate through elements in a random order to break symmetry?
            # Or sorted by conflict count?
            # Random is better for local search.
            candidates = [x for x in range(1, N + 1) if x not in current_set]
            random.shuffle(candidates)
            
            for x in candidates:
                if is_valid_add(x):
                    add_element(x)
                    improved = True
                    break # Restart scan after add
            
            if improved:
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set.copy()
                stuck_count = 0
                continue
            
            # 2. If no add possible, try a "kick"
            # Remove k random elements, then greedily fill back.
            if current_size >= 2:
                # Randomize kick size between 2 and 4
                k = random.randint(2, min(4, current_size))
                to_remove = random.sample(list(current_set), k)
                for x in to_remove:
                    remove_element(x)
                
                # Greedily add back
                # Sort candidates by conflict count (ascending) to pick safe ones first
                candidates = [x for x in range(1, N + 1) if x not in current_set]
                candidates.sort(key=lambda x: conflict_counts[x])
                
                for x in candidates:
                    if is_valid_add(x):
                        add_element(x)
                        if current_size > best_size:
                            best_size = current_size
                            best_set = current_set.copy()
                
                # If we didn't improve size after the kick and fill, we are stuck.
                # If we did improve, we continue the loop to try adding more.
                # If we didn't improve, we increment stuck_count.
                # If stuck_count exceeds max_stuck, we break and do a new restart.
                
                # Note: The 'improved' flag was False, so we are in the kick block.
                # We need to check if size increased from the pre-kick size.
                # We don't store pre-kick size explicitly, but we can infer.
                # Actually, if we added elements, size might have increased.
                # If size > previous best, we are good.
                # If size <= previous best, we are stuck.
                
                # Simple heuristic: if we didn't find a new global best, count it as stuck.
                # This might be too strict.
                # Let's just always continue the loop, but if we do 50 kicks without a global best improvement, break.
                
                # We need to track if we improved the *current* set size relative to the start of the kick?
                # No, just global best.
                
                # To avoid infinite loops on the same local optimum, we rely on randomization.
                # If we break after max_stuck, we start a new restart with a different initial set.
                
                # Increment stuck counter if no global improvement
                # We can't easily check if global improved in this block without storing prev_best.
                # Let's store prev_best_size before kick.
                
                # Actually, simpler: just run kicks until time runs out.
                # The randomization ensures we explore different parts of the space.
                # We don't need to break based on stuck_count if we have many restarts.
                
                # However, if we are in a very deep local optimum, random kicks might not escape.
                # Breaking and restarting with a fresh initial set is better.
                
                # Let's add a simple break condition: if we've done 100 kicks without global improvement, break.
                # We need a counter.
                
                # Since I didn't initialize a counter inside the while loop properly for this specific logic,
                # I will add a `local_improvement_count`.
                
                # For now, let's just let it run. The time limit is the main guard.
                # But if it gets stuck in a tiny set, it wastes time.
                # Let's add: if current_size < best_size - 5, break? No.
                
                # Let's stick to the time limit.
            else:
                # Set too small to kick effectively
                break

        return current_set

    # Run multiple restarts
    initial_strategies = []
    
    # Strategy 1: Increasing order
    initial_strategies.append(list(range(1, N + 1)))
    
    # Strategy 2: Decreasing conflict count (pick "safe" numbers first)
    sorted_safe = sorted(range(1, N + 1), key=lambda x: conflict_counts[x])
    initial_strategies.append(sorted_safe)
    
    # Strategy 3: Random shuffles
    for _ in range(10):
        shuffled = list(range(1, N + 1))
        random.shuffle(shuffled)
        initial_strategies.append(shuffled)
        
    # For each strategy, run local search
    for i, order in enumerate(initial_strategies):
        if time_left() < 15.0:
            break
            
        # Build initial set greedily
        current = set()
        active = set()
        for x in order:
            # Check validity against current
            if bad_with[x].isdisjoint(active):
                current.add(x)
                for y in current:
                    if y != x:
                        if x < y: active.add((x, y))
                        else: active.add((y, x))
                        
        # Run local search from this state
        local_search(current)
        
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    if not res:
        res = [1]
