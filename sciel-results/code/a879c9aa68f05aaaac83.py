import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second margin
    
    def is_valid_addition(S, x, sum_counts):
        if sum_counts.get(2*x, 0) >= 2:
            return False
        for y in S:
            if y >= x:
                if sum_counts.get(x + y, 0) >= 2:
                    return False
        return True

    def add_to_set(S, x, sum_counts):
        S.add(x)
        sum_counts[2*x] = sum_counts.get(2*x, 0) + 1
        for y in S:
            if y > x:
                sum_counts[x + y] = sum_counts.get(x + y, 0) + 1

    def remove_from_set(S, x, sum_counts):
        S.remove(x)
        sum_counts[2*x] -= 1
        if sum_counts[2*x] == 0:
            del sum_counts[2*x]
        for y in S:
            if y > x:
                s = x + y
                sum_counts[s] -= 1
                if sum_counts[s] == 0:
                    del sum_counts[s]

    best_set = set()
    best_size = 0

    # 1. Initial Greedy (Seed)
    S = set()
    counts = {}
    for x in range(1, 1001):
        if is_valid_addition(S, x, counts):
            add_to_set(S, x, counts)
    best_set = S
    best_size = len(S)

    # 2. Randomized Greedy Restarts (GRASP)
    # Each restart: Shuffle 1..1000, add if valid.
    while time.time() < deadline - 10:
        S = set()
        counts = {}
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_addition(S, x, counts):
                add_to_set(S, x, counts)
        
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
            # Optional: Reset deadline? No, keep global deadline.
            # But if we find a better one, we might want to polish it.

    # 3. Local Search on Best Set
    # Try to swap elements to allow expansion.
    if time.time() < deadline - 2:
        S = set(best_set)
        counts = dict(best_set and {} or {}) # Rebuild counts? No, we lost it.
        # We need to rebuild counts for best_set
        S = set(best_set)
        counts = {}
        # Rebuild is O(N^2) which is 250k, fine.
        for x in S:
             add_to_set(S, x, counts) # Wait, add_to_set assumes x not in S. 
             # My add_to_set adds to S. So I should use a different init.
        
        # Let's rebuild properly
        S = set(best_set)
        counts = {}
        for x in sorted(S):
             # is_valid_addition is not needed if we know it's valid, but add_to_set updates counts.
             # We need to ensure we don't double count or mess up.
             # Just use add_to_set logic manually or a helper.
             pass
        
        # Easier: Just re-run the greedy add on the set? No, order matters for counts?
        # No, counts are just sums.
        # Let's write a helper to build counts from a set.
        def build_counts(S):
            c = {}
            for x in S:
                c[2*x] = c.get(2*x, 0) + 1
                for y in S:
                    if y > x:
                        c[x+y] = c.get(x+y, 0) + 1
            return c
        
        S = set(best_set)
        counts = build_counts(S)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try a few random swaps
            for _ in range(100):
                if time.time() > deadline - 2: break
                if not S: break
                u = random.choice(list(S))
                unused = [i for i in range(1, 1001) if i not in S]
                if not unused: break
                v = random.choice(unused)
                
                # Try swap u -> v
                remove_from_set(S, u, counts)
                if is_valid_addition(S, v, counts):
                    add_to_set(S, v, counts)
                    # Try to expand
                    old_size = len(S)
                    # Quick expansion: try all unused? Too slow.
                    # Try random sample of unused
                    random.shuffle(unused)
                    for w in unused[:50]:
                        if is_valid_addition(S, w, counts):
                            add_to_set(S, w, counts)
                    
                    if len(S) > old_size:
                        improved = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = set(S)
                            # Update best_set, but continue searching from current S
                        break # Restart swap loop with new S
                else:
                    # Revert
                    add_to_set(S, u, counts)

    return sorted(list(best_set))
