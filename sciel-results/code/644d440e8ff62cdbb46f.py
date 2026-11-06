import random
import time
import math

def construct():
    N = 256
    # Precompute reciprocals
    recips = [0.0] + [1.0 / i for i in range(1, N + 1)]
    
    # Precompute conflicts for each x
    # conflicts[x] is a list of pairs (a, b) such that {a, b, x} is a 3-AP
    conflicts = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        # x is the middle term: a = x-d, b = x+d
        max_d_mid = min(x - 1, N - x)
        for d in range(1, max_d_mid + 1):
            a = x - d
            b = x + d
            conflicts[x].append((a, b))
            
        # x is the end term: a = x-2d, b = x-d
        max_d_end = (x - 1) // 2
        for d in range(1, max_d_end + 1):
            a = x - 2 * d
            b = x - d
            conflicts[x].append((a, b))
            
        # x is the start term: a = x+d, b = x+2d
        max_d_start = (N - x) // 2
        for d in range(1, max_d_start + 1):
            a = x + d
            b = x + 2 * d
            conflicts[x].append((a, b))

    def is_valid_add(S, x):
        """Check if adding x to S results in a valid set."""
        for a, b in conflicts[x]:
            if a in S and b in S:
                return False
        return True

    def get_score(S):
        return sum(recips[x] for x in S)

    def greedy_init():
        S = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if is_valid_add(S, x):
                S.add(x)
        return S

    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin
    
    best_S = greedy_init()
    best_score = get_score(best_S)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    current_S = set(best_S)
    current_score = best_score
    
    # We will run SA for a while, then restart with a new greedy init if time permits
    while time.time() < deadline:
        # Check if we should restart with a new initial solution
        # If T is very low, we might be stuck.
        if T < T_min:
            # Try a new greedy init
            new_S = greedy_init()
            new_score = get_score(new_S)
            if new_score > best_score:
                best_score = new_score
                best_S = set(new_S)
                current_S = set(new_S)
                current_score = new_score
            # Reset T
            T = 1.0
            # Maybe keep the current best as starting point for next SA?
            # Or start from the new greedy?
            # Let's start from the new greedy to explore new regions.
            
        # Pick a random element to flip
        x = random.randint(1, N)
        
        if x in current_S:
            # Try to remove
            delta = -recips[x]
            # Acceptance probability
            if random.random() < math.exp(delta / T):
                current_S.remove(x)
                current_score += delta
        else:
            # Try to add
            if is_valid_add(current_S, x):
                delta = recips[x]
                # Always accept if valid (delta > 0)
                current_S.add(x)
                current_score += delta
        
        # Periodically try to fill the set (greedy fill)
        # This helps escape states where we removed a number but didn't replace it
        if random.random() < 0.05:
            changed = True
            while changed:
                changed = False
                # Iterate through all possible additions
                # To make it faster, we can just check all 1..N
                for y in range(1, N + 1):
                    if y not in current_S and is_valid_add(current_S, y):
                        current_S.add(y)
                        current_score += recips[y]
                        changed = True
        
        if current_score > best_score:
            best_score = current_score
            best_S = set(current_S)
            
        # Cool down
        T *= cooling

    return list(best_S)
