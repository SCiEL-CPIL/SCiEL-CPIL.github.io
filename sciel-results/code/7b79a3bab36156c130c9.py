import random
import time

def construct():
    n = 512
    start_time = time.time()
    deadline = start_time + 55
    
    def check_add(x, selected):
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - d) in selected and (x - 2*d) in selected and (x - 3*d) in selected:
                return False
        return True

    def run_greedy(order):
        selected = set()
        for x in order:
            if check_add(x, selected):
                selected.add(x)
        return selected

    best_set = run_greedy(range(1, n + 1))
    
    # Randomized Greedy
    while time.time() < deadline - 5:
        order = list(range(1, n + 1))
        random.shuffle(order)
        s = run_greedy(order)
        if len(s) > len(best_set):
            best_set = s
            
    # Local Search with Kick-start
    current_size = len(best_set)
    while time.time() < deadline:
        S_trial = set(best_set)
        # Remove 1 random element
        if S_trial:
            x_to_remove = random.choice(list(S_trial))
            S_trial.remove(x_to_remove)
            
        # Re-greedy the rest? No, just try to add elements.
        # Since we removed an element, some previously invalid elements might now be valid.
        # We need to check ALL elements not in S_trial.
        # We can just do a single pass over all 1..512.
        for x in range(1, n + 1):
            if x not in S_trial and check_add(x, S_trial):
                S_trial.add(x)
                
        if len(S_trial) > current_size:
            best_set = S_trial
            current_size = len(S_trial)
            
    return sorted(best_set)
