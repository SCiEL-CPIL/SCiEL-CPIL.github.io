import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    N = 1000
    
    # Helper: check if adding x to a set (given sum_counts array) is valid
    def can_add(x, sum_counts):
        # New sums: x + y for all y in S, and 2*x
        # We need sum_counts[s] + 1 <= 2 for all new sums s
        # i.e., sum_counts[s] <= 1 for all new sums
        for y in S:
            s = x + y
            if sum_counts[s] > 1:
                return False
        if sum_counts[2 * x] > 1:
            return False
        return True
    
    # Helper: add x to S and update sum_counts
    def do_add(x):
        S.append(x)
        for y in S[:-1]:
            s = x + y
            sum_counts[s] += 1
        sum_counts[2 * x] += 1
    
    # Helper: remove x from S and update sum_counts
    def do_remove(x):
        S.remove(x)
        for y in S:
            s = x + y
            sum_counts[s] -= 1
        sum_counts[2 * x] -= 1
    
    # Helper: rebuild sum_counts from scratch
    def build_counts(S_list):
        sc = np.zeros(N + 1, dtype=np.int32)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                sc[S_list[i] + S_list[j]] += 1
        return sc
    
    # Phase 1: Deterministic greedy baseline
    S = []
    sum_counts = np.zeros(N + 1, dtype=np.int32)
    for x in range(1, N + 1):
        if can_add(x, sum_counts):
            do_add(x)
    
    best_set = S.copy()
    best_size = len(S)
    
    # Phase 2: Local search with add/remove/swap
    # Use a list-based S for O(1) membership via set
    S_set = set(S)
    
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Restart with random greedy
            order = list(range(1, N + 1))
            random.shuffle(order)
            S = []
            S_set = set()
            sum_counts = np.zeros(N + 1, dtype=np.int32)
            for x in order:
                if can_add(x, sum_counts):
                    do_add(x)
                    S_set.add(x)
            no_improve = 0
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
            continue
        
        # Try random operations
        op = random.random()
        
        if op < 0.4:
            # Try to add a random element
            x = random.randint(1, N)
            if x not in S_set and can_add(x, sum_counts):
                do_add(x)
                S_set.add(x)
                no_improve = 0
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
            else:
                no_improve += 1
        
        elif op < 0.7:
            # Try to remove a random element (to free up sum capacity)
            if S:
                y = random.choice(S)
                do_remove(y)
                S_set.discard(y)
                no_improve += 1
        
        else:
            # Swap: remove one, try to add another
            if S:
                y = random.choice(S)
                do_remove(y)
                S_set.discard(y)
                
                # Try to add a random replacement
                z = random.randint(1, N)
                if z not in S_set and can_add(z, sum_counts):
                    do_add(z)
                    S_set.add(z)
                    no_improve = 0
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = S.copy()
                else:
                    # Put y back
                    do_add(y)
                    S_set.add(y)
                    no_improve += 1
    
    return best_set
