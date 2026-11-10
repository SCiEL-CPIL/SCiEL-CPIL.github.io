import time
from math import lcm, gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Precompute lcm table for [2, 256]
    # Index 0..254 corresponds to values 2..256
    N = 255
    vals = list(range(2, 257))
    val_to_idx = {v: i for i, v in enumerate(vals)}
    
    # lcm_table[i][j] = lcm(vals[i], vals[j])
    lcm_table = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(i, N):
            l = lcm(vals[i], vals[j])
            lcm_table[i][j] = l
            lcm_table[j][i] = l
    
    # Precompute reciprocal as float for fast scoring
    recip = [1.0/vals[i] for i in range(N)]
    
    def valid_set(S):
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if lcm(S[i], S[j]) <= 256:
                    return False
        return True

    def score(S):
        return sum(1.0/x for x in S)
    
    def score_set(indices):
        return sum(recip[i] for i in indices)
    
    def can_add(x_idx, current_indices):
        for y_idx in current_indices:
            if lcm_table[x_idx][y_idx] <= 256:
                return False
        return True
    
    def greedy_construct(order):
        """Build a set by greedily adding elements in the given order."""
        current = []
        current_score = 0.0
        for idx in order:
            if can_add(idx, current):
                current.append(idx)
                current_score += recip[idx]
        return current, current_score
    
    def local_search(current_indices, current_score):
        """Local search with 1-swap and 1-add improvements."""
        current = current_indices[:]
        cur_score = current_score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add any element
            for idx in range(N):
                if idx not in current:
                    if can_add(idx, current):
                        current.append(idx)
                        cur_score += recip[idx]
                        improved = True
                        break
            if improved:
                continue
            
            # Try 1-swap: remove one, add one
            best_swap = None
            best_gain = 1e-9
            for i, rem_idx in enumerate(current):
                # Try removing rem_idx
                remaining = current[:i] + current[i+1:]
                for add_idx in range(N):
                    if add_idx in current:
                        continue
                    if can_add(add_idx, remaining):
                        gain = recip[add_idx] - recip[rem_idx]
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (i, add_idx)
            
            if best_swap is not None:
                i, add_idx = best_swap
                rem_idx = current[i]
                current[i] = add_idx
                cur_score += recip[add_idx] - recip[rem_idx]
                improved = True
        
        return current, cur_score
    
    best_score = 0.0
    best_set = []
    
    # Multiple random restarts with different orderings
    random.seed(42)
    
    # First, try deterministic orders
    orders = [
        list(range(N)),  # natural order
        list(range(N-1, -1, -1)),  # reverse order
        [i for i in range(N) if vals[i] % 2 == 1],  # odd numbers first
        [i for i in range(N) if vals[i] % 2 == 0],  # even numbers first
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        current, cur_score = greedy_construct(order)
        if cur_score > best_score:
            best_score = cur_score
            best_set = [vals[i] for i in current]
        
        # Local search
        current, cur_score = local_search(current, cur_score)
        if cur_score > best_score:
            best_score = cur_score
            best_set = [vals[i] for i in current]
    
    # Random restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create a random ordering with some bias towards small values
        order = list(range(N))
        # Partial shuffle: keep first 20 elements mostly in place
        random.shuffle(order)
        # Sort by a random weight that favors small values
        weights = [1.0/vals[i] + random.random()*0.1 for i in range(N)]
        order.sort(key=lambda i: -weights[i])
        
        current, cur_score = greedy_construct(order)
        if cur_score > best_score:
            best_score = cur_score
            best_set = [vals[i] for i in current]
        
        # Local search with time check
        current, cur_score = local_search(current, cur_score)
        if cur_score > best_score:
            best_score = cur_score
            best_set = [vals[i] for i in current]
    
    # Final validation
    if best_set:
        if not valid_set(best_set):
            # Fallback to a known valid set
            best_set = []
            for x in range(2, 257):
                if all(lcm(x, y) > 256 for y in best_set):
                    best_set.append(x)
            best_score = score(best_set)
    
    return best_set
