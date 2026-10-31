"""Rewritten program: search for optimal finite sets maximizing the sumset/difference-set exponent lower bound."""
import itertools
import math

# EVOLVE-BLOCK-START
import random

def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize the score."""
    
    def score_of(U):
        U = sorted(set(U))
        if not U or U[0] != 0:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        ratio = len(diffs) / len(sums)
        if ratio <= 0:
            return 0
        denom = math.log(2 * max_u + 1)
        if denom <= 0:
            return 0
        return 1 + math.log(ratio) / denom

    # Simulated Annealing
    random.seed(42)
    
    # Initial set
    current = [0, 1, 3, 7]
    current_score = score_of(current)
    
    best_set = list(current)
    best_score = current_score
    
    T = 1.0
    cooling = 0.99
    min_T = 0.01
    max_val = 200
    max_size = 20
    
    while T > min_T:
        # Generate a new state
        new_set = list(current)
        
        # Choose a move
        move = random.randint(0, 2)
        if move == 0: # Add
            if len(new_set) < max_size:
                x = random.randint(1, max_val)
                if x not in new_set:
                    new_set.append(x)
                else:
                    continue
        elif move == 1: # Remove
            if len(new_set) > 2:
                idx = random.randint(1, len(new_set)-1)
                new_set.pop(idx)
            else:
                continue
        else: # Replace
            if len(new_set) > 2:
                idx = random.randint(1, len(new_set)-1)
                old = new_set[idx]
                x = random.randint(1, max_val)
                if x != old and x not in new_set:
                    new_set[idx] = x
                else:
                    continue
        
        new_score = score_of(new_set)
        
        # Acceptance probability
        delta = new_score - current_score
        if delta > 0:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = list(current)
        else:
            if random.random() < math.exp(delta / T):
                current = new_set
                current_score = new_score
        
        T *= cooling
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
