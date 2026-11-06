import time
import random

def _count_progression_diffs(S):
    """Count distinct d > 0 such that exists a with a, a+d, a+2d in S."""
    S_set = set(S)
    diffs = set()
    for a in S:
        for b in S:
            d = b - a
            if d > 0:
                if a + 2 * d in S_set:
                    diffs.add(d)
    return len(diffs)

def _count_progression_diffs_fast(S):
    """Faster counting using sorted list and set lookup."""
    S_set = set(S)
    S_sorted = sorted(S)
    diffs = set()
    n = len(S_sorted)
    for i in range(n):
        a = S_sorted[i]
        for j in range(i + 1, n):
            d = S_sorted[j] - a
            if a + 2 * d in S_set:
                diffs.add(d)
    return len(diffs)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s
    
    random.seed(42)
    N = 32
    MAXV = 512
    
    # Start with a good initial construction
    # Use a structured set: try to maximize progression differences
    # A good strategy: use numbers that form many 3-term arithmetic progressions
    
    # Initial: try a set based on modular arithmetic or specific patterns
    # Let's start with a random set and improve via simulated annealing
    
    current = set(random.sample(range(1, MAXV + 1), N))
    current_score = _count_progression_diffs_fast(current)
    best = set(current)
    best_score = current_score
    
    # Simulated annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    max_iters = 200000
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Generate a move: swap one element with a random new element
        if random.random() < 0.5:
            # Remove one element and add a random new one
            to_remove = random.choice(list(current))
            to_add = random.randint(1, MAXV)
            if to_add in current:
                continue
            current.discard(to_remove)
            current.add(to_add)
        else:
            # Swap two elements
            a, b = random.sample(list(current), 2)
            new_a = random.randint(1, MAXV)
            new_b = random.randint(1, MAXV)
            if new_a in current or new_b in current:
                continue
            if new_a == new_b:
                continue
            current.discard(a)
            current.discard(b)
            current.add(new_a)
            current.add(new_b)
        
        new_score = _count_progression_diffs_fast(current)
        
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = set(current)
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if random.random() < pow(2.718281828, delta / T):
                current_score = new_score
            else:
                # Revert
                if random.random() < 0.5:
                    current.add(to_remove)
                    current.discard(to_add)
                else:
                    current.add(a)
                    current.add(b)
                    current.discard(new_a)
                    current.discard(new_b)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return list(best)
