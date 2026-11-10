import math
import time
import sys
from math import gcd
import random

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # Safety margin
    
    N = 512
    # Candidates are 1..511
    # Weights
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
        
    # Precompute prime factors
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    pf_cache = [None] * N
    for i in range(1, N):
        pf_cache[i] = prime_factors(i)
        
    # Greedy Initialization
    def greedy_init(order):
        chosen = []
        chosen_set = set()
        for a in order:
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
                chosen_set.add(a)
        return chosen, chosen_set

    # Standard order: descending
    best_order = list(range(511, 0, -1))
    best, best_set = greedy_init(best_order)
    best_score = sum(weights[a] for a in best)
    
    # Local Search: Simulated Annealing
    # Move: Swap one element with another, or Add/Remove
    # To keep it simple and effective, we try:
    # 1. Add any coprime a (Greedy fill)
    # 2. Swap b in chosen with a not in chosen if gain > 0
    
    def try_add(chosen, chosen_set):
        added = False
        # Try to add any unchosen a that is coprime to all
        # Sort by weight descending
        for a in best_order:
            if a in chosen_set:
                continue
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
                chosen_set.add(a)
                added = True
        return added

    # Run greedy fill first
    try_add(best, best_set)
    best_score = sum(weights[a] for a in best)
    
    # Simulated Annealing
    T = 0.1
    T_min = 0.001
    cooling = 0.999
    current = best[:]
    current_set = best_set.copy()
    current_score = best_score
    
    while time.time() < deadline - 5.0:
        if T < T_min:
            break
            
        # Pick a random move
        # 1. Swap
        if current:
            b = random.choice(current)
            # Find candidates a coprime to current \ {b}
            # To speed up, we can just check a few random a's or the top ones
            # Let's try swapping with a random unselected a
            # Or better, try swapping b with the best possible a
            
            # Optimization: Precompute list of unselected
            # This is O(N) per step, might be slow.
            # Let's just try a few random swaps
            
            # Try to find a better a
            best_gain = 0
            best_a = None
            
            # Check top 100 unselected numbers
            # We can maintain a list of unselected
            # For simplicity, sample random a
            
            for _ in range(20):
                a = random.randint(1, 511)
                if a in current_set:
                    continue
                # Check if a is coprime to all in current except b
                ok = True
                for c in current:
                    if c == b:
                        continue
                    if gcd(a, c) != 1:
                        ok = False
                        break
                if ok:
                    gain = weights[a] - weights[b]
                    if gain > best_gain:
                        best_gain = gain
                        best_a = a
                        
            if best_a is not None and best_gain > 0:
                # Accept
                current.remove(b)
                current_set.discard(b)
                current.append(best_a)
                current_set.add(best_a)
                current_score += best_gain
                if current_score > best_score:
                    best_score = current_score
                    best = current[:]
                    best_set = current_set.copy()
            elif random.random() < math.exp(best_gain / T) if best_gain < 0 else 1.0:
                # Accept worse move
                current.remove(b)
                current_set.discard(b)
                current.append(best_a)
                current_set.add(best_a)
                current_score += best_gain
        else:
            # If empty, just add
            a = random.randint(1, 511)
            if a not in current_set:
                current.append(a)
                current_set.add(a)
                current_score += weights[a]
        
        T *= cooling
        
    return best

