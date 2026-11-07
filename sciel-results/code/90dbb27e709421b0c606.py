"""Optimized program: minimize interval length for distinct multiples of fixed divisors."""
import time
import math
import random
from itertools import product

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = 10

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # Precompute for fast ceiling division
    # For a given x and divisor d, the smallest multiple >= x is:
    # k = ceil(x / d) = (x + d - 1) // d
    # value = k * d

    # Strategy:
    # We want to find 10 distinct values a_i = k_i * d_i such that max(a_i) - min(a_i) is minimized.
    # Let L be the minimum value and R be the maximum.
    # We can iterate over possible "centers" or "starts" of the window.
    # However, a more effective approach for this specific problem size is to use a
    # branch-and-bound or a smart local search.
    
    # Given the small number of divisors (10) and the structure, we can try to find
    # a tight cluster. The optimal window is likely quite small.
    
    # Let's try a different approach:
    # 1. Generate candidate values for each divisor in a range.
    # 2. Use a constraint satisfaction / backtracking approach with pruning.
    
    # But backtracking might be too slow if the window is large.
    
    # Let's combine:
    # 1. Use simulated annealing to find a good initial window.
    # 2. Then do a precise local search to minimize the window.
    
    # Actually, let's try a direct search over the "minimum" value.
    # The minimum value must be a multiple of one of the divisors.
    # Let's guess that the optimal window is small, say < 1000.
    # We can iterate over possible minimum values L.
    # For each L, we try to find the smallest R such that we can pick distinct multiples.
    
    # This is equivalent to:
    # For a fixed L, what is the minimum R such that for each d_i, there is a multiple
    # in [L, R] and all chosen multiples are distinct?
    
    # This is still a complex problem.
    
    # Let's stick to a robust heuristic:
    # 1. Simulated Annealing to get close to the optimum.
    # 2. Local search to refine.
    
    random.seed(42)
    
    def get_a(state):
        return [k * d for k, d in zip(state, DIVISORS)]
    
    def evaluate(a):
        if any(x < 1 or x > MAX_VAL for x in a):
            return 10**10
        if len(set(a)) != N:
            return 10**10
        return max(a) - min(a)
    
    def init_state():
        # Initialize with random small multiples
        state = [1 + random.randint(0, 50) for _ in range(N)]
        return state
    
    best_obj = float('inf')
    best_state = None
    
    # Run multiple random restarts
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        state = init_state()
        a = get_a(state)
        obj = evaluate(a)
        
        T = 50.0
        cooling = 0.95
        iters = 0
        max_iters = 5000
        
        while iters < max_iters and time.time() < deadline - 5.0:
            iters += 1
            idx = random.randint(0, N - 1)
            step = random.randint(-3, 3)
            if step == 0:
                continue
            
            new_state = state[:]
            new_state[idx] += step
            if new_state[idx] < 1:
                new_state[idx] = 1
            if new_state[idx] * DIVISORS[idx] > MAX_VAL:
                continue
                
            new_a = get_a(new_state)
            new_obj = evaluate(new_a)
            
            if new_obj < obj or random.random() < math.exp((obj - new_obj) / max(T, 1e-9)):
                state = new_state
                a = new_a
                obj = new_obj
                if obj < best_obj:
                    best_obj = obj
                    best_state = state[:]
            
            T *= cooling
            if T < 0.1:
                T = 1.0
                state = init_state()
                a = get_a(state)
                obj = evaluate(a)
                if obj < best_obj:
                    best_obj = obj
                    best_state = state[:]
        
        # Local search on best_state
        if best_state is not None:
            final_state = best_state[:]
            final_a = get_a(final_state)
            final_obj = max(final_a) - min(final_a)
            
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                min_idx = final_a.index(min(final_a))
                max_idx = final_a.index(max(final_a))
                
                # Try to decrease max
                k = final_state[max_idx]
                if k > 1:
                    new_k = k - 1
                    new_val = new_k * DIVISORS[max_idx]
                    others = set(final_a)
                    others.discard(final_a[max_idx])
                    if new_val not in others and new_val >= min(final_a):
                        final_state[max_idx] = new_k
                        final_a[max_idx] = new_val
                        improved = True
                
                # Try to increase min
                k = final_state[min_idx]
                max_val = max(final_a)
                new_k = k + 1
                new_val = new_k * DIVISORS[min_idx]
                others = set(final_a)
                others.discard(final_a[min_idx])
                if new_val not in others and new_val <= max_val:
                    final_state[min_idx] = new_k
                    final_a[min_idx] = new_val
                    improved = True
            
            final_obj = max(final_a) - min(final_a)
            if final_obj < best_obj:
                best_obj = final_obj
                best_state = final_state[:]
        
        # If we found a very small window, we can stop early
        if best_obj <= 10:
            break

    # Final result
    if best_state is None:
        # Fallback: just return the divisors themselves (which are distinct)
        return [int(d) for d in DIVISORS]
    
    result = [int(x) for x in get_a(best_state)]
    return result

# EVOLVE-BLOCK-END
