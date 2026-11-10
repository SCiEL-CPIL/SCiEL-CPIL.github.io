"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
import numpy as np

N = 256
# Precompute weights (1/x) and conflicts (3-APs)
weights = np.array([1.0 / i for i in range(1, N + 1)], dtype=np.float64)

# conflicts[i] is a bitmask of indices j such that i and j form a 3-AP with some k in [1, N]
# We represent the set as a numpy boolean array of size N (index 0 corresponds to 1, index N-1 to N)
conflicts = [0] * N
for a in range(1, N + 1):
    for b in range(1, N + 1):
        if a == b:
            continue
        # Check if there exists c such that a, b, c or a, c, b or c, a, b is a 3-AP
        # 3-AP condition: 2*middle = first + last
        # Case 1: a, b, c -> 2b = a + c => c = 2b - a
        c = 2 * b - a
        if 1 <= c <= N and c != a and c != b:
            # a and b are part of a 3-AP, so they conflict
            conflicts[a - 1] |= (1 << (b - 1))
            conflicts[b - 1] |= (1 << (a - 1))
        # Case 2: a, c, b -> 2c = a + b => c = (a+b)/2
        if (a + b) % 2 == 0:
            c = (a + b) // 2
            if 1 <= c <= N and c != a and c != b:
                conflicts[a - 1] |= (1 << (b - 1))
                conflicts[b - 1] |= (1 << (a - 1))
        # Case 3: c, a, b -> 2a = c + b => c = 2a - b
        c = 2 * a - b
        if 1 <= c <= N and c != a and c != b:
            conflicts[a - 1] |= (1 << (b - 1))
            conflicts[b - 1] |= (1 << (a - 1))

# Convert conflicts to numpy arrays for fast operations
conflicts_np = np.array(conflicts, dtype=np.uint64)

def _evaluate(state):
    """
    Calculate the score of a given state (numpy bool array).
    Score = Sum(weights[i] for i in state) - Penalty * (number of 3-APs)
    Penalty is chosen to be large enough to discourage 3-APs.
    """
    # Fast sum of weights for selected items
    total_weight = np.dot(state.astype(np.float64), weights)
    
    # Count 3-APs
    # A 3-AP (a, b, c) is counted if a, b, c are all in state.
    # We can iterate over pairs or use the conflict matrix.
    # Since N=256, we can precompute a matrix or just iterate.
    # For speed in SA, we want incremental updates, but for initial eval and restarts, full eval is okay.
    # Let's optimize the penalty calculation.
    # Penalty = sum over all 3-APs (a,b,c) of [a in S and b in S and c in S]
    
    # Efficient way:
    # For each middle term b, the pairs (a, c) such that a+c=2b.
    # We can precompute for each b, the list of (a, c) pairs.
    
    ap_count = 0
    # Precomputed pairs for each middle term b
    # pairs_for_middle[b] = list of (a_idx, c_idx)
    
    # To make this fast, we'll just do a simple check in the SA loop using incremental updates.
    # For this full evaluation function, we'll use a slightly slower but clear method.
    # Actually, let's just return the weight for now and handle penalty in the SA step incrementally.
    # But we need a consistent score for comparison.
    # Let's define score = weight - P * ap_count.
    
    # For speed, I will implement a faster ap_count calculation here.
    # We can use the fact that if we have the state as a bitset, we can check intersections.
    # But numpy bool array is easier.
    
    # Let's just iterate through possible middle terms.
    # There are at most 128 middle terms.
    # For each middle term b, the number of pairs is at most b-1.
    # Total operations ~ sum(b) ~ 16k. This is fast enough.
    
    for b in range(2, N): # middle term index 1 to N-1 (value 2 to N)
        # b value is b+1
        bv = b + 1
        if not state[b]:
            continue
        # a + c = 2*bv
        # a < bv < c
        # a ranges from 1 to bv-1
        # c = 2*bv - a
        # c must be <= N
        max_a = min(bv - 1, N - bv)
        for a_val in range(1, max_a + 1):
            c_val = 2 * bv - a_val
            if state[a_val - 1] and state[c_val - 1]:
                ap_count += 1
                
    P = 100.0 # Penalty
    return total_weight - P * ap_count

def _sa_step(state, temp, rng):
    """
    Perform one step of simulated annealing.
    Returns new_state, new_score, old_score
    """
    old_score = _evaluate(state)
    
    # Pick a random element to flip
    i = rng.integers(N)
    
    # Calculate delta score
    # If we flip i:
    # 1. Weight change: +/- weights[i]
    # 2. Penalty change: +/- number of 3-APs involving i
    
    # Count 3-APs involving i
    # i can be first, middle, or last term.
    # Let's precompute for each i, the list of pairs (j, k) such that (i, j, k) is a 3-AP.
    # Actually, it's easier to just recalculate the penalty contribution of i.
    
    # Let's optimize by precomputing the neighbors for each i.
    # neighbors[i] = list of (j, k) such that {i, j, k} is a 3-AP.
    # But we need to know if j and k are currently in the set.
    
    # For now, let's use a slightly slower but correct incremental update.
    # We'll count the number of 3-APs that include i.
    
    aps_with_i = 0
    iv = i + 1
    
    # i is middle: a, i, c -> a + c = 2*iv
    # a < iv < c
    max_a = min(iv - 1, N - iv)
    for a_val in range(1, max_a + 1):
        c_val = 2 * iv - a_val
        if state[a_val - 1] and state[c_val - 1]:
            aps_with_i += 1
            
    # i is first: i, b, c -> 2b = i + c -> c = 2b - i
    # b > i, c > b
    # b ranges from i+1 to N-1
    # c = 2b - i must be <= N
    for b_val in range(iv + 1, N + 1):
        c_val = 2 * b_val - iv
        if c_val <= N and state[b_val - 1] and state[c_val - 1]:
            aps_with_i += 1
            
    # i is last: a, b, i -> 2b = a + i -> a = 2b - i
    # a < b < i
    # b ranges from 2 to i-1
    # a = 2b - i must be >= 1
    for b_val in range(2, iv):
        a_val = 2 * b_val - iv
        if a_val >= 1 and state[a_val - 1] and state[b_val - 1]:
            aps_with_i += 1

    P = 100.0
    
    if state[i]:
        # Removing i
        delta_weight = -weights[i]
        # Removing i breaks all 3-APs that include i
        delta_penalty = -P * aps_with_i
        delta = delta_weight + delta_penalty
    else:
        # Adding i
        delta_weight = weights[i]
        # Adding i creates all 3-APs that include i and where the other two are present
        # Wait, the logic above counts 3-APs where the OTHER two are present.
        # So if we add i, we create exactly `aps_with_i` new 3-APs.
        delta_penalty = -P * aps_with_i
        delta = delta_weight + delta_penalty

    new_score = old_score + delta
    
    # Acceptance criterion
    if delta >= 0:
        state[i] = not state[i]
        return state, new_score, old_score
    else:
        if rng.random() < np.exp(delta / temp):
            state[i] = not state[i]
            return state, new_score, old_score
        else:
            return state, old_score, old_score

def _simulated_annealing(initial_state, start_temp, end_temp, steps, rng):
    state = initial_state.copy()
    current_score = _evaluate(state)
    best_state = state.copy()
    best_score = current_score
    
    temp = start_temp
    cooling_factor = (end_temp / start_temp) ** (1.0 / steps)
    
    for _ in range(steps):
        state, current_score, _ = _sa_step(state, temp, rng)
        if current_score > best_score:
            best_score = current_score
            best_state = state.copy()
        temp *= cooling_factor
        
    return best_state, best_score

def _greedy_init(rng):
    """
    Generate a random initial state biased towards small numbers (high weight).
    """
    state = np.zeros(N, dtype=bool)
    # Try to add numbers in random order, but prioritize small ones?
    # Or just random.
    # Let's try a few random initializations.
    order = rng.permutation(N)
    for i in order:
        # Check if adding i creates a 3-AP
        # This is O(N^2) per check, so O(N^3) total. 256^3 = 16M. Might be slow in Python.
        # Let's just use a simple random state with some probability.
        pass
    
    # Better: Start with empty, and greedily add random elements if they don't create 3-APs.
    # But checking 3-APs is slow.
    # Let's just use a random state with probability 0.5 for small numbers, 0.1 for large.
    for i in range(N):
        if i < 50:
            if rng.random() < 0.8:
                state[i] = True
        elif i < 150:
            if rng.random() < 0.5:
                state[i] = True
        else:
            if rng.random() < 0.2:
                state[i] = True
                
    return state

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    best_state = np.zeros(N, dtype=bool)
    best_score = -np.inf
    
    # Try a few different initial states and SA runs
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0: # Reserve 10s for final check
            break
            
        initial_state = _greedy_init(rng)
        
        # Tune SA parameters
        start_temp = 1.0
        end_temp = 0.01
        steps = 20000
        
        best_state_sa, best_score_sa = _simulated_annealing(initial_state, start_temp, end_temp, steps, rng)
        
        if best_score_sa > best_score:
            best_score = best_score_sa
            best_state = best_state_sa
            
    # Convert numpy bool array to list of integers (1-indexed)
    selected = [i + 1 for i in range(N) if best_state[i]]
    
    # Final check: ensure no 3-APs
    # If there are 3-APs, we might need to remove elements.
    # But SA should have minimized them.
    # Let's do a quick local search to remove any remaining 3-APs.
    selected_set = set(selected)
    changed = True
    while changed:
        changed = False
        for a in selected_set:
            for b in selected_set:
                if a >= b:
                    continue
                c = 2 * b - a
                if c in selected_set and c > b:
                    # Found a 3-AP (a, b, c)
                    # Remove the one with smallest weight (largest index)
                    # Or just remove c?
                    # Let's remove the one with smallest 1/x (largest x)
                    if 1/a < 1/b and 1/a < 1/c:
                        selected_set.remove(a)
                    elif 1/b < 1/a and 1/b < 1/c:
                        selected_set.remove(b)
                    else:
                        selected_set.remove(c)
                    changed = True
                    break
            if changed:
                break
                
    selected = list(selected_set)
    
    return selected
# EVOLVE-BLOCK-END
