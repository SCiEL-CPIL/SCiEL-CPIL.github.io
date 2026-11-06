import time
import random
import sys
import numpy as np

# Global cache for priority sets
_PRIORITY_CACHE = {}

# Deadline guard
_START_TIME = time.time()
DEADLINE = _START_TIME + 310.0  # Leave 20s safety margin before 330s kill

def _int_to_tuple(idx, n):
    """Convert an integer index to a tuple of digits in base 3."""
    v = []
    x = idx
    for _ in range(n):
        v.append(x % 3)
        x //= 3
    return tuple(v)

def _tuple_to_int(v, n):
    """Convert a tuple of digits in base 3 to an integer index."""
    idx = 0
    for i, d in enumerate(reversed(v)):
        idx += d * (3 ** i)
    return idx

def _generate_initial_cap_set(n, seed, target_size_hint=None):
    """
    Generate an initial cap set using a fast greedy heuristic.
    This provides a valid starting point for local search.
    """
    rng = random.Random(seed)
    total = 3 ** n
    
    # Use a simple greedy approach with a specific ordering to get a decent start
    # Ordering: prioritize vectors with many zeros, then randomize within groups
    # This tends to create sets that are "spread out"
    
    indices = list(range(total))
    
    # Simple heuristic: sort by number of non-zero elements (Hamming weight over {1,2})
    # Actually, let's just use random shuffle but biased towards low indices or specific patterns
    # To be fast, we'll just do a standard random greedy with a limited time budget
    start = time.time()
    time_limit = 5.0 # Short time for initial generation
    
    S = set()
    forbidden = set()
    
    # Shuffle
    rng.shuffle(indices)
    
    for idx in indices:
        if time.time() - start > time_limit:
            break
        if idx in forbidden:
            continue
        
        v = _int_to_tuple(idx, n)
        
        # Check if adding v creates a cap set violation
        # A set is a cap set if it contains no 3-term arithmetic progressions.
        # For any a, b in S, the third point c = 2b - a (mod 3) must not be in S.
        # When adding v, we must check against all existing a in S:
        # c = 2v - a and c = 2a - v must not be in S.
        
        is_valid = True
        for a_idx in S:
            a = _int_to_tuple(a_idx, n)
            
            # Third point on line (a, v): c1 = 2v - a
            c1 = tuple((2 * v[i] - a[i]) % 3 for i in range(n))
            c1_idx = _tuple_to_int(c1, n)
            if c1_idx in S:
                is_valid = False
                break
            
            # Third point on line (v, a): c2 = 2a - v
            c2 = tuple((2 * a[i] - v[i]) % 3 for i in range(n))
            c2_idx = _tuple_to_int(c2, n)
            if c2_idx in S:
                is_valid = False
                break
            
        if is_valid:
            S.add(idx)
            # Update forbidden set for faster rejection? 
            # Actually, for the local search, we maintain the set S explicitly.
            # We don't strictly need a global forbidden set if we check against S directly,
            # but for large n, checking against S is O(|S|).
            # For n up to ~8-10, |S| is manageable (hundreds/thousands).
            
    return S

def _local_search_cap_set(n, initial_S, seed, time_limit):
    """
    Perform stochastic local search (Simulated Annealing) to improve the cap set size.
    Moves:
    1. Add a random vector not in S if it doesn't create a cap.
    2. Remove a random vector from S and try to replace it with a different one.
    3. Swap: Remove one, Add another.
    
    Objective: Maximize |S|.
    """
    rng = random.Random(seed)
    total = 3 ** n
    S = set(initial_S)
    best_S = set(S)
    best_size = len(S)
    
    start_time = time.time()
    end_time = start_time + time_limit
    
    # Precompute all vectors as tuples for fast access? 
    # For n=8, 3^8 = 6561. For n=10, 59049. 
    # Storing tuples is fine.
    all_vecs = [_int_to_tuple(i, n) for i in range(total)]
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    # We will run for a fixed number of steps or until time limit
    
    current_T = T_start
    
    # Helper to check if a vector can be added to S
    def can_add(v_idx, S):
        v = all_vecs[v_idx]
        for a_idx in S:
            if a_idx == v_idx:
                continue
            a = all_vecs[a_idx]
            # Check 2v - a
            c1 = tuple((2 * v[i] - a[i]) % 3 for i in range(n))
            if _tuple_to_int(c1, n) in S:
                return False
            # Check 2a - v
            c2 = tuple((2 * a[i] - v[i]) % 3 for i in range(n))
            if _tuple_to_int(c2, n) in S:
                return False
        return True

    # Helper to check if removing a vector and adding another is valid
    # This is equivalent to: is (S - {a}) + {v} a cap set?
    # This requires checking that v doesn't form a cap with any pair in S-{a},
    # AND that the pairs in S-{a} don't form a cap with each other (which they don't, since S was valid).
    # So we only need to check v against S-{a}.
    
    def can_swap(a_idx, v_idx, S):
        if v_idx in S:
            return False
        v = all_vecs[v_idx]
        for a2_idx in S:
            if a2_idx == a_idx or a2_idx == v_idx:
                continue
            a2 = all_vecs[a2_idx]
            # Check 2v - a2
            c1 = tuple((2 * v[i] - a2[i]) % 3 for i in range(n))
            if _tuple_to_int(c1, n) in S:
                return False
            # Check 2a2 - v
            c2 = tuple((2 * a2[i] - v[i]) % 3 for i in range(n))
            if _tuple_to_int(c2, n) in S:
                return False
        return True

    steps = 0
    while time.time() < end_time:
        steps += 1
        # Update temperature
        # Simple linear cooling
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / time_limit)
        current_T = T_start - (T_start - T_end) * progress
        
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.2:
            # Try to ADD a random vector
            # Sample a random vector not in S
            # To avoid infinite loops if S is large, we can sample a few times
            added = False
            for _ in range(10):
                v_idx = rng.randint(0, total - 1)
                if v_idx not in S:
                    if can_add(v_idx, S):
                        S.add(v_idx)
                        added = True
                        # Since we increased size, this is always an improvement (accept)
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
                        break
                    else:
                        # Rejection: keep state, maybe accept with prob exp(-delta/T)
                        # Delta = -1 (we want to increase size, so failing to increase is "worse" by 1 in objective? 
                        # No, objective is size. If we fail to add, size stays same. 
                        # If we accept a "bad" move, we would have to remove something. 
                        # Here, if we can't add, we just don't.
                        pass
            
        elif move_type < 0.6:
            # Try to SWAP: Remove one, Add one
            # This keeps size constant, but might allow future additions.
            if len(S) > 0:
                a_idx = rng.choice(list(S))
                # Sample a candidate v_idx
                for _ in range(10):
                    v_idx = rng.randint(0, total - 1)
                    if v_idx in S:
                        continue
                    if can_swap(a_idx, v_idx, S):
                        # Perform swap
                        S.remove(a_idx)
                        S.add(v_idx)
                        
                        # Acceptance: 
                        # Since size is unchanged, we accept if it helps future additions?
                        # Hard to evaluate. 
                        # Simpler heuristic: Accept with some probability to explore.
                        # Or, only accept if the new vector is "better" in some metric (e.g., fewer zeros)?
                        # Let's use a simple acceptance: always accept swap if valid, 
                        # but this can lead to random walks. 
                        # Better: Accept with probability exp(-1/T) to allow some noise.
                        if rng.random() < 0.5: # High acceptance rate for swaps
                            pass
                        else:
                            # Revert
                            S.remove(v_idx)
                            S.add(a_idx)
                            
        else:
            # Try to REMOVE a vector (to allow adding two later?)
            # This is risky as it decreases size.
            # Only accept if we can immediately add two back? 
            # Too complex. Let's stick to Add and Swap.
            # Or, just do a "kick": Remove a random vector.
            if len(S) > 1:
                a_idx = rng.choice(list(S))
                # Accept removal with low probability to escape local optima
                if rng.random() < 0.1:
                    S.remove(a_idx)
                    # This decreases size. 
                    # We only keep this if we find a better configuration later.
                    # Since we track best_S, this is safe.
                    # But we might get stuck with a smaller set.
                    # Let's revert immediately unless we add it back in next step?
                    # Actually, let's just not do pure removals. They are rarely beneficial 
                    # unless coupled with specific additions.
                    S.add(a_idx) # Revert for now

        # Periodic check for best
        # (Already done in add case)
        
        # If we have time, we can also try "2-swap" moves: remove 2, add 2.
        # This allows escaping local optima where no single swap works.
        # Given time constraints, we'll stick to 1-swap for now.
        
    return best_S

def priority(vector, n):
    if n not in _PRIORITY_CACHE:
        # Perform search for this n
        time_limit = 250.0  # Leave time for overhead
        
        best_set = set()
        best_size = 0
        
        start_search = time.time()
        
        # Strategy:
        # 1. Run multiple random initializations
        # 2. For each, run local search
        # 3. Keep the best
        
        seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111]
        
        # We'll run a few iterations of (Init + LocalSearch)
        # Each iteration should take ~10-20s to allow many trials.
        
        for i, seed in enumerate(seeds):
            if time.time() - start_search > time_limit:
                break
            
            # Estimate time remaining
            remaining = time_limit - (time.time() - start_search)
            if remaining < 5.0:
                break
            
            # Time budget for this trial: 
            # We want to fit ~10-15 trials.
            # Let's allocate 15s per trial.
            trial_time = min(15.0, remaining)
            
            # 1. Generate initial set
            init_time = min(2.0, trial_time * 0.1)
            S_init = _generate_initial_cap_set(n, seed, init_time)
            
            if len(S_init) > best_size:
                best_size = len(S_init)
                best_set = set(S_init)
            
            # 2. Local Search
            ls_time = trial_time - init_time - 0.5 # buffer
            if ls_time > 1.0:
                S_improved = _local_search_cap_set(n, S_init, seed, ls_time)
                
                if len(S_improved) > best_size:
                    best_size = len(S_improved)
                    best_set = set(S_improved)
            
            # Check time
            if time.time() - start_search > time_limit - 10.0:
                break
        
        # Final check: if we have time, run one more long local search on the best found so far
        remaining = time_limit - (time.time() - start_search)
        if remaining > 10.0 and best_set:
            S_final = _local_search_cap_set(n, best_set, 12345, remaining - 2.0)
            if len(S_final) > best_size:
                best_size = len(S_final)
                best_set = set(S_final)
        
        # Convert best_set (indices) to set of tuples
        best_tuples = set()
        for idx in best_set:
            best_tuples.add(_int_to_tuple(idx, n))
        
        _PRIORITY_CACHE[n] = best_tuples
    
    return 1.0 if vector in _PRIORITY_CACHE[n] else 0.0
