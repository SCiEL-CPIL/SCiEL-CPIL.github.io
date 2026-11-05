import time
import random

N = 12
FULL_STATE = (1 << N) - 1

def calculate_threshold(a_trans, b_trans):
    # a_trans and b_trans are lists of length N
    # Returns the shortest reset word length, or -1 if not synchronizing
    
    # Precompute next states for each subset? No, subsets are dynamic.
    # But we can precompute the effect of a and b on single states, 
    # and then apply to subsets using bit operations?
    # Actually, for a subset S, delta_a(S) = {a_trans[i] for i in S}.
    # We can represent S as a bitmask.
    # To compute delta_a(S):
    #   We need to map each set bit i in S to a_trans[i].
    #   This can be done by iterating over set bits.
    
    # Optimization: Precompute for each letter and each state i, the bit mask 1 << a_trans[i].
    # Then for a subset S, delta_a(S) is the OR of (1 << a_trans[i]) for all i in S.
    
    a_bits = [1 << a_trans[i] for i in range(N)]
    b_bits = [1 << b_trans[i] for i in range(N)]
    
    def next_state_mask(mask, letter_bits):
        res = 0
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            res |= letter_bits[idx]
            m ^= lsb
        return res

    # BFS
    visited = [False] * (1 << N)
    visited[FULL_STATE] = True
    queue = [(FULL_STATE, 0)]
    
    while queue:
        state, dist = queue.pop(0)
        if state.bit_count() == 1:
            return dist
            
        # Try 'a'
        next_a = next_state_mask(state, a_bits)
        if not visited[next_a]:
            visited[next_a] = True
            queue.append((next_a, dist + 1))
            
        # Try 'b'
        next_b = next_state_mask(state, b_bits)
        if not visited[next_b]:
            visited[next_b] = True
            queue.append((next_b, dist + 1))
            
    return -1

def construct():
    # Start with a cycle for 'a'
    best_a = [(i + 1) % N for i in range(N)]
    
    # Initialize 'b' with a known good pattern or random
    # A pattern that often works well: b[i] = (i + 1) % N for i > 0, b[0] = 0
    # This gives threshold N-1.
    # Let's start with something slightly different to allow search to climb.
    # Or just random.
    # Let's try a few random starts and keep the best.
    
    start_time = time.time()
    timeout = 55.0
    best_score = -1
    best_b = None
    
    # We will run local search for a few different initial b's
    # Or just one long local search.
    
    # Let's use a simple local search.
    # Current b
    curr_b = [random.randint(0, N-1) for _ in range(N)]
    curr_score = calculate_threshold(best_a, curr_b)
    
    if curr_score > best_score:
        best_score = curr_score
        best_b = curr_b[:]
        
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() - start_time < timeout:
        if no_improve > max_no_improve:
            # Random restart
            curr_b = [random.randint(0, N-1) for _ in range(N)]
            curr_score = calculate_threshold(best_a, curr_b)
            if curr_score > best_score:
                best_score = curr_score
                best_b = curr_b[:]
            no_improve = 0
            continue
            
        # Pick a random state to change
        i = random.randint(0, N-1)
        old_val = curr_b[i]
        new_val = random.randint(0, N-1)
        if new_val == old_val:
            continue
            
        curr_b[i] = new_val
        new_score = calculate_threshold(best_a, curr_b)
        
        if new_score > curr_score:
            curr_score = new_score
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_b = curr_b[:]
                # If we hit the theoretical max, stop
                if best_score == (N-1)**2:
                    break
        else:
            curr_b[i] = old_val
            no_improve += 1
            
    if best_b is None:
        # Fallback to a simple synchronizing automaton
        best_b = [0] * N
        
    return [best_a, best_b]
