import random
import time

N = 12
FULL_SET = (1 << N) - 1

def subset_transition(mask, transitions):
    """Compute the next subset mask given current mask and transition table for a letter."""
    new_mask = 0
    # Iterate through bits in mask
    m = mask
    while m:
        # Get lowest set bit
        lsb = m & -m
        state = lsb.bit_length() - 1
        new_mask |= 1 << transitions[state]
        m ^= lsb
    return new_mask

def compute_threshold(automaton):
    """
    Compute the shortest reset word length for a given automaton.
    automaton: list of two lists [trans_a, trans_b]
    Returns the length, or -1 if not synchronizing.
    """
    trans_a = automaton[0]
    trans_b = automaton[1]
    
    # BFS on subset automaton
    # Queue stores (current_mask, distance)
    visited = set()
    queue = [(FULL_SET, 0)]
    visited.add(FULL_SET)
    
    while queue:
        curr_mask, dist = queue.pop(0)
        
        # Check if singleton
        if curr_mask != 0 and (curr_mask & (curr_mask - 1)) == 0:
            return dist
            
        # Expand
        next_a = subset_transition(curr_mask, trans_a)
        next_b = subset_transition(curr_mask, trans_b)
        
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
            
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
            
    return -1

def is_synchronizing(automaton):
    """Quick check if synchronizing. Uses BFS but stops early if singleton found."""
    trans_a = automaton[0]
    trans_b = automaton[1]
    
    visited = set()
    queue = [FULL_SET]
    visited.add(FULL_SET)
    
    while queue:
        curr_mask = queue.pop(0)
        
        if curr_mask != 0 and (curr_mask & (curr_mask - 1)) == 0:
            return True
            
        next_a = subset_transition(curr_mask, trans_a)
        next_b = subset_transition(curr_mask, trans_b)
        
        if next_a not in visited:
            visited.add(next_a)
            queue.append(next_a)
            
        if next_b not in visited:
            visited.add(next_b)
            queue.append(next_b)
            
    return False

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    # Initialize with a known synchronizing automaton
    # Cycle for a, constant 0 for b
    current = [
        [(i + 1) % N for i in range(N)],
        [0] * N
    ]
    
    current_score = compute_threshold(current)
    best_automaton = [current[0][:], current[1][:]]
    best_score = current_score
    
    # Simulated Annealing / Hill Climbing
    # We will try to improve the score.
    # Temperature starts high, cools down.
    temp = 10.0
    cool_rate = 0.999
    
    while time.time() < deadline:
        # Create a neighbor
        neighbor = [current[0][:], current[1][:]]
        
        # Pick a random letter and state to modify
        letter_idx = random.randint(0, 1)
        state_idx = random.randint(0, N - 1)
        new_dest = random.randint(0, N - 1)
        
        # Skip if no change
        if neighbor[letter_idx][state_idx] == new_dest:
            continue
            
        neighbor[letter_idx][state_idx] = new_dest
        
        # Check if synchronizing
        if not is_synchronizing(neighbor):
            # Reject non-synchronizing moves (or penalize, but rejection is simpler and keeps us in valid space)
            # However, sometimes we need to jump over non-synchronizing regions?
            # For now, strict rejection.
            continue
            
        new_score = compute_threshold(neighbor)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0:
            current = neighbor
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_automaton = [current[0][:], current[1][:]]
        else:
            # Accept with probability exp(delta / temp)
            if random.random() < pow(2.71828, delta / temp):
                current = neighbor
                current_score = new_score
                
        temp *= cool_rate
        if temp < 0.1:
            temp = 0.1
            
    return best_automaton
