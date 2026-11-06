"""
Černý problem: 12-state binary automata with long reset thresholds.
Uses bitmask BFS for fast evaluation and Simulated Annealing to search
for automata with high reset thresholds.
"""
import time
import random
from collections import deque

N = 12
FULL_SET = (1 << N) - 1  # bitmask for all 12 states

# Deadline guard
DEADLINE = 85.0  # seconds (safety margin before 90s kill)

def _apply_mask(a, b, mask):
    """Apply letter a to all states in mask, return new bitmask."""
    result = 0
    for i in range(N):
        if mask & (1 << i):
            result |= (1 << a[i])
    return result

def _apply_mask_b(a, b, mask):
    """Apply letter b to all states in mask, return new bitmask."""
    result = 0
    for i in range(N):
        if mask & (1 << i):
            result |= (1 << b[i])
    return result

def _reset_length_bitmask(a, b):
    """
    Compute shortest reset word length via BFS on subset automaton using bitmasks.
    Returns -1 if not synchronizing, else the exact shortest reset word length.
    """
    if len(set(a)) == 1 or len(set(b)) == 1:
        # One letter is constant, so it resets in 1 step
        return 1
    
    start = FULL_SET
    if start == 0:
        return -1
    
    visited = set()
    visited.add(start)
    queue = deque()
    queue.append((start, 0))
    
    while queue:
        mask, dist = queue.popleft()
        
        if mask & (mask - 1) == 0:  # single bit set -> reset
            return dist
        
        # Apply 'a'
        new_mask_a = 0
        for i in range(N):
            if mask & (1 << i):
                new_mask_a |= (1 << a[i])
        
        if new_mask_a not in visited:
            visited.add(new_mask_a)
            queue.append((new_mask_a, dist + 1))
        
        # Apply 'b'
        new_mask_b = 0
        for i in range(N):
            if mask & (1 << i):
                new_mask_b |= (1 << b[i])
        
        if new_mask_b not in visited:
            visited.add(new_mask_b)
            queue.append((new_mask_b, dist + 1))
    
    return -1  # not synchronizing

def _random_automaton():
    """Generate a random complete deterministic binary automaton."""
    a = [random.randint(0, N - 1) for _ in range(N)]
    b = [random.randint(0, N - 1) for _ in range(N)]
    return a, b

def _perturb(a, b):
    """Randomly perturb one or a few transitions."""
    a = a[:]
    b = b[:]
    # Change 1-3 random transitions
    num_changes = random.randint(1, 3)
    for _ in range(num_changes):
        if random.random() < 0.5:
            i = random.randint(0, N - 1)
            a[i] = random.randint(0, N - 1)
        else:
            i = random.randint(0, N - 1)
            b[i] = random.randint(0, N - 1)
    return a, b

def _is_synchronizing(a, b):
    """Check if the automaton synchronizes (returns True/False)."""
    return _reset_length_bitmask(a, b) >= 0

# EVOLVE-BLOCK-START
def construct():
    """
    Search for a synchronizing 12-state binary automaton with maximum
    shortest reset word length using Simulated Annealing.
    Returns (a, b) where a and b are length-N lists of destination states.
    """
    start_time = time.time()
    
    # Initialize with a known synchronizing automaton
    # Cycle + reset structure: a is a cycle, b maps some states to 0
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [0] * N
    best_score = _reset_length_bitmask(best_a, best_b)
    
    # Ensure we start with a synchronizing automaton
    if best_score < 0:
        # Try a few random starts
        for attempt in range(100):
            a, b = _random_automaton()
            score = _reset_length_bitmask(a, b)
            if score >= 0:
                best_a, best_b = a, b
                best_score = score
                break
        else:
            # Fallback: use the simple cycle+reset
            best_a = [(i + 1) % N for i in range(N)]
            best_b = [0] * N
            best_score = _reset_length_bitmask(best_a, best_b)
    
    # Simulated Annealing
    current_a, current_b = best_a[:], best_b[:]
    current_score = best_score
    
    # SA parameters
    T_start = 10.0
    T_end = 0.01
    alpha = 0.995  # cooling rate
    T = T_start
    
    iterations = 0
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < DEADLINE:
        # Perturb
        new_a, new_b = _perturb(current_a, current_b)
        new_score = _reset_length_bitmask(new_a, new_b)
        
        iterations += 1
        
        if new_score >= 0:
            # Valid (synchronizing)
            if new_score > current_score:
                # Accept improvement
                current_a, current_b = new_a, new_b
                current_score = new_score
                no_improve = 0
                
                if new_score > best_score:
                    best_a, best_b = new_a[:], new_b[:]
                    best_score = new_score
                    no_improve = 0
            else:
                # Accept with probability based on temperature
                delta = current_score - new_score
                if random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                    current_a, current_b = new_a, new_b
                    current_score = new_score
                no_improve += 1
        else:
            # Not synchronizing - accept with lower probability
            if random.random() < pow(2.718281828, -5.0 / max(T, 1e-10)):
                current_a, current_b = new_a, new_b
                current_score = new_score
            no_improve += 1
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
        
        # If stuck for too long, restart from best or a new random point
        if no_improve > max_no_improve:
            if random.random() < 0.5:
                # Restart from best
                current_a, current_b = best_a[:], best_b[:]
                current_score = best_score
            else:
                # Try a new random starting point
                for attempt in range(50):
                    a, b = _random_automaton()
                    score = _reset_length_bitmask(a, b)
                    if score >= 0:
                        current_a, current_b = a, b
                        current_score = score
                        break
                else:
                    current_a, current_b = best_a[:], best_b[:]
                    current_score = best_score
            no_improve = 0
            T = max(T, T_start * 0.5)  # Reheat slightly
    
    # Final check: make sure best is valid
    final_score = _reset_length_bitmask(best_a, best_b)
    if final_score < 0:
        # Fallback to simple automaton
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
    
    return best_a, best_b
# EVOLVE-BLOCK-END
