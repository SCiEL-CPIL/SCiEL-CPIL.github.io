"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

def _compute_threshold(trans):
    """
    Compute the exact shortest reset word length for a 2-letter deterministic automaton.
    Returns the length, or -1 if not synchronizing.
    Uses BFS on the subset automaton.
    """
    n = len(trans[0])
    # Check if synchronizing: BFS from full set {0..n-1}
    # State: frozenset of states
    # Goal: any singleton
    
    full_set = frozenset(range(n))
    visited = {full_set}
    queue = deque([(full_set, 0)])
    
    while queue:
        current, dist = queue.popleft()
        if len(current) == 1:
            return dist
        
        for letter in range(2):
            t = trans[letter]
            next_set = frozenset(t[s] for s in current)
            if next_set not in visited:
                visited.add(next_set)
                queue.append((next_set, dist + 1))
    
    return -1  # Not synchronizing

def _is_synchronizing(trans):
    """Quick check if automaton is synchronizing."""
    return _compute_threshold(trans) != -1

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a synchronizing automaton with 12 states and 2 letters,
    maximizing the shortest reset word length.
    
    Strategy:
    1. Start with a known good construction (cycle + decrement) which gives threshold N-1=11.
    2. Use local search (hill climbing) to try to improve the threshold.
    3. Each step: pick a random state and letter, change its destination,
       check if still synchronizing, and if the threshold improved, keep it.
    4. Use a time budget to ensure we finish within limits.
    """
    # Initial construction: cycle for letter 0, decrement-with-fixed-point for letter 1
    # This is known to have reset threshold N-1 for a cycle automaton
    trans = [
        [(i + 1) % N for i in range(N)],  # Letter 0: cycle
        [0] + [i - 1 for i in range(1, N)]  # Letter 1: 0->0, i->i-1
    ]
    
    # Verify it's synchronizing
    initial_threshold = _compute_threshold(trans)
    if initial_threshold < 0:
        # Fallback to the simple construction
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    best_trans = [list(t) for t in trans]
    best_threshold = initial_threshold
    
    # Time budget: we need to finish well before 90s, leave margin for evaluator
    start_time = time.time()
    time_limit = 80.0  # Leave 10s margin
    
    # Local search: hill climbing with random restarts
    rng = random.Random(42)  # Fixed seed for reproducibility
    
    current_trans = [list(t) for t in trans]
    current_threshold = initial_threshold
    
    while time.time() - start_time < time_limit:
        # Pick a random transition to modify
        letter = rng.randint(0, 1)
        state = rng.randint(0, N - 1)
        old_dest = current_trans[letter][state]
        
        # Try a new destination
        new_dest = rng.randint(0, N - 1)
        if new_dest == old_dest:
            continue
        
        current_trans[letter][state] = new_dest
        
        # Check if still synchronizing and compute threshold
        new_threshold = _compute_threshold(current_trans)
        
        if new_threshold > 0:  # Synchronizing
            if new_threshold > best_threshold:
                best_threshold = new_threshold
                best_trans = [list(t) for t in current_trans]
                # Reset current to best for further exploration
                current_trans = [list(t) for t in best_trans]
            elif new_threshold >= current_threshold:
                # Keep the change if it doesn't make things worse
                current_threshold = new_threshold
            else:
                # Revert
                current_trans[letter][state] = old_dest
        else:
            # Not synchronizing, revert
            current_trans[letter][state] = old_dest
        
        # Occasionally do a random restart from best
        if rng.random() < 0.05:
            current_trans = [list(t) for t in best_trans]
            current_threshold = best_threshold
    
    return best_trans
# EVOLVE-BLOCK-END
