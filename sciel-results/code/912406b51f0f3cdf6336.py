import time
import random
from collections import deque

N = 12

def _compute_reset_length(transitions):
    """
    Compute the shortest reset word length for a given automaton.
    transitions: list of two lists of ints. transitions[0] for letter 'a', transitions[1] for 'b'.
    Returns the length of the shortest reset word, or -1 if not synchronizing.
    """
    # Start with the full set of states
    start_state = frozenset(range(N))
    if len(start_state) == 1:
        return 0
    
    # BFS on the subset automaton
    # State: frozenset of states
    # We want to reach a singleton state
    queue = deque()
    queue.append((start_state, 0))
    visited = {start_state}
    
    while queue:
        current_set, dist = queue.popleft()
        
        # If current_set is a singleton, we found a reset word
        if len(current_set) == 1:
            return dist
            
        # Try both letters
        for l in range(2):
            next_set = frozenset(transitions[l][s] for s in current_set)
            if next_set not in visited:
                visited.add(next_set)
                queue.append((next_set, dist + 1))
                
    return -1 # Not synchronizing

def _is_synchronizing(transitions):
    return _compute_reset_length(transitions) != -1

def construct():
    start_time = time.time()
    deadline = 50.0 # Safety margin
    
    best_transitions = None
    best_length = -1
    
    # Helper to generate a random automaton
    def random_automaton():
        return [[random.randint(0, N-1) for _ in range(N)] for _ in range(2)]
    
    # Initial phase: Find a synchronizing automaton and try to improve it
    # We will do a series of random restarts with local search
    
    while time.time() < deadline:
        # Generate a random automaton
        trans = random_automaton()
        
        # Check if it synchronizes
        length = _compute_reset_length(trans)
        if length == -1:
            continue
            
        # If this is the best so far, or if we just found one, start local search
        if length > best_length:
            best_length = length
            best_transitions = [row[:] for row in trans]
            
        # Local search (Hill Climbing)
        current_trans = [row[:] for row in trans]
        current_length = length
        
        # Limit local search iterations to avoid spending too much time on a bad start
        # But we can adapt based on time
        local_start = time.time()
        max_local_time = min(2.0, deadline - (time.time() - start_time)) # Spend at most 2s per restart or remaining time
        
        while time.time() - local_start < max_local_time:
            # Pick a random state and letter to change
            i = random.randint(0, N-1)
            l = random.randint(0, 1)
            
            # Save old value
            old_val = current_trans[l][i]
            
            # Change to a new random value
            new_val = random.randint(0, N-1)
            if new_val == old_val:
                continue
                
            current_trans[l][i] = new_val
            
            # Evaluate
            new_length = _compute_reset_length(current_trans)
            
            if new_length != -1 and new_length > current_length:
                # Accept
                current_length = new_length
                if new_length > best_length:
                    best_length = new_length
                    best_transitions = [row[:] for row in current_trans]
            else:
                # Reject, revert
                current_trans[l][i] = old_val
                
        # Check deadline
        if time.time() > deadline:
            break

    # If no synchronizing automaton was found (unlikely), return a trivial one
    if best_transitions is None:
        return [[(i + 1) % N for i in range(N)], [0] * N]
        
    return best_transitions
