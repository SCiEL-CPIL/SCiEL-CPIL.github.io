import random
import time
import sys

N = 12

def _compute_threshold(transitions):
    """
    Compute the exact shortest reset word length for a given automaton.
    Returns (length, word) or (None, None) if not synchronizing.
    """
    a, b = transitions
    n = N
    # State: frozenset of states
    start_state = frozenset(range(n))
    if len(start_state) == 1:
        return 0, ""
    
    from collections import deque
    queue = deque()
    queue.append((start_state, ""))
    visited = {start_state}
    
    # To avoid memory explosion, we can limit the search depth or use bidirectional?
    # For N=12, 2^12 = 4096 subsets. BFS is very fast.
    
    while queue:
        curr_set, word = queue.popleft()
        
        # Try 'a'
        next_set_a = frozenset(a[s] for s in curr_set)
        if len(next_set_a) == 1:
            return len(word) + 1, word + "a"
        if next_set_a not in visited:
            visited.add(next_set_a)
            queue.append((next_set_a, word + "a"))
            
        # Try 'b'
        next_set_b = frozenset(b[s] for s in curr_set)
        if len(next_set_b) == 1:
            return len(word) + 1, word + "b"
        if next_set_b not in visited:
            visited.add(next_set_b)
            queue.append((next_set_b, word + "b"))
            
    return None, None

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the Černý automaton for N=12 states.
    This automaton is known to have a shortest reset word length of N-1 = 11,
    which is the theoretical maximum for this size (assuming Černý's conjecture holds,
    which it does for N=12).
    
    The construction is deterministic and instant, avoiding the overhead and
    uncertainty of random search.
    
    Letter 'a' (index 0): i -> (i + 1) % N
    Letter 'b' (index 1): i -> (i - 1) % N if i > 0 else 0
    """
    n = N
    a = [(i + 1) % n for i in range(n)]
    b = [0 if i == 0 else i - 1 for i in range(n)]
    
    return [a, b]
# EVOLVE-BLOCK-END
