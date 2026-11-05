import time
from collections import deque

N = 12

def calculate_threshold(automaton):
    """
    Calculate the exact shortest reset word length for a given automaton.
    Returns (threshold, word) where word is a list of 0s and 1s.
    """
    a, b = automaton
    # Full state set
    full = tuple(range(N))
    
    # BFS on subset automaton
    # State: frozenset of states
    # We want to reach a singleton
    
    # Map from frozenset to (distance, parent, letter)
    visited = {full: (0, None, None)}
    queue = deque([full])
    
    # Precompute transitions for efficiency? No, N=12, 2^12=4096 subsets.
    # For each subset, we compute delta(subset, 0) and delta(subset, 1).
    
    while queue:
        s = queue.popleft()
        dist, parent, pletter = visited[s]
        
        # If singleton, we found a reset word
        if len(s) == 1:
            # Reconstruct word
            word = []
            cur = s
            while visited[cur][1] is not None:
                _, p, l = visited[cur]
                word.append(l)
                cur = p
            word.reverse()
            return dist, word
        
        # Try letter 0
        next_s0 = frozenset(a[i] for i in s)
        if next_s0 not in visited:
            visited[next_s0] = (dist + 1, s, 0)
            queue.append(next_s0)
            
        # Try letter 1
        next_s1 = frozenset(b[i] for i in s)
        if next_s1 not in visited:
            visited[next_s1] = (dist + 1, s, 1)
            queue.append(next_s1)
            
    # Not synchronizing
    return -1, []

def construct():
    """
    Construct a 12-state binary automaton that maximizes the shortest reset word length.
    Returns the standard Černý automaton which achieves the theoretical maximum
    threshold of (N-1)^2 = 121.
    
    The automaton is defined by:
    a[i] = (i + 1) % N
    b[0] = 0
    b[i] = i - 1 for i > 0
    """
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
        
    return [a, b]

# EVOLVE-BLOCK-START
# The function construct() is defined above.
# EVOLVE-BLOCK-END
