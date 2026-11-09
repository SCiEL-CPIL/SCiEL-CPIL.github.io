import time
import random
import sys

N = 12
DEADLINE = 80.0 # Safety margin

def compute_threshold(a_map, b_map):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Returns the length or None if not synchronizing.
    """
    # State is a bitmask of 12 bits
    # Start state is all 1s: (1 << N) - 1
    start = (1 << N) - 1
    
    # BFS
    visited = {start: 0}
    queue = [(start, 0)]
    
    while queue:
        state, dist = queue.pop(0)
        
        # Check if singleton
        if state & (state - 1) == 0:
            return dist
            
        # Try 'a'
        next_a = 0
        for i in range(N):
            if state & (1 << i):
                next_a |= (1 << a_map[i])
                
        if next_a not in visited:
            visited[next_a] = dist + 1
            queue.append((next_a, dist + 1))
            
        # Try 'b'
        next_b = 0
        for i in range(N):
            if state & (1 << i):
                next_b |= (1 << b_map[i])
                
        if next_b not in visited:
            visited[next_b] = dist + 1
            queue.append((next_b, dist + 1))
            
    return None

def is_synchronizing(a_map, b_map):
    """
    Quick check if the automaton synchronizes.
    """
    return compute_threshold(a_map, b_map) is not None

def construct_pawlik_variant(offset=0):
    """
    Constructs a Pawlik-style automaton with a specific topological arrangement.
    The Pawlik automaton has:
    - a: a long cycle (permutation)
    - b: collapses two adjacent states in the cycle
    
    We vary the offset to create different topological arrangements that
    may force different BFS paths in the subset lattice.
    """
    a = list(range(N))
    b = list(range(N))
    
    # Create a cycle for 'a': 0 -> 1 -> 2 -> ... -> N-1 -> 0
    for i in range(N):
        a[i] = (i + 1) % N
    
    # For 'b', collapse states at positions offset and offset+1
    # This creates a bottleneck that forces synchronization
    collapse_target = offset % N
    for i in range(N):
        if i == offset or i == (offset + 1) % N:
            b[i] = collapse_target
        else:
            b[i] = i
    
    return a, b

def construct():
    """
    Returns the transition tables for a synchronizing automaton with N=12 states,
    maximizing the shortest reset word length.
    
    Strategy: Instead of random mutation, we construct multiple Pawlik-style
    automata with different topological arrangements (offsets) and evaluate
    each one. We also try small perturbations of the best structure to see
    if we can exceed the known Pawlik threshold of 110.
    """
    start_time = time.time()
    
    best_a = None
    best_b = None
    best_score = 0
    
    # Try all possible offsets for the Pawlik structure
    for offset in range(N):
        a, b = construct_pawlik_variant(offset)
        score = compute_threshold(a, b)
        
        if score is not None and score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
    
    # If we found a good structure, try to improve it with targeted perturbations
    if best_score > 0 and best_a is not None:
        # Try swapping adjacent states in the 'a' cycle to create different cycle structures
        for i in range(N):
            for j in range(i+1, N):
                if time.time() > DEADLINE - 5:
                    break
                
                a = best_a[:]
                b = best_b[:]
                
                # Swap the images of i and j in 'a'
                a[i], a[j] = a[j], a[i]
                
                score = compute_threshold(a, b)
                
                if score is not None and score > best_score:
                    best_score = score
                    best_a = a[:]
                    best_b = b[:]
        
        # Try changing the collapse target in 'b'
        for target in range(N):
            if time.time() > DEADLINE - 5:
                break
                
            a = best_a[:]
            b = best_b[:]
            
            # Find the two states that are collapsed and change their target
            for i in range(N):
                if b[i] != i:
                    b[i] = target
                    break
            
            score = compute_threshold(a, b)
            
            if score is not None and score > best_score:
                best_score = score
                best_a = a[:]
                best_b = b[:]
    
    # If we haven't found anything good, fall back to a simple synchronizing automaton
    if best_a is None or best_score == 0:
        best_a = list(range(N))
        best_b = [0] * N
        for i in range(1, N):
            best_b[i] = i - 1
    
    return [best_a, best_b]

if __name__ == "__main__":
    # Quick self-test
    a, b = construct()
    print(f"Best score found: {compute_threshold(a, b)}")
    print(f"Time: {time.time()}")
