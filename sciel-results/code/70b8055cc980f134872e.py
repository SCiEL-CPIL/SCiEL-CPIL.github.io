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

def construct():
    """
    Returns the transition tables for a synchronizing automaton with N=12 states,
    maximizing the shortest reset word length.
    """
    start_time = time.time()
    
    # Initial candidate: A known good structure or random
    # Let's start with a random synchronizing automaton or a heuristic one.
    # A common heuristic is to start with the "cycle" and "decrement" and modify.
    
    # Let's try the "Černý-like" structure that actually works.
    # For N=12, we want threshold 121.
    # One known optimal automaton is:
    # a: i -> i+1 mod N
    # b: i -> i-1 if i>0, 0->0 (This one loops for N>=3, so skip)
    
    # Let's use a random search with a strong initial seed.
    # Seed: A permutation for a, and a map for b that collapses 0 and shifts others.
    
    best_a = [0] * N
    best_b = [0] * N
    best_score = 0
    
    # Initialize with a random permutation for a
    perm = list(range(N))
    random.shuffle(perm)
    best_a = perm[:]
    
    # Initialize b to collapse 0 and shift others slightly
    # b: 0->0, i->i-1 for i>0
    for i in range(N):
        if i == 0:
            best_b[i] = 0
        else:
            best_b[i] = i - 1
            
    # Check if this initial one synchronizes
    score = compute_threshold(best_a, best_b)
    if score is not None:
        best_score = score
    else:
        # If not, try to fix it or start random
        # Force synchronization by making b collapse everything to 0?
        # That gives score 1.
        for i in range(N):
            best_b[i] = 0
        score = compute_threshold(best_a, best_b)
        best_score = score if score is not None else 0

    # Local Search
    while time.time() < DEADLINE:
        # Randomly perturb
        a = best_a[:]
        b = best_b[:]
        
        # Perturb a: swap two transitions
        i, j = random.sample(range(N), 2)
        a[i], a[j] = a[j], a[i]
        
        # Perturb b: change one transition
        k = random.randint(0, N-1)
        b[k] = random.randint(0, N-1)
        
        # Check synchronization and score
        # To save time, we can do a quick check first?
        # compute_threshold is BFS, which is fast for N=12.
        
        score = compute_threshold(a, b)
        
        if score is not None and score > best_score:
            best_score = score
            best_a = a
            best_b = b
            
            # If we reached the theoretical max, stop
            if best_score == (N-1)**2:
                break

    return [best_a, best_b]

if __name__ == "__main__":
    # Quick self-test
    a, b = construct()
    print(f"Best score found: {compute_threshold(a, b)}")
    print(f"Time: {time.time()}")
