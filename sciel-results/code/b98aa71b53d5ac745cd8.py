"""
Černý problem: 12-state binary automata with long reset thresholds.

Strategy:
1. Use bitmask-based BFS to compute exact reset thresholds in <1ms.
2. Perform large-scale random restart local search optimizing both transition functions.
3. The local search uses simulated annealing / hill-climbing with random restarts.
"""

import random
import time

N = 12


def reset_threshold(trans_a, trans_b):
    """
    Compute the exact shortest reset word length using BFS on the subset automaton.
    trans_a, trans_b: lists of length N, where trans_a[i] is the destination of state i under letter 'a'.
    Returns the length of the shortest reset word, or -1 if not synchronizing.
    """
    # Full set represented as bitmask: all 12 bits set
    full_mask = (1 << N) - 1
    
    # BFS from full set
    # State is a bitmask representing the current subset of states
    from collections import deque
    
    visited = {full_mask: 0}
    queue = deque([full_mask])
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        # If we've reached a singleton, we found the reset word
        if current & (current - 1) == 0:
            return dist
        
        # Apply 'a' to all states in current subset
        next_a = 0
        for i in range(N):
            if current & (1 << i):
                next_a |= (1 << trans_a[i])
        
        # Apply 'b' to all states in current subset
        next_b = 0
        for i in range(N):
            if current & (1 << i):
                next_b |= (1 << trans_b[i])
        
        for nxt in (next_a, next_b):
            if nxt not in visited:
                visited[nxt] = dist + 1
                queue.append(nxt)
    
    return -1  # Not synchronizing


def construct():
    """
    Construct a synchronizing automaton with 12 states and two input letters,
    maximizing the length of its shortest reset word.
    
    Uses random restart local search with bitmask BFS for threshold computation.
    """
    start_time = time.time()
    timeout = 55.0  # Leave 5 seconds buffer
    
    best_trans_a = None
    best_trans_b = None
    best_score = -1
    
    # Precompute for speed: use lists for transitions
    # We'll represent transitions as lists of ints
    
    def evaluate(trans_a, trans_b):
        """Return reset threshold, or -1 if not synchronizing."""
        return reset_threshold(trans_a, trans_b)
    
    def random_automaton():
        """Generate a random complete binary deterministic automaton."""
        ta = [random.randint(0, N - 1) for _ in range(N)]
        tb = [random.randint(0, N - 1) for _ in range(N)]
        return ta, tb
    
    def mutate(trans_a, trans_b):
        """
        Mutate the automaton by changing a few random transitions.
        Returns new trans_a, trans_b.
        """
        ta = list(trans_a)
        tb = list(trans_b)
        
        # Change 1-3 random transitions
        num_changes = random.randint(1, 3)
        for _ in range(num_changes):
            if random.random() < 0.5:
                idx = random.randint(0, N - 1)
                ta[idx] = random.randint(0, N - 1)
            else:
                idx = random.randint(0, N - 1)
                tb[idx] = random.randint(0, N - 1)
        
        return ta, tb
    
    def local_search(initial_a, initial_b, max_iter=2000):
        """
        Perform local search starting from initial automaton.
        Returns the best (trans_a, trans_b, score) found.
        """
        current_a = list(initial_a)
        current_b = list(initial_b)
        current_score = evaluate(current_a, current_b)
        
        best_a = list(current_a)
        best_b = list(current_b)
        best_local_score = current_score
        
        if current_score < 0:
            # Try to find a synchronizing one by random mutations
            for _ in range(max_iter):
                new_a, new_b = mutate(current_a, current_b)
                new_score = evaluate(new_a, new_b)
                if new_score > 0:
                    current_a, current_b = new_a, new_b
                    current_score = new_score
                    break
            if current_score < 0:
                return best_a, best_b, best_local_score
        
        # Hill climbing with occasional random restarts within local search
        for _ in range(max_iter):
            new_a, new_b = mutate(current_a, current_b)
            new_score = evaluate(new_a, new_b)
            
            if new_score > 0:
                # Accept if it's synchronizing and better, or with some probability if worse
                if new_score >= current_score:
                    current_a, current_b = new_a, new_b
                    current_score = new_score
                    if current_score > best_local_score:
                        best_local_score = current_score
                        best_a = list(current_a)
                        best_b = list(current_b)
                elif random.random() < 0.1:
                    # Occasional acceptance of worse solutions to escape local optima
                    current_a, current_b = new_a, new_b
                    current_score = new_score
            # If new_score is -1 (not synchronizing), reject
            # Keep current
        
        return best_a, best_b, best_local_score
    
    # Main loop: random restarts
    num_restarts = 0
    while time.time() - start_time < timeout:
        # Generate a random starting point
        init_a, init_b = random_automaton()
        
        # Run local search
        best_a, best_b, score = local_search(init_a, init_b, max_iter=1000)
        
        if score > best_score:
            best_score = score
            best_trans_a = best_a
            best_trans_b = best_b
        
        num_restarts += 1
    
    # If no synchronizing automaton found (unlikely), fallback to a known one
    if best_trans_a is None or best_score < 0:
        # Simple cycle + reset
        best_trans_a = [(i + 1) % N for i in range(N)]
        best_trans_b = [0] * N
        best_score = reset_threshold(best_trans_a, best_trans_b)
    
    return best_trans_a, best_trans_b


# For testing
if __name__ == "__main__":
    ta, tb = construct()
    score = reset_threshold(ta, tb)
    print(f"Reset threshold: {score}")
    print(f"Trans_a: {ta}")
    print(f"Trans_b: {tb}")
