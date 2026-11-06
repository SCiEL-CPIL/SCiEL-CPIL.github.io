"""
Černý problem: 12-state binary automata with long reset thresholds.
Strategy: time-bounded stochastic hill-climbing starting from a known
good structure (pinwheel-like), plus local search to maximize the
shortest reset word length.
"""
import time
import random
from collections import deque

N = 12

def _compute_threshold(automaton, max_depth=500):
    """BFS on subset automaton to find shortest reset word length.
    Returns (threshold, word) or (-1, None) if not synchronizing within max_depth."""
    # states are frozensets of states
    start = frozenset(range(N))
    target = None
    # We want to reach a singleton
    if len(start) == 1:
        return (0, "")
    
    # BFS
    visited = {start}
    queue = deque([(start, "")])
    
    # Precompute transitions for speed
    # automaton: list of two lists, each of length N
    a0, a1 = automaton[0], automaton[1]
    
    while queue:
        subset, word = queue.popleft()
        for letter in (0, 1):
            if letter == 0:
                new_subset = frozenset(a0[s] for s in subset)
            else:
                new_subset = frozenset(a1[s] for s in subset)
            
            if len(new_subset) == 1:
                return (len(word) + 1, word + str(letter))
            
            if new_subset not in visited:
                visited.add(new_subset)
                queue.append((new_subset, word + str(letter)))
    
    return (-1, None)

def _is_synchronizing(automaton):
    """Quick check: BFS on subset automaton, just check if singleton reachable."""
    a0, a1 = automaton[0], automaton[1]
    start = frozenset(range(N))
    if len(start) == 1:
        return True
    visited = {start}
    queue = deque([start])
    while queue:
        subset = queue.popleft()
        for letter in (0, 1):
            if letter == 0:
                new_subset = frozenset(a0[s] for s in subset)
            else:
                new_subset = frozenset(a1[s] for s in subset)
            if len(new_subset) == 1:
                return True
            if new_subset not in visited:
                visited.add(new_subset)
                queue.append(new_subset)
    return False

def _pinwheel_automaton():
    """Generate a pinwheel-like automaton with known good threshold."""
    # Pinwheel: state i on letter 0 goes to (i+1)%N, on letter 1 goes to 0
    # This has threshold N-1 = 11
    a0 = [(i + 1) % N for i in range(N)]
    a1 = [0] * N
    return [a0, a1]

def _random_automaton():
    a0 = [random.randint(0, N-1) for _ in range(N)]
    a1 = [random.randint(0, N-1) for _ in range(N)]
    return [a0, a1]

def _mutate(automaton):
    """Randomly change one transition."""
    new_a0 = automaton[0][:]
    new_a1 = automaton[1][:]
    letter = random.randint(0, 1)
    state = random.randint(0, N-1)
    if letter == 0:
        new_a0[state] = random.randint(0, N-1)
    else:
        new_a1[state] = random.randint(0, N-1)
    return [new_a0, new_a1]

def construct():
    deadline = time.time() + 80  # 80s safety margin (limit is 90s)
    
    # Start with pinwheel (threshold 11)
    best_automaton = _pinwheel_automaton()
    best_threshold = _compute_threshold(best_automaton)[0]
    
    if best_threshold < 0:
        best_threshold = 0
    
    current = best_automaton[:]
    current_a0 = current[0][:]
    current_a1 = current[1][:]
    
    iteration = 0
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        iteration += 1
        
        # Mutate
        mutated = _mutate(current)
        
        # Check if synchronizing
        if not _is_synchronizing(mutated):
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Reset to best
                current = best_automaton[:]
                no_improve_count = 0
            continue
        
        # Compute threshold
        threshold, _ = _compute_threshold(mutated)
        
        if threshold > best_threshold:
            best_threshold = threshold
            best_automaton = mutated[:]
            current = mutated[:]
            no_improve_count = 0
        else:
            # Accept with some probability (simulated annealing-like)
            if threshold >= best_threshold - 2 and random.random() < 0.3:
                current = mutated[:]
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Reset to best and try different mutation strategy
            current = best_automaton[:]
            no_improve_count = 0
            # Occasionally try a random restart
            if random.random() < 0.1:
                for _ in range(100):
                    rand_auto = _random_automaton()
                    if _is_synchronizing(rand_auto):
                        t, _ = _compute_threshold(rand_auto)
                        if t > best_threshold:
                            best_threshold = t
                            best_automaton = rand_auto[:]
                            current = rand_auto[:]
                            break
    
    return best_automaton
