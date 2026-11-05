"""
Constructs a 12-state binary deterministic automaton that synchronizes,
maximizing the length of its shortest reset word.

Uses bitmask-based BFS for fast threshold computation and a large-scale
random restart local search to directly optimize both transition functions.
"""
import time
import random
from collections import deque

N = 12

# EVOLVE-BLOCK-START
def _reset_threshold(a, b):
    """
    Compute the exact shortest reset word length for a 12-state binary automaton
    defined by transition lists a and b.
    Returns (threshold, word) where word is a list of 'a'/'b' chars, or (None, None) if not synchronizing.
    Uses bitmask representation: each state is a bit in an integer.
    """
    full = (1 << N) - 1
    
    # BFS on subset automaton
    # State: bitmask of reachable states
    # Start: full set
    # Target: any singleton {s} -> (1 << s)
    
    # Precompute transitions for each letter on each subset? No, compute on the fly.
    # For a subset S (bitmask), applying letter 'a' gives subset {a[s] for s in S}
    # We can precompute the image of each state under 'a' and 'b'.
    
    # Actually, let's precompute the transition of each single state, then compose.
    # For a bitmask S, the image under letter L is:
    #   OR over s in S of (1 << L[s])
    
    # Let's precompute for each letter and each state the destination bit.
    # a_bits[i] = 1 << a[i]
    # b_bits[i] = 1 << b[i]
    
    a_bits = [1 << a[i] for i in range(N)]
    b_bits = [1 << b[i] for i in range(N)]
    
    # Function to compute image of subset under letter
    def image(subset, bits):
        result = 0
        s = subset
        while s:
            lsb = s & (-s)
            idx = lsb.bit_length() - 1
            result |= bits[idx]
            s ^= lsb
        return result
    
    # BFS
    visited = set()
    visited.add(full)
    
    # Queue: (subset, word_length, last_letter)
    # We store the path to reconstruct the word
    # Actually, for just the threshold, we don't need the word.
    # But the task says "returns a shortest word" - wait, the construct() function
    # just returns the transition lists. The evaluator computes the threshold.
    # So we just need the threshold value.
    
    # Let's just compute the length.
    queue = deque()
    queue.append((full, 0))
    
    while queue:
        subset, dist = queue.popleft()
        
        # Check if singleton
        if subset & (subset - 1) == 0:
            return dist
        
        # Try 'a'
        new_subset_a = image(subset, a_bits)
        if new_subset_a not in visited:
            visited.add(new_subset_a)
            queue.append((new_subset_a, dist + 1))
        
        # Try 'b'
        new_subset_b = image(subset, b_bits)
        if new_subset_b not in visited:
            visited.add(new_subset_b)
            queue.append((new_subset_b, dist + 1))
    
    return None  # Not synchronizing


def _random_automaton():
    """Generate a random 12-state binary automaton."""
    a = [random.randint(0, N - 1) for _ in range(N)]
    b = [random.randint(0, N - 1) for _ in range(N)]
    return a, b


def _mutate(a, b):
    """Apply a random mutation to the automaton."""
    a = a[:]
    b = b[:]
    # Mutate a few random entries
    num_mutations = random.randint(1, 3)
    for _ in range(num_mutations):
        if random.random() < 0.5:
            idx = random.randint(0, N - 1)
            a[idx] = random.randint(0, N - 1)
        else:
            idx = random.randint(0, N - 1)
            b[idx] = random.randint(0, N - 1)
    return a, b


def construct():
    """
    Construct a 12-state binary deterministic automaton that synchronizes,
    maximizing the shortest reset word length.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin: 5s before 90s limit
    
    best_threshold = 0
    best_a = None
    best_b = None
    
    # Start with a known synchronizing automaton
    # The Černý construction: a is a cycle, b maps 0 to 0 and i to i+1 for i>0
    a0 = [(i + 1) % N for i in range(N)]
    b0 = [0] * N
    for i in range(1, N):
        b0[i] = (i + 1) % N
    
    thr0 = _reset_threshold(a0, b0)
    if thr0 is not None and thr0 > best_threshold:
        best_threshold = thr0
        best_a = a0[:]
        best_b = b0[:]
    
    # Random restart local search
    random.seed(42)
    
    num_restarts = 0
    while time.time() < deadline:
        # Start a new random automaton
        a, b = _random_automaton()
        thr = _reset_threshold(a, b)
        num_restarts += 1
        
        if thr is None:
            # Try to repair or just restart
            continue
        
        if thr > best_threshold:
            best_threshold = thr
            best_a = a[:]
            best_b = b[:]
        
        # Local search: try mutations
        current_a, current_b = a, b
        current_thr = thr
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try a few mutations
            for _ in range(10):
                if time.time() >= deadline:
                    break
                new_a, new_b = _mutate(current_a, current_b)
                new_thr = _reset_threshold(new_a, new_b)
                if new_thr is not None and new_thr > current_thr:
                    current_a, current_b = new_a, new_b
                    current_thr = new_thr
                    improved = True
                    if current_thr > best_threshold:
                        best_threshold = current_thr
                        best_a = current_a[:]
                        best_b = current_b[:]
                    break  # Restart mutation loop after improvement
        
        # Also try local search from the best known automaton
        if time.time() < deadline:
            a, b = best_a[:], best_b[:]
            thr = best_threshold
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for _ in range(10):
                    if time.time() >= deadline:
                        break
                    new_a, new_b = _mutate(a, b)
                    new_thr = _reset_threshold(new_a, new_b)
                    if new_thr is not None and new_thr > thr:
                        a, b = new_a, new_b
                        thr = new_thr
                        improved = True
                        if thr > best_threshold:
                            best_threshold = thr
                            best_a = a[:]
                            best_b = b[:]
                        break
    
    if best_a is None:
        # Fallback to the simple construction
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
