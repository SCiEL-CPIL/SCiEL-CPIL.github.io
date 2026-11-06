import time
import random
from collections import deque

N = 12

def _compute_threshold(trans):
    """
    Compute the exact shortest reset word length using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    """
    full_set = frozenset(range(N))
    
    # visited stores the distance from the full set
    visited = {full_set: 0}
    queue = deque([full_set])
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        # If current is a singleton, we found the threshold
        if len(current) == 1:
            return dist
            
        # Expand
        for letter in range(2):
            # Optimization: use list comprehension for speed if possible, 
            # but frozenset comprehension is usually fine for N=12
            next_set = frozenset(trans[letter][s] for s in current)
            if next_set not in visited:
                visited[next_set] = dist + 1
                queue.append(next_set)
                
    return -1

def _seed_population():
    """
    Generate a diverse initial population of candidates.
    Includes:
    1. The classic Černý automaton (cycle + shift).
    2. Random permutations for letter 0 and random functions for letter 1.
    3. 'Ping-pong' structures where letter 0 is a cycle and letter 1 has a specific bottleneck.
    """
    population = []
    
    # 1. Classic Černý
    # a: i -> i+1 mod N
    # b: i -> i-1 mod N if i>0 else 0
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
    population.append([a, b])
    
    # 2. Random restarts
    rng = random.Random(123)
    for _ in range(50):
        # Letter 0: Random permutation (often a cycle helps)
        a = list(range(N))
        rng.shuffle(a)
        
        # Letter 1: Random function
        b = [rng.randint(0, N-1) for _ in range(N)]
        
        population.append([a, b])
        
    # 3. Structured 'Sliding Window' variants
    # Letter 0 is a cycle. Letter 1 maps most states to i-1 but has a 'gap'
    for _ in range(50):
        a = list(range(N))
        rng.shuffle(a)
        
        b = [0] * N
        # Base shift
        shift = rng.randint(1, N-1)
        for i in range(N):
            b[i] = (i - shift) % N
            
        # Introduce 1-2 defects to allow sync
        num_defects = rng.randint(1, 2)
        for _ in range(num_defects):
            idx = rng.randint(0, N-1)
            # Map to a random state, often creating a bottleneck
            b[idx] = rng.randint(0, N-1)
            
        population.append([a, b])
        
    return population

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    population = _seed_population()
    
    # Evaluate initial population
    best_score = -1
    best_trans = None
    
    for trans in population:
        if time.time() > deadline:
            break
        score = _compute_threshold(trans)
        if score > best_score:
            best_score = score
            best_trans = trans[:]
            
    # If we didn't find anything synchronizing, fallback to a known one
    if best_trans is None:
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        for i in range(1, N):
            b[i] = i - 1
        return [a, b]
        
    # Evolutionary Search
    # We maintain a list of candidates. We pick the best one, mutate it, and evaluate.
    # To avoid getting stuck, we also occasionally pick a random candidate from the population.
    
    # Store (score, trans)
    candidates = [(best_score, best_trans)]
    # Add some other good ones from initial pop if they were good
    for trans in population:
        score = _compute_threshold(trans)
        if score > -1 and score < best_score:
             candidates.append((score, trans))
             
    # Sort by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    rng = random.Random(456)
    
    while time.time() < deadline:
        # Select parent
        # With high probability, select the best. With low probability, select a random one.
        if rng.random() < 0.8:
            parent_score, parent_trans = candidates[0]
        else:
            parent_score, parent_trans = rng.choice(candidates)
            
        # Mutate
        # Pick a random letter (0 or 1) and a random state (0..N-1)
        # Change the destination of that state in that letter
        letter = rng.randint(0, 1)
        state = rng.randint(0, N - 1)
        old_dest = parent_trans[letter][state]
        
        # Generate new destination
        # Bias towards nearby states or specific hubs? 
        # Let's try uniform random first, but maybe bias towards the current best's neighbors?
        # Actually, just uniform random is a good baseline for mutation.
        new_dest = rng.randint(0, N - 1)
        
        if new_dest == old_dest:
            continue
            
        # Create child
        child_trans = [parent_trans[0][:], parent_trans[1][:]]
        child_trans[letter][state] = new_dest
        
        # Evaluate
        child_score = _compute_threshold(child_trans)
        
        if child_score > -1:
            # If it's better than the current best, update
            if child_score > best_score:
                best_score = child_score
                best_trans = child_trans
                candidates.insert(0, (child_score, child_trans))
                # Keep population size manageable
                if len(candidates) > 50:
                    candidates.pop()
            else:
                # Add to population if it's not too bad, to maintain diversity
                if len(candidates) < 50:
                    candidates.append((child_score, child_trans))
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    candidates.pop() # Keep top 50
        
    return best_trans

# EVOLVE-BLOCK-END
