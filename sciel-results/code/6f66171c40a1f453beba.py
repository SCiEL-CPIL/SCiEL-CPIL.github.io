import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin

    N = 180
    
    # 1. Precompute forbidden triples
    # a = bc / (b+c) => a(b+c) = bc => ab + ac = bc => b(a-c) = -ac => b = ac / (c-a)
    # Or just iterate b, c and check if a is integer.
    forbidden = set()
    for b in range(1, N+1):
        for c in range(b+1, N+1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    # Ensure distinct
                    if len({a, b, c}) == 3:
                        forbidden.add(frozenset([a, b, c]))
    
    # Precompute conflict pairs for each element
    # conflicts[x] = list of (y, z) such that {x, y, z} is forbidden
    conflicts = {i: [] for i in range(1, N+1)}
    for triple in forbidden:
        elems = list(triple)
        for i in range(3):
            x = elems[i]
            y = elems[(i+1) % 3]
            z = elems[(i+2) % 3]
            conflicts[x].append((y, z))
            
    # For fast checking: for each pair (y, z), list of x that conflict
    pair_to_third = {}
    for triple in forbidden:
        elems = list(triple)
        for i in range(3):
            x = elems[i]
            y = elems[(i+1) % 3]
            z = elems[(i+2) % 3]
            key = (min(y, z), max(y, z))
            if key not in pair_to_third:
                pair_to_third[key] = []
            pair_to_third[key].append(x)

    def is_valid(S):
        S_set = set(S)
        for x in S:
            for y, z in conflicts[x]:
                if y in S_set and z in S_set:
                    return False
        return True

    def can_add(elem, S):
        S_set = set(S)
        for y, z in conflicts[elem]:
            if y in S_set and z in S_set:
                return False
        return True

    # 2. Structural Initialization
    # Start with large numbers
    S = set(range(91, N+1))
    
    # Verify S is valid (it should be)
    if not is_valid(S):
        # Fallback to greedy if structural fails (unlikely)
        S = set()
        for i in range(N, 0, -1):
            if can_add(i, S):
                S.add(i)

    # Add small numbers that don't conflict with S
    for x in range(1, 91):
        if can_add(x, S):
            S.add(x)
            
    # 3. Greedy Polish
    # Try to add any remaining elements
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 20:
        improved = False
        for x in range(1, N+1):
            if x not in S:
                if can_add(x, S):
                    S.add(x)
                    improved = True

    best_S = S.copy()
    best_size = len(S)
    
    # 4. Local Search
    # Moves:
    # 1. Add x (if valid)
    # 2. Swap y in S with x not in S (if valid)
    #    - Accept if size increases (impossible for swap)
    #    - Accept if swap enables immediate addition of z (lookahead 1)
    # 3. Remove y in S (if it enables addition of 2+ others? Too complex, stick to swap)
    
    # To improve swap acceptance, we can calculate a "potential" score
    # Potential = number of elements that can be added immediately
    # We accept a swap if Potential(new_S) > Potential(old_S) or (== and random)
    
    def get_potential(S):
        pot = 0
        S_set = set(S)
        for x in range(1, N+1):
            if x not in S_set:
                if can_add(x, S):
                    pot += 1
        return pot

    current_potential = get_potential(S)
    
    max_steps = 10000
    step = 0
    temperature = 1.0
    cooling = 0.99
    
    while step < max_steps and (time.time() - start_time) < DEADLINE - 10:
        step += 1
        if step % 100 == 0:
            temperature *= cooling
            
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try swap
            if not S: continue
            y = random.choice(list(S))
            x = random.choice([i for i in range(1, N+1) if i not in S])
            if x == y: continue
            
            # Check if swap is valid
            S_new = (S - {y}) | {x}
            if is_valid(S_new):
                new_potential = get_potential(S_new)
                # Accept if better potential, or same potential (to explore)
                if new_potential > current_potential:
                    S = S_new
                    current_potential = new_potential
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S.copy()
                elif new_potential == current_potential:
                    # Accept with some probability to escape local minima
                    if random.random() < 0.5:
                        S = S_new
                        current_potential = new_potential
                else:
                    # Worse potential, accept with prob exp(-delta/T)
                    delta = current_potential - new_potential
                    if random.random() < pow(2.718, -delta / max(temperature, 0.1)):
                        S = S_new
                        current_potential = new_potential
        else:
            # Try single addition
            x = random.choice([i for i in range(1, N+1) if i not in S])
            if can_add(x, S):
                S.add(x)
                current_potential = get_potential(S)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S.copy()

    # Final polish
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 5:
        improved = False
        for x in range(1, N+1):
            if x not in S:
                if can_add(x, S):
                    S.add(x)
                    improved = True
    
    if len(S) > best_size:
        best_S = S.copy()

    result = sorted(best_S)
    if not result:
        result = [91]
        
    return result
