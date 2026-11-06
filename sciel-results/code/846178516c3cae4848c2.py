import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    N = 180
    # Precompute forbidden triples
    # 1/a = 1/b + 1/c => a(b+c) = bc => bc - ab - ac = 0 => (a-b)(a-c) = -a^2
    # We iterate a, b, c
    triples = []
    triples_by_elem = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        # b must be > a
        for b in range(a + 1, N + 1):
            # c = ab / (b - a)
            # b - a must divide ab
            # Since gcd(b-a, a) = gcd(b, a), let g = gcd(a, b)
            # This is getting complicated, just use integer check
            denom = b - a
            num = a * b
            if num % denom == 0:
                c = num // denom
                # Check distinctness and range
                if c > a and c != b and c <= N:
                    # Ensure a, b, c are distinct
                    # b > a, c > a, c != b covers it
                    t = tuple(sorted((a, b, c)))
                    triples.append(t)
                    idx = len(triples) - 1
                    for x in t:
                        triples_by_elem[x].append(idx)
    
    num_triples = len(triples)
    counts = [0] * num_triples
    
    # Initial set: 91 to 180
    S = [False] * (N + 1)
    for i in range(91, N + 1):
        S[i] = True
        
    # Initialize counts
    for i in range(91, N + 1):
        for t_idx in triples_by_elem[i]:
            counts[t_idx] += 1
            
    current_size = 90
    best_size = 90
    best_S = S.copy()
    
    # Function to check if x can be added
    def can_add(x):
        for t_idx in triples_by_elem[x]:
            if counts[t_idx] == 2:
                return False
        return True

    def add(x):
        S[x] = True
        for t_idx in triples_by_elem[x]:
            counts[t_idx] += 1

    def remove(x):
        S[x] = False
        for t_idx in triples_by_elem[x]:
            counts[t_idx] -= 1

    # Function to greedily add as many elements as possible
    def greedy_fill():
        nonlocal current_size
        # Randomize order to break symmetry
        unselected = [i for i in range(1, N + 1) if not S[i]]
        random.shuffle(unselected)
        
        added_any = False
        for x in unselected:
            if can_add(x):
                add(x)
                current_size += 1
                added_any = True
        return added_any

    # Initial greedy fill
    greedy_fill()
    if current_size > best_size:
        best_size = current_size
        best_S = S.copy()
        
    # Local Search Loop
    while time.time() < deadline:
        # Try to add a random element (quick check)
        # This is covered by greedy_fill if we call it, but greedy_fill is O(N*deg)
        # If we are at a local optimum, greedy_fill adds nothing.
        
        # Perform a kick
        # Choose a random element to remove
        # To be more effective, we can bias towards elements that are in many triples
        # But random is a good baseline.
        
        # Get list of selected elements
        selected_elems = [i for i in range(1, N + 1) if S[i]]
        if not selected_elems:
            break
            
        y = random.choice(selected_elems)
        
        # Save state to revert if it doesn't help?
        # Or just update best if it does.
        # Since we want to maximize, we can just keep the current state
        # and update best if we improve.
        # If we degrade, we might want to revert to avoid getting stuck in a hole.
        # However, simulated annealing style degradation can help escape.
        # Let's try: if new_size < best_size, revert to best_S?
        # No, that's "restart from best".
        # Let's try: if new_size < current_size_before_kick, revert.
        # This is "Hill Climbing with Plateau Walking" (only accept equal or better).
        # But we might need to accept worse to escape.
        
        # Let's use a simple strategy:
        # 1. Remove y
        # 2. Greedy Fill
        # 3. If current_size > best_size, update best.
        # 4. If current_size < best_size - 5, revert to best_S (restart from best).
        #    This prevents drifting too far from the known best.
        
        remove(y)
        current_size -= 1
        
        greedy_fill()
        
        if current_size > best_size:
            best_size = current_size
            best_S = S.copy()
            
        # If we are significantly worse than the best, reset to best
        # This is a form of "restart"
        if current_size < best_size - 10:
            S = best_S.copy()
            # Reinitialize counts
            for i in range(1, N + 1):
                counts[i] = 0 # Wrong index, counts is per triple
            counts = [0] * num_triples
            for i in range(1, N + 1):
                if S[i]:
                    for t_idx in triples_by_elem[i]:
                        counts[t_idx] += 1
            current_size = best_size
            
    # Return the best set found
    return [i for i in range(1, N + 1) if best_S[i]]
# EVOLVE-BLOCK-END
