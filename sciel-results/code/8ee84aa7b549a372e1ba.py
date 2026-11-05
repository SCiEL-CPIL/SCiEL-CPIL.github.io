import random
import time

N = 256

def construct():
    best_set = None
    best_size = N + 1
    
    # Helper to check if x can be added to S
    def can_add(x, S, sums):
        if x in S:
            return False
        if 2 * x in sums:
            return False
        for a in S:
            if x + a in sums:
                return False
        return True

    # Helper to get all addable elements
    def get_addable(S, sums):
        U = []
        for x in range(1, N + 1):
            if can_add(x, S, sums):
                U.append(x)
        return U

    # Helper to compute sums for a set
    def compute_sums(S):
        sums = set()
        for a in S:
            sums.add(2 * a)
            for b in S:
                if a < b:
                    sums.add(a + b)
        return sums

    # Helper to greedily build a maximal set from a starting set S_start
    # forbid_element: if set, try to avoid adding this element
    def build_maximal(S_start, forbid_element=None, randomize=True):
        S = set(S_start)
        sums = compute_sums(S)
        
        while True:
            U = get_addable(S, sums)
            if not U:
                break
            
            # Filter out forbidden element if possible, but if it's the only one, we must take it
            candidates = U
            if forbid_element is not None and forbid_element in U:
                candidates = [x for x in U if x != forbid_element]
                if not candidates:
                    candidates = U # Must take it if no other choice
            
            if randomize:
                # Heuristic: pick randomly from candidates, or weighted by degree?
                # Simple random is good for exploration in local search
                chosen = random.choice(candidates)
            else:
                # Greedy: pick one that blocks most
                # Re-use the degree logic from original code for speed if needed, 
                # but simple random is often sufficient for the "repair" phase 
                # if we do many iterations.
                # Let's stick to random for simplicity and speed in the inner loop,
                # as the "greedy" part was in the initialization.
                chosen = random.choice(candidates)
            
            S.add(chosen)
            sums.add(2 * chosen)
            for a in S:
                if a != chosen:
                    sums.add(chosen + a)
                    
        return S

    # 1. Initialize with a few good greedy sets
    initial_sets = []
    for _ in range(20):
        S_init = set()
        # Use the original greedy logic for initialization to get a good starting point
        # (This part is similar to the original code's inner loop)
        sums = set()
        while True:
            U = []
            for x in range(1, N + 1):
                if x in S_init: continue
                if 2*x in sums: continue
                conflict = False
                for a in S_init:
                    if x + a in sums:
                        conflict = True
                        break
                if not conflict:
                    U.append(x)
            if not U: break
            
            # Greedy pick: max degree in conflict graph
            degrees = {x: 0 for x in U}
            U_set = set(U)
            S_list = list(S_init)
            for x in U:
                neighbors = set()
                for a in S_list:
                    y = 2 * x - a
                    if 1 <= y <= N and y in U_set and y != x:
                        neighbors.add(y)
                for b in S_list:
                    val = x + b
                    if val % 2 == 0:
                        y = val // 2
                        if 1 <= y <= N and y in U_set and y != x:
                            neighbors.add(y)
                for a in S_list:
                    for b in S_list:
                        y = x + b - a
                        if 1 <= y <= N and y in U_set and y != x:
                            neighbors.add(y)
                degrees[x] = len(neighbors)
                
            max_deg = -1
            cands = []
            for x in U:
                if degrees[x] > max_deg:
                    max_deg = degrees[x]
                    cands = [x]
                elif degrees[x] == max_deg:
                    cands.append(x)
            chosen = random.choice(cands)
            
            S_init.add(chosen)
            sums.add(2 * chosen)
            for a in S_init:
                if a != chosen:
                    sums.add(chosen + a)
        initial_sets.append(S_init)
        if len(S_init) < best_size:
            best_size = len(S_init)
            best_set = list(S_init)

    # 2. Local Search: Remove and Repair
    # For each initial set, try to shrink it
    for S_curr in initial_sets:
        current_size = len(S_curr)
        if current_size >= best_size:
            continue
            
        # Simulated Annealing / Random Walk on the space of maximal sets
        # We try to remove elements and repair.
        # Time limit check
        start_time = time.time()
        
        for step in range(500):
            if time.time() - start_time > 15: # Leave time for other sets
                break
                
            if not S_curr:
                break
                
            # Pick a random element to remove
            s_to_remove = random.choice(list(S_curr))
            S_removed = S_curr - {s_to_remove}
            
            # Repair: Build a maximal set from S_removed, avoiding s_to_remove if possible
            # We use build_maximal with forbid_element=s_to_remove
            # To ensure we find *a* maximal set, we might need to allow adding s_to_remove
            # if no other elements can be added to make it maximal.
            # build_maximal handles this: if candidates (excluding forbidden) is empty, it takes forbidden.
            
            S_new = build_maximal(S_removed, forbid_element=s_to_remove, randomize=True)
            
            new_size = len(S_new)
            
            # Acceptance Criteria
            if new_size < current_size:
                # Strict improvement
                S_curr = S_new
                current_size = new_size
                if current_size < best_size:
                    best_size = current_size
                    best_set = list(S_curr)
            elif new_size == current_size:
                # Same size, accept with some probability to explore (annealing)
                # Probability decreases as we get closer to the end?
                # Simple random walk: accept with prob 0.5
                if random.random() < 0.5:
                    S_curr = S_new
            # If new_size > current_size, reject
            
    return best_set
