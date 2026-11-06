import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256
    
    # Precompute all conflicting triples (a, b, c) where a < b <= c and a | (b+c)
    # Each triple is a "clause" that must NOT have all three variables true
    all_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            k_min = (2 * b + a - 1) // a
            k_max = (n + b) // a
            for k in range(k_min, k_max + 1):
                c = k * a - b
                if b <= c <= n:
                    all_triples.append((a, b, c))
    
    # For fast lookup: for each variable x, list of triples containing x
    triples_by_var = {x: [] for x in range(1, n + 1)}
    for i, (a, b, c) in enumerate(all_triples):
        triples_by_var[a].append(i)
        triples_by_var[b].append(i)
        triples_by_var[c].append(i)
    
    num_triples = len(all_triples)
    
    # WalkSAT-style local search
    # State: boolean array of size n+1 (index 1..n)
    # Objective: maximize number of True variables while keeping all clauses satisfied
    # A clause (a,b,c) is violated if all three are True
    
    def count_violations(state):
        """Count number of violated triples."""
        count = 0
        for a, b, c in all_triples:
            if state[a] and state[b] and state[c]:
                count += 1
        return count
    
    def get_violated(state):
        """Return list of violated triples."""
        violated = []
        for i, (a, b, c) in enumerate(all_triples):
            if state[a] and state[b] and state[c]:
                violated.append(i)
        return violated
    
    def walk_sat_search(initial_state, time_limit, max_steps=50000, noise=0.3):
        """
        WalkSAT local search.
        state: list of booleans (index 1..n)
        Returns best state found.
        """
        current = list(initial_state)
        best = list(initial_state)
        best_size = sum(current[1:])
        
        # Track clause weights for dynamic weighting
        clause_weights = [1.0] * num_triples
        
        steps_since_improve = 0
        max_no_improve = 5000
        
        for step in range(max_steps):
            if time.time() > time_limit:
                break
            
            violated = get_violated(current)
            
            if not violated:
                # No violations - try to add more True variables
                improved = False
                for x in range(1, n + 1):
                    if not current[x]:
                        current[x] = True
                        if not get_violated(current):
                            improved = True
                        else:
                            current[x] = False
                if improved:
                    new_size = sum(current[1:])
                    if new_size > best_size:
                        best_size = new_size
                        best = list(current)
                    steps_since_improve = 0
                else:
                    steps_since_improve += 1
            else:
                # Pick a violated clause
                vi = random.choice(violated)
                a, b, c = all_triples[vi]
                
                # Weighted random choice among the three variables in this clause
                # Prefer flipping variables that appear in fewer other violated clauses
                weights = []
                for var in [a, b, c]:
                    # Count how many other violated clauses contain this var
                    other_count = 0
                    for vj in violated:
                        if vj != vi:
                            va, vb, vc = all_triples[vj]
                            if var in (va, vb, vc):
                                other_count += 1
                    weights.append(clause_weights[vj] * (1 + other_count))
                
                # Actually, let's use a simpler weighted choice
                # Choose variable to flip with probability proportional to weight
                total_w = sum(weights)
                r = random.random() * total_w
                cumulative = 0
                chosen_var = a
                for var, w in zip([a, b, c], weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen_var = var
                        break
                
                # With probability noise, flip a random variable from the violated clause
                # Otherwise, flip the chosen variable
                if random.random() < noise:
                    flip_var = random.choice([a, b, c])
                else:
                    flip_var = chosen_var
                
                # Flip the variable
                current[flip_var] = not current[flip_var]
                
                # Update clause weights
                for vj in violated:
                    clause_weights[vj] *= 1.05
                # Decay all weights slightly
                for i in range(num_triples):
                    clause_weights[i] *= 0.995
                
                # Check if we improved
                new_violated = get_violated(current)
                if len(new_violated) < len(violated):
                    new_size = sum(current[1:])
                    if new_size > best_size and not new_violated:
                        best_size = new_size
                        best = list(current)
                    steps_since_improve = 0
                elif len(new_violated) == 0:
                    new_size = sum(current[1:])
                    if new_size > best_size:
                        best_size = new_size
                        best = list(current)
                    steps_since_improve = 0
                else:
                    steps_since_improve += 1
            
            if steps_since_improve > max_no_improve:
                # Restart from best with some randomization
                current = list(best)
                for _ in range(10):
                    x = random.randint(1, n)
                    current[x] = not current[x]
                steps_since_improve = 0
                max_no_improve = max(1000, max_no_improve // 2)
        
        return best
    
    # Phase 1: Find a good initial solution using greedy
    # Start with empty set, greedily add elements
    greedy_state = [False] * (n + 1)
    candidates = list(range(1, n + 1))
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline - 60:
            break
        greedy_state[x] = True
        if get_violated(greedy_state):
            greedy_state[x] = False
    
    greedy_size = sum(greedy_state[1:])
    
    # Phase 2: Run WalkSAT from multiple starting points
    best_state = greedy_state[:]
    best_size = greedy_size
    
    # Starting points: greedy solution, and some random perturbations
    start_points = [greedy_state[:]]
    
    # Add some random starting points
    for _ in range(5):
        if time.time() > deadline - 55:
            break
        sp = [False] * (n + 1)
        for x in range(1, n + 1):
            if random.random() < 0.5:
                sp[x] = True
        # Remove violations
        changed = True
        while changed and time.time() < deadline - 55:
            changed = False
            violated = get_violated(sp)
            if violated:
                vi = random.choice(violated)
                a, b, c = all_triples[vi]
                flip_var = random.choice([a, b, c])
                sp[flip_var] = False
                changed = True
        start_points.append(sp)
    
    # Run WalkSAT from each starting point
    for sp in start_points:
        if time.time() > deadline - 15:
            break
        
        time_per_run = (deadline - 15 - time.time()) / max(1, len(start_points))
        time_limit = time.time() + time_per_run
        
        result = walk_sat_search(sp, time_limit, max_steps=30000, noise=0.3)
        result_size = sum(result[1:])
        
        if result_size > best_size:
            best_size = result_size
            best_state = result[:]
    
    # Phase 3: Final local search to try to add more elements
    current = best_state[:]
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for x in range(1, n + 1):
            if time.time() > deadline - 3:
                break
            if not current[x]:
                current[x] = True
                if not get_violated(current):
                    improved = True
                else:
                    current[x] = False
    
    final_size = sum(current[1:])
    if final_size > best_size:
        best_size = final_size
        best_state = current[:]
    
    # Return the set
    result_set = [x for x in range(1, n + 1) if best_state[x]]
    
    # Ensure non-empty
    if not result_set:
        result_set = [n]
    
    return result_set

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
