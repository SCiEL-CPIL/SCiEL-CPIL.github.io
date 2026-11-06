import time
import random

N = 9
MAX_V = 1 << N

# Precompute neighbors for all vertices
NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(MAX_V)]

def construct():
    start_time = time.time()
    # Safety margin: return before 120s limit. 115s is safe.
    deadline = start_time + 115.0
    
    best_path = []
    
    # Helper to run a single greedy search with specific strategy
    def run_greedy(strategy='random'):
        path = [0]
        used = {0}
        forbidden = set() # Neighbors of path[0:-1]
        
        while True:
            cur = path[-1]
            cur_neighbors = NEIGHBORS[cur]
            
            valid_candidates = []
            for v in cur_neighbors:
                if v in used:
                    continue
                if v in forbidden:
                    continue
                valid_candidates.append(v)
                
            if not valid_candidates:
                break
            
            # Calculate next_forbidden for heuristic scoring
            # next_forbidden = forbidden U neighbors(cur)
            # We can just iterate and check, or build the set.
            # Building set is O(N).
            next_forbidden = set(forbidden)
            for nb in cur_neighbors:
                next_forbidden.add(nb)
                
            if strategy == 'random':
                v = random.choice(valid_candidates)
            else:
                # Heuristic: Score by future degree
                scored = []
                for v in valid_candidates:
                    v_neighbors = NEIGHBORS[v]
                    count = 0
                    for w in v_neighbors:
                        if w in used:
                            continue
                        if w == v:
                            continue
                        if w in next_forbidden:
                            continue
                        count += 1
                    scored.append((count, v))
                    
                if strategy == 'min':
                    scored.sort(key=lambda x: x[0])
                    # Pick from top 3 to add variance
                    k = min(3, len(scored))
                    v = random.choice(scored[:k])[1]
                elif strategy == 'max':
                    scored.sort(key=lambda x: -x[0])
                    k = min(3, len(scored))
                    v = random.choice(scored[:k])[1]
                else:
                    v = random.choice(valid_candidates)
                    
            path.append(v)
            used.add(v)
            forbidden = next_forbidden
            
        return path

    # Run a mix of strategies
    # Start with deterministic-ish runs to find a good baseline
    # Then randomize
    
    # Initial run with 'min' heuristic (deterministic if k=1, but we use k=3 random)
    # Let's do a few fixed seeds or just loop
    
    while time.time() < deadline:
        # Choose strategy
        r = random.random()
        if r < 0.4:
            strategy = 'min'
        elif r < 0.7:
            strategy = 'max'
        else:
            strategy = 'random'
            
        path = run_greedy(strategy)
        
        if len(path) > len(best_path):
            best_path = path
            
    return best_path
