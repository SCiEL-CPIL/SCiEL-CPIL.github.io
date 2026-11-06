import time
import random

N = 9
NUM_VERTICES = 1 << N

# Precompute neighbors for all vertices
neighbors = [[] for _ in range(NUM_VERTICES)]
for v in range(NUM_VERTICES):
    for b in range(N):
        neighbors[v].append(v ^ (1 << b))

def construct():
    deadline = time.time() + 110.0  # Safety margin
    
    def greedy_search(start):
        path = [start]
        used = {start}
        # blocked_by_prev contains neighbors of all vertices in path[:-1]
        # Initially path[:-1] is empty, so blocked_by_prev is empty
        blocked_by_prev = set()
        
        while time.time() < deadline:
            cur = path[-1]
            
            # Find valid candidates
            candidates = []
            for v in neighbors[cur]:
                if v in used:
                    continue
                if v in blocked_by_prev:
                    continue
                candidates.append(v)
            
            if not candidates:
                break
            
            # Warnsdorff Heuristic:
            # Score = number of valid moves from candidate v
            best_candidate = None
            min_score = float('inf')
            
            for v in candidates:
                score = 0
                for u in neighbors[v]:
                    if u in used:
                        continue
                    if u in blocked_by_prev:
                        continue
                    score += 1
                
                # Random tie-breaking
                if score < min_score:
                    min_score = score
                    best_candidate = v
                elif score == min_score:
                    if random.random() < 0.5:
                        best_candidate = v
            
            if best_candidate is None:
                break
            
            # Update state
            path.append(best_candidate)
            used.add(best_candidate)
            
            # Update blocked_by_prev:
            # cur is now in path[:-1] for the next iteration
            for u in neighbors[cur]:
                blocked_by_prev.add(u)
                
        return path

    best_path = []
    best_len = 0
    
    # Run multiple iterations with random starts
    # We can also try to bias the random starts or use a specific strategy
    num_starts = 5000 # Increase restarts since each run is fast
    
    for i in range(num_starts):
        if time.time() >= deadline:
            break
        
        start = random.randint(0, NUM_VERTICES - 1)
        path = greedy_search(start)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
            
    # Fallback
    if not best_path:
        best_path = greedy_search(0)
        
    return best_path
