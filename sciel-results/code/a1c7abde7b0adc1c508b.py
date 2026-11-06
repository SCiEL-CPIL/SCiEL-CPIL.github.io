import random
import time

N = 9
MAX_TIME = 110  # Leave 10s buffer for overhead

def construct():
    best_path = []
    best_len = 0
    
    # Initial deterministic run (same as seed) to ensure we have a baseline
    # Actually, let's just rely on the random runs, but include the seed logic as one specific priority
    # The seed uses lowest bit first: priority [0, 1, ..., 8]
    
    def run_greedy(priority):
        path = [0]
        used = {0}
        # We need to check against all previous vertices for the induced path constraint
        # "non-consecutive vertices differ in at least two bits"
        # This means for a new vertex v, for all p in path[:-1], (v ^ p).bit_count() != 1
        
        while True:
            cur = path[-1]
            next_v = None
            for b in priority:
                v = cur ^ (1 << b)
                if v in used:
                    continue
                # Check induced path constraint
                valid = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    next_v = v
                    break
            
            if next_v is None:
                return path
            
            path.append(next_v)
            used.add(next_v)
            
    # Run many random trials
    start_time = time.time()
    iterations = 0
    
    # Always try the standard greedy first (lowest bit priority)
    p = run_greedy(list(range(N)))
    if len(p) > best_len:
        best_len = len(p)
        best_path = p
        
    while time.time() - start_time < MAX_TIME:
        # Random priority
        priority = list(range(N))
        random.shuffle(priority)
        
        p = run_greedy(priority)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
        iterations += 1
        
        # Optional: If we find a very long snake, we might want to stop or adjust strategy,
        # but for now, just keep looking for longer ones.
        
    return best_path
