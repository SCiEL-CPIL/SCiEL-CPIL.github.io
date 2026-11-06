import time
import random

def construct():
    N = 18
    D = 6
    # Total number of possible codewords
    num_candidates = 1 << N
    
    # Generate all candidates
    candidates = list(range(num_candidates))
    
    # Helper function to run greedy algorithm on a specific order
    def run_greedy(order, deadline):
        code = []
        bc = int.bit_count
        for w in order:
            # Check time limit with a safety margin
            if time.time() > deadline:
                break
            
            # Check if w is at least distance D from all existing codewords
            valid = True
            for c in code:
                if bc(w ^ c) < D:
                    valid = False
                    break
            
            if valid:
                code.append(w)
                
        return code

    # Start timer
    start_time = time.time()
    # Leave 5 seconds for safety
    deadline = start_time + 55.0
    
    best_code = []
    
    # Strategy 1: Sorted by weight (ascending), then by value
    # This tends to pack low-weight words first, which can be good for some structures
    sorted_candidates = sorted(candidates, key=lambda x: (x.bit_count(), x))
    code1 = run_greedy(sorted_candidates, deadline)
    if len(code1) > len(best_code):
        best_code = code1
        
    # Strategy 2: Random shuffles
    # Run multiple times to explore different local optima
    # We estimate each run takes ~5-10s, so we can do ~5 runs
    runs_completed = 1
    max_runs = 6 # Arbitrary limit, will stop based on time
    
    for i in range(max_runs):
        if time.time() > deadline - 10.0:
            # Not enough time for another full run
            break
            
        random.shuffle(candidates)
        code_r = run_greedy(candidates, deadline)
        
        if len(code_r) > len(best_code):
            best_code = code_r
            
    return best_code
