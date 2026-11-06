import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    N = 1000
    best_size = 0
    best_set = []
    
    # Helper to check if x can be added to S with current sum_counts
    # Returns True if valid, False otherwise
    def can_add(x, S, sum_counts):
        # Check 2*x
        if sum_counts.get(2 * x, 0) >= 2:
            return False
        # Check x + y for y in S
        for y in S:
            s = x + y
            if sum_counts.get(s, 0) >= 2:
                return False
        return True

    # Helper to add x to S and update sum_counts
    def add(x, S, sum_counts):
        S.append(x)
        sum_counts[2 * x] = sum_counts.get(2 * x, 0) + 1
        for y in S[:-1]: # S[:-1] is S before appending x
            s = x + y
            sum_counts[s] = sum_counts.get(s, 0) + 1

    # Helper to remove x from S and update sum_counts
    def remove(x, S, sum_counts):
        S.remove(x)
        sum_counts[2 * x] -= 1
        if sum_counts[2 * x] == 0:
            del sum_counts[2 * x]
        for y in S:
            s = x + y
            sum_counts[s] -= 1
            if sum_counts[s] == 0:
                del sum_counts[s]

    # Run multiple random greedy attempts
    while time.time() < deadline:
        S = []
        sum_counts = {}
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S, sum_counts):
                add(x, S, sum_counts)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        # Optional: Try a few local search moves on the current best?
        # Or just keep restarting.
        
    return best_set
