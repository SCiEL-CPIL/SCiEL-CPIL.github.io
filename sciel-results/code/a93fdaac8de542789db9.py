import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_seq = []
    best_len = 0
    
    # Helper to check if y can be added to prefix sums Q with sum set P
    # Q is list of prefix sums, P is set of all pairwise sums u+v for u<=v in Q
    def can_add(y, Q, P):
        # New sums to check: y+q for q in Q, and y+y
        # If any of these are in P, conflict.
        # Note: We must ensure we don't count the same pair twice if logic overlaps, 
        # but since P contains ALL sums from previous steps, and y is new, 
        # any sum involving y is new.
        # So we just check intersection.
        for q in Q:
            s = y + q
            if s in P:
                return False
        if (y + y) in P:
            return False
        return True

    def update_state(y, Q, P):
        # Add y to Q
        new_Q = Q + [y]
        # Add new sums to P
        # Sums are y+q for q in Q (old) and y+y
        for q in Q:
            P.add(y + q)
        P.add(y + y)
        return new_Q

    def run_greedy(strategy, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        Q = [0]
        P = {0}
        last_a = 0
        seq = []
        
        # We iterate through possible values for next element
        # To allow randomization, we generate candidates and filter
        while True:
            valid_candidates = []
            # Candidates must be > last_a and <= 256
            # To speed up, we can iterate range
            for a in range(last_a + 1, 257):
                if can_add(Q[-1] + a, Q, P):
                    valid_candidates.append(a)
            
            if not valid_candidates:
                break
            
            if strategy == "smallest":
                a = valid_candidates[0]
            elif strategy == "largest":
                a = valid_candidates[-1]
            else: # random
                a = random.choice(valid_candidates)
            
            # Update state
            new_Q_val = Q[-1] + a
            Q = update_state(new_Q_val, Q, P)
            last_a = a
            seq.append(a)
            
            # Time check
            if time.time() > deadline:
                return seq

        return seq

    # 1. Run the baseline greedy (smallest)
    seq = run_greedy("smallest")
    if len(seq) > best_len:
        best_seq = seq
        best_len = len(seq)
        
    # 2. Run randomized searches
    # We can run many iterations
    # Each iteration is fast.
    # Let's run "random" and "largest" strategies.
    
    # Estimate time per run: ~1ms. 85s allows ~80,000 runs.
    # But we should check time inside the loop.
    
    for i in range(100000):
        if time.time() > deadline:
            break
            
        # Alternate strategies or just use random
        # "random" picks a random valid next step.
        # This explores different branches.
        
        # To ensure diversity, we can also vary the "range" of candidates considered
        # But simple random choice from all valid is a good start.
        
        seq = run_greedy("random")
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
            
        # Also try "largest" occasionally?
        # Largest tends to make Q grow fast, limiting length.
        # But maybe it opens up different structures? Unlikely to beat smallest/random.
        
        # Let's stick to random for now.
        
    return best_seq
