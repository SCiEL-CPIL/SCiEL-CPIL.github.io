import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    n = 256
    best_seq = None
    best_score = float('inf')
    
    def eval_seq(seq):
        # Returns max abs AP sum
        max_disc = 0
        # Step s
        for s in range(1, n // 2 + 1):
            for a in range(s):
                # Chain a, a+s, ...
                max_end = 0
                max_far = 0
                min_end = 0
                min_far = 0
                first = True
                idx = a
                while idx < n:
                    val = seq[idx]
                    if first:
                        max_end = val
                        max_far = val
                        min_end = val
                        min_far = val
                        first = False
                    else:
                        # Max
                        if max_end + val > val:
                            max_end += val
                        else:
                            max_end = val
                        if max_end > max_far:
                            max_far = max_end
                            
                        # Min
                        if min_end + val < val:
                            min_end += val
                        else:
                            min_end = val
                        if min_end < min_far:
                            min_far = min_end
                    idx += s
                
                if max_far > max_disc:
                    max_disc = max_far
                if -min_far > max_disc:
                    max_disc = -min_far
        return max_disc

    # Initial random sequence
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Generate random sequence
        seq = [rng.choice((-1, 1)) for _ in range(n)]
        score = eval_seq(seq)
        
        if score < best_score:
            best_score = score
            best_seq = seq[:]
            
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle indices to avoid bias? Or just iterate 0..n-1
            indices = list(range(n))
            rng.shuffle(indices)
            for i in indices:
                if time.time() > deadline:
                    break
                # Flip i
                old_val = seq[i]
                seq[i] = -old_val
                new_score = eval_seq(seq)
                
                if new_score < score:
                    score = new_score
                    if score < best_score:
                        best_score = score
                        best_seq = seq[:]
                    improved = True
                else:
                    # Revert
                    seq[i] = old_val
                    
    return best_seq
