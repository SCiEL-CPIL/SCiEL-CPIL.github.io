"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    def greedy_run(seed):
        rng = random.Random(seed)
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        
        A = []
        B = []
        sum_A = set()
        sum_B = set()
        diff_A = set()
        diff_B = set()
        
        for x in candidates:
            # Try adding to A
            # Check Sidon property for A
            # New sums: x+a for a in A, and 2x
            # New diffs: x-a for a in A
            valid_A = True
            new_sums_A = [2*x] + [x+a for a in A]
            if any(s in sum_A for s in new_sums_A):
                valid_A = False
            
            if valid_A:
                new_diffs_A = [x-a for a in A]
                if any(d in diff_B for d in new_diffs_A):
                    valid_A = False
            
            # Try adding to B
            valid_B = True
            new_sums_B = [2*x] + [x+b for b in B]
            if any(s in sum_B for s in new_sums_B):
                valid_B = False
            
            if valid_B:
                new_diffs_B = [x-b for b in B]
                if any(d in diff_A for d in new_diffs_B):
                    valid_B = False
            
            # Choose the side that allows addition. 
            # Strategy: If both valid, pick one (e.g., A). 
            # A better strategy might be to pick the one that creates less "conflict" or is more balanced, 
            # but simple greedy often works. Let's pick A if valid, else B.
            if valid_A:
                A.append(x)
                sum_A.update(new_sums_A)
                diff_A.update(new_diffs_A)
            elif valid_B:
                B.append(x)
                sum_B.update(new_sums_B)
                diff_B.update(new_diffs_B)
                
        return A, B

    best_A = []
    best_B = []
    best_size = 0
    seed = 0
    
    while time.time() < deadline:
        A, B = greedy_run(seed)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
            # Early stopping if we hit a theoretical max? 
            # Max Sidon set size in [1,256] is approx sqrt(256) = 16? 
            # Actually Sidon sets are sparse. Max size is around 16-17.
            # Two sets with disjoint diffs might allow larger total size.
            # If we reach e.g. 30, it's very good.
            # Let's not stop early unless we are sure, but 30+ is likely near optimal for this constraint.
            if size >= 32: # Heuristic threshold
                break
        seed += 1
        
    return [best_A, best_B]
# EVOLVE-BLOCK-END
