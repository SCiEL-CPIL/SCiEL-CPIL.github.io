import numpy as np
import time
import random

N = 32
MAX_VAL = 512
DEADLINE_MARGIN = 5.0  # seconds before the 150s limit

def _score_set_numpy(S):
    """
    Compute the number of distinct differences d > 0 such that there exists a 
    3-term arithmetic progression (a, a+d, a+2d) within the set S.
    S is a numpy array of distinct integers.
    """
    if len(S) < 3:
        return 0
    
    # For each pair (i, j) with i < j, compute d = S[j] - S[i]
    # We want to know if S[i] + 2*d is in S.
    # Note: S[i] + 2*d = S[i] + 2*(S[j] - S[i]) = 2*S[j] - S[i]
    
    # Vectorized approach:
    # Create a 2D array of all differences D[i, j] = S[j] - S[i] for i < j
    # This is tricky to do purely vectorized without huge memory, but N=32 is small.
    # Let's use a set for membership for speed in Python logic, or bitset.
    # Given N=32, a set of integers is very fast.
    
    S_set = set(S)
    diffs = set()
    
    # To make it faster, we can iterate over d directly?
    # Max d is 255.
    # For each d in 1..255:
    #   Check if there exists a such that a in S, a+d in S, a+2d in S.
    #   This is equivalent to: (S & (S - d) & (S - 2d)) is not empty.
    #   Using bitsets (numpy arrays of 0/1 or Python ints) is extremely fast.
    
    # Let's use a Python integer as a bitset for the set S.
    # S contains values in 1..512.
    # Bit i represents if value i is in S.
    # We need to shift the bitset.
    
    mask = 0
    for x in S:
        mask |= (1 << x)
        
    # For each d in 1..255:
    #   Check if mask & (mask >> d) & (mask >> 2*d) != 0
    #   Note: shifting a Python int is very fast.
    
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> 2*d)) != 0:
            count += 1
            
    return count

def _score_mask(mask):
    """
    Score a bitset mask (Python int) representing a set of integers in 1..512.
    """
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> 2*d)) != 0:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Greedy construction
    # Start with an empty set (mask = 0, count = 0)
    current_mask = 0
    current_set = []
    current_score = 0
    
    # Candidates: all integers 1..512
    candidates = list(range(1, MAX_VAL + 1))
    
    # We want to add N=32 elements.
    # At each step, pick the candidate that maximizes the new score.
    # To speed this up, we can pre-filter or just evaluate all.
    # Evaluating 512 candidates * 32 steps = 16384 evaluations.
    # Each evaluation is O(256) bit ops. Very fast.
    
    for step in range(N):
        if time.time() > deadline:
            break
            
        best_candidate = None
        best_new_score = -1
        
        # Optimization: If we have many candidates, we can sample?
        # No, 512 is small enough to check all.
        # But we can skip candidates that are "too far" or similar?
        # Let's just check all.
        
        for c in candidates:
            new_mask = current_mask | (1 << c)
            # Quick check: does adding c create any new APs?
            # We can compute the score incrementally?
            # Score(mask) is not easily incremental for distinct d count.
            # But we can just call _score_mask.
            
            new_score = _score_mask(new_mask)
            
            if new_score > best_new_score:
                best_new_score = new_score
                best_candidate = c
        
        if best_candidate is None:
            # Fallback: pick random
            best_candidate = random.choice(candidates)
            best_new_score = _score_mask(current_mask | (1 << best_candidate))
            
        # Add best candidate
        current_mask |= (1 << best_candidate)
        current_set.append(best_candidate)
        candidates.remove(best_candidate)
        current_score = best_new_score
        
        # Optional: Local improvement after each addition?
        # Maybe after a few steps, do a swap-based local search to escape local optima.
        # But let's first see how good the greedy is.
        
        # To enhance, we can do a "lookahead" or "swap" step every few iterations.
        # Let's try a simple local search after the greedy construction is complete.
        
    # Convert mask to list
    result = [i for i in range(1, MAX_VAL + 1) if (current_mask >> i) & 1]
    
    # Local Search Phase: Try to improve via swaps
    # Time remaining?
    if time.time() < deadline:
        best_mask = current_mask
        best_score = current_score
        best_set = result
        
        # Iterate swaps
        # To speed up, we can try random swaps.
        # Or systematic swaps: for each element in set, try swapping with each element outside.
        # 32 * 480 = 15360 swaps.
        # Each swap requires re-scoring the whole set?
        # Or incremental?
        # Removing x and adding y.
        # The score depends on all triples.
        # It's hard to update incrementally easily without complex bookkeeping.
        # Let's just re-score. 15360 * 256 ops is ~4M ops. Fast in Python if using bitsets.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all possible swaps?
            # Or random subset?
            # Let's try all swaps in one pass.
            
            set_elems = [i for i in range(1, MAX_VAL + 1) if (best_mask >> i) & 1]
            not_in_set = [i for i in range(1, MAX_VAL + 1) if not ((best_mask >> i) & 1)]
            
            # To avoid O(N^2) full scans if slow, we can randomize.
            # But 15k evaluations is fine.
            
            for x in set_elems:
                for y in not_in_set:
                    if time.time() > deadline:
                        break
                        
                    # Swap x and y
                    new_mask = (best_mask & ~(1 << x)) | (1 << y)
                    new_score = _score_mask(new_mask)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_mask = new_mask
                        best_set = [i for i in range(1, MAX_VAL + 1) if (best_mask >> i) & 1]
                        improved = True
                        # Note: set_elems and not_in_set are now stale.
                        # But we are in a while loop, so we will recompute them.
                        # However, breaking out of inner loops is messy.
                        # Let's just break and restart the scan.
                        break
                if improved:
                    break
            
        result = best_set
        
    return sorted(result)

# Ensure construct is the entry point
# The problem asks for construct() -> exactly 32 distinct integers in 1..512.
# The code above returns sorted(result) which is a list of ints.

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Result: {res}")
    print(f"Score: {_score_mask(0)}") # dummy
    mask = 0
    for r in res:
        mask |= (1 << r)
    print(f"Actual Score: {_score_mask(mask)}")
