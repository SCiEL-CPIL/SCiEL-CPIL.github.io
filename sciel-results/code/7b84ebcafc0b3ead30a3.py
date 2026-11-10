import time
import random
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 290  # Safety margin
    
    # Precompute all points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def is_compatible(p, S):
        """
        Check if point p can be added to set S without forming an isosceles triangle.
        S is a list of (x, y) tuples.
        """
        px, py = p
        for a in S:
            ax, ay = a
            d_pa2 = (px - ax)**2 + (py - ay)**2
            for b in S:
                if b == a:
                    continue
                bx, by = b
                d_pb2 = (px - bx)**2 + (py - by)**2
                d_ab2 = (ax - bx)**2 + (ay - by)**2
                
                # Check for isosceles
                if d_pa2 == d_pb2 or d_pa2 == d_ab2 or d_pb2 == d_ab2:
                    return False
        return True

    def greedy_construct(order):
        S = []
        for p in order:
            if is_compatible(p, S):
                S.append(p)
        return S

    def local_search(S):
        """
        Try to improve S by removing one point and adding two.
        Returns the improved set.
        """
        S = list(S)
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try each point in S
            for i in range(len(S)):
                p_remove = S[i]
                S_temp = S[:i] + S[i+1:]
                
                # Find candidates compatible with S_temp
                candidates = []
                for p in all_points:
                    if p in S_temp:
                        continue
                    if is_compatible(p, S_temp):
                        candidates.append(p)
                
                if len(candidates) >= 2:
                    # Try to pick 2 candidates that are compatible with each other
                    # Random sampling from candidates to find a valid pair
                    # If candidates list is large, this is expensive. 
                    # Limit candidates to first 50 for speed?
                    # Or just try a few random pairs.
                    
                    # Sort candidates by "randomness" or just take a subset
                    # To be fast, we only try a few random pairs
                    tried = 0
                    while tried < 10 and time.time() - start_time < time_limit:
                        c1, c2 = random.sample(candidates, 2)
                        # Check if c1 and c2 are compatible with each other (and S_temp)
                        # is_compatible(c1, S_temp) is True.
                        # is_compatible(c2, S_temp) is True.
                        # Need to check if (c1, c2) form isosceles with any point in S_temp?
                        # No, is_compatible(c2, S_temp + [c1]) checks this.
                        if is_compatible(c2, S_temp + [c1]):
                            S_new = S_temp + [c1, c2]
                            # Verify full validity just in case (optional, but safe)
                            # S_new is valid by construction if checks passed.
                            S = S_new
                            improved = True
                            break
                        tried += 1
                    if improved:
                        break # Restart the for loop with new S
        return S

    best_set = []
    best_score = 0
    
    # Run multiple greedy + local search iterations
    while time.time() - start_time < time_limit:
        random.shuffle(all_points)
        S = greedy_construct(all_points)
        S = local_search(S)
        
        if len(S) > best_score:
            best_score = len(S)
            best_set = S
            # If we hit a high score, we might want to keep searching for better
            # But if we are close to the limit, stop?
            # The loop condition handles the stop.

    return best_set
# EVOLVE-BLOCK-END
