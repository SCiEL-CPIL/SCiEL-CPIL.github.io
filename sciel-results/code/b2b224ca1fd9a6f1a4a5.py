import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def compute_score(U):
        """
        Compute the score for a set U.
        Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
        """
        if not U:
            return 0.0
        U_set = set(U)
        max_U = max(U_set)
        if max_U == 0:
            return 0.0
        
        # For efficiency, we only compute exact scores for smaller sets or at the end.
        # For intermediate steps in the constructive algorithm, we use approximations or 
        # direct set operations if the set is small enough.
        
        n = len(U_set)
        if n > 200:
            # Approximation for large sets to avoid O(n^2) blowup during search
            # This is a rough estimate. We rely on exact calculation for final candidates.
            # For now, let's just return a lower bound or 0 to force exact calc only on small sets?
            # No, the constructive algorithm needs feedback. 
            # If n is large, O(n^2) is too slow for 115s if we do it many times.
            # We will restrict the constructive search to n <= 100 where O(n^2) is fast (~10k ops).
            pass

        sums = set()
        diffs = set()
        for a in U_set:
            for b in U_set:
                sums.add(a + b)
                diffs.add(a - b)
        
        if not sums:
            return 0.0
            
        ratio = len(diffs) / len(sums)
        if ratio <= 0:
            return 0.0
        
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0
        return exponent

    def fast_score_numpy(U):
        """
        Fast score using numpy for sets up to size ~500.
        """
        import numpy as np
        U_arr = np.array(sorted(U), dtype=np.int64)
        max_U = int(U_arr.max())
        if max_U == 0:
            return 0.0
        
        # Compute sumset and diffset
        # Memory check: n=500 -> 250k elements. Fine.
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        sumset_size = len(np.unique(sums))
        diffset_size = len(np.unique(diffs))
        
        ratio = diffset_size / sumset_size
        if ratio <= 0:
            return 0.0
        
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0
        return exponent

    def greedy_constructive_search(max_n=80, max_val=5000):
        """
        Construct a set by greedily adding elements that maximize the score.
        We start with {0, 1} and add one element at a time.
        At each step, we try candidate values and pick the one that yields the highest score.
        To make this efficient, we maintain the set and only check candidates that are 'promising'.
        
        Strategy:
        1. Start with U = {0, 1}.
        2. For size k from 3 to max_n:
           a. Generate candidate values for the next element.
           b. For each candidate, calculate the new score.
           c. Pick the best candidate.
           
        To limit candidates:
        - Try small integers (1 to 50).
        - Try integers near current max (max-10 to max+50).
        - Try random integers.
        """
        U = [0, 1]
        best_score = fast_score_numpy(U)
        
        for target_size in range(3, max_n + 1):
            if time.time() > deadline - 5:
                break
                
            best_candidate = None
            best_candidate_score = -1.0
            
            # Generate candidates
            candidates = set()
            current_max = max(U)
            
            # Small values
            for v in range(2, 30):
                if v not in U:
                    candidates.add(v)
            
            # Values near max
            for v in range(max(2, current_max - 20), current_max + 50):
                if v not in U:
                    candidates.add(v)
                    
            # Random values
            for _ in range(50):
                v = random.randint(1, max_val)
                if v not in U:
                    candidates.add(v)
            
            # Evaluate candidates
            # To speed up, we can prune candidates that are too large (increasing denominator)
            # or too small (likely to cause sum collisions that don't help diffset).
            
            for v in candidates:
                new_U = U + [v]
                # Sort and unique
                new_U = sorted(set(new_U))
                
                # Quick check: if new max is too large, score might drop
                # But let's just compute score.
                # Optimization: If len(new_U) is large, fast_score_numpy is O(n^2).
                # For n=80, n^2=6400. 100 candidates * 6400 ops = 640k ops per step.
                # 80 steps * 640k = 50M ops. Should be fine in Python/NumPy hybrid?
                # NumPy unique is fast.
                
                score = fast_score_numpy(new_U)
                
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = new_U
            
            if best_candidate is not None and best_candidate_score > best_score:
                U = best_candidate
                best_score = best_candidate_score
            else:
                # If no improvement, try to break ties or stop?
                # If we can't improve by adding an element, maybe we should stop.
                # Or try a different candidate generation strategy?
                # For now, if no improvement, stop growing.
                break
                
        return U, best_score

    def local_search_refinement(U, time_limit):
        """
        Refine the set using local search.
        """
        current_set = sorted(set(U))
        current_score = fast_score_numpy(current_set)
        best_set = current_set[:]
        best_score = current_score
        
        while time.time() < time_limit:
            improved = False
            n = len(current_set)
            
            # Randomly pick an index to modify (keep 0 fixed)
            if n <= 1:
                break
                
            idx = random.randint(1, n - 1)
            old_val = current_set[idx]
            
            # Try perturbations
            for _ in range(30):
                # Strategy: move value slightly, or replace with random
                if random.random() < 0.5:
                    shift = random.randint(-5, 5)
                    new_val = old_val + shift
                else:
                    max_val = max(current_set)
                    new_val = random.randint(1, max_val + 20)
                
                if new_val <= 0 or new_val in current_set:
                    continue
                
                candidate_set = current_set[:]
                candidate_set[idx] = new_val
                candidate_set = sorted(set(candidate_set))
                
                new_score = fast_score_numpy(candidate_set)
                
                if new_score > current_score:
                    current_set = candidate_set
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_set = current_set[:]
                    improved = True
                    break # Accept and move to next iteration
            
            if not improved:
                # Try a more aggressive move: swap two elements?
                # Or just continue.
                pass
                
        return best_set, best_score

    best_overall_score = 0.0
    best_overall_set = [0, 1]
    
    # Phase 1: Greedy Constructive Search
    # Run multiple times with different random seeds for candidate generation
    for run in range(3):
        if time.time() > deadline - 20:
            break
        random.seed(run)
        U, score = greedy_constructive_search(max_n=60, max_val=2000)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_set = U[:]

    # Phase 2: Local Search Refinement
    remaining_time = deadline - time.time() - 2
    if remaining_time > 5:
        # Refine the best set found
        improved_set, improved_score = local_search_refinement(best_overall_set, time.time() + remaining_time)
        if improved_score > best_overall_score:
            best_overall_score = improved_score
            best_overall_set = improved_set[:]
            
        # Also try refining a standard quadratic set for comparison
        U_quad = [i*i for i in range(30)]
        s_quad = fast_score_numpy(U_quad)
        if s_quad > best_overall_score:
            best_overall_score = s_quad
            best_overall_set = U_quad[:]
            
        if time.time() < deadline - 2:
            improved_set2, improved_score2 = local_search_refinement(best_overall_set, time.time() + 5)
            if improved_score2 > best_overall_score:
                best_overall_score = improved_score2
                best_overall_set = improved_set2[:]

    return best_overall_set

if __name__ == "__main__":
    U = construct()
    print(f"Result: {U}")
    # Calculate final score for verification
    if U:
        max_U = max(U)
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a+b)
                diffs.add(a-b)
        if sums:
            ratio = len(diffs)/len(sums)
            score = 1 + math.log(ratio)/math.log(2*max_U+1)
            print(f"Score: {score}")
