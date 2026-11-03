import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def exact_score(U):
        """
        Compute exact score for a set U.
        Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
        """
        if not U:
            return 0.0
        U_set = set(U)
        max_U = max(U_set)
        if max_U == 0:
            return 0.0
        
        n = len(U_set)
        # For small sets, exact set operations are fast enough
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
        U_arr = np.array(sorted(U), dtype=np.int64)
        max_U = int(U_arr.max())
        if max_U == 0:
            return 0.0
        
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

    def simulated_annealing(initial_set, time_limit):
        """
        Simulated annealing to refine the set.
        State: a sorted list of nonnegative integers containing 0.
        Moves:
        1. Replace a random non-zero element with a new random value.
        2. Add a new random element (if size < max_size).
        3. Remove a random non-zero element (if size > min_size).
        4. Shift a random non-zero element by a small delta.
        
        We use a fast approximate score for evaluation during the search,
        and exact score for final verification.
        """
        current_set = sorted(set(initial_set))
        current_score = fast_score_numpy(current_set)
        best_set = current_set[:]
        best_score = current_score
        
        # Annealing parameters
        T = 1.0
        cooling_rate = 0.995
        min_T = 0.01
        max_size = 100
        min_size = 2
        
        # Precompute some random candidates for efficiency? No, generate on the fly.
        
        while time.time() < time_limit and T > min_T:
            if len(current_set) == 0:
                break
                
            # Choose a move
            move_type = random.random()
            n = len(current_set)
            
            if move_type < 0.4 and n > min_size:
                # Replace a random non-zero element
                idx = random.randint(1, n - 1)
                old_val = current_set[idx]
                # New value: random in range or near old value
                if random.random() < 0.5:
                    new_val = random.randint(1, max(current_set) + 10)
                else:
                    new_val = max(1, old_val + random.randint(-10, 10))
                
                if new_val in current_set:
                    continue
                    
                new_set = current_set[:]
                new_set[idx] = new_val
                new_set = sorted(set(new_set))
                
            elif move_type < 0.7 and n < max_size:
                # Add a new element
                new_val = random.randint(1, max(current_set) + 10)
                if new_val in current_set:
                    continue
                new_set = current_set + [new_val]
                new_set = sorted(set(new_set))
                
            elif move_type < 0.85 and n > min_size:
                # Remove a random non-zero element
                idx = random.randint(1, n - 1)
                new_set = current_set[:]
                del new_set[idx]
                
            else:
                # Shift a random non-zero element
                if n <= 1:
                    continue
                idx = random.randint(1, n - 1)
                old_val = current_set[idx]
                delta = random.randint(-5, 5)
                new_val = old_val + delta
                if new_val <= 0 or new_val in current_set:
                    continue
                new_set = current_set[:]
                new_set[idx] = new_val
                new_set = sorted(set(new_set))
            
            if not new_set or 0 not in new_set:
                continue
                
            new_score = fast_score_numpy(new_set)
            delta_score = new_score - current_score
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta_score > 0 or random.random() < math.exp(delta_score / T):
                current_set = new_set
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_set = current_set[:]
            
            T *= cooling_rate
            
        return best_set, best_score

    # Phase 1: Generate diverse initial populations
    # Use small sets with high ratio potential
    initial_candidates = []
    
    # 1. Small geometric-like sets
    for base in [2, 3, 4, 5]:
        for length in [4, 5, 6, 7]:
            s = [0] + [base**i for i in range(1, length)]
            initial_candidates.append(s)
            
    # 2. Random small sets
    for _ in range(20):
        size = random.randint(3, 10)
        s = [0]
        for _ in range(size - 1):
            s.append(random.randint(1, 50))
        initial_candidates.append(sorted(set(s)))
        
    # 3. The known good seed-like structure
    initial_candidates.append([0, 1, 2, 4, 8])
    initial_candidates.append([0, 1, 3, 7, 15, 31])
    
    best_overall_score = 0.0
    best_overall_set = [0, 1]
    
    # Evaluate initial candidates
    for cand in initial_candidates:
        if not cand:
            continue
        score = fast_score_numpy(cand)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_set = cand[:]
            
    # Phase 2: Simulated Annealing from the best initial candidate
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        improved_set, improved_score = simulated_annealing(best_overall_set, time.time() + remaining_time)
        if improved_score > best_overall_score:
            best_overall_score = improved_score
            best_overall_set = improved_set[:]
            
    # Phase 3: Try a few more SA runs from different random starts if time permits
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            # Random start
            size = random.randint(3, 15)
            start_set = [0] + random.sample(range(1, 100), size - 1)
            start_set = sorted(set(start_set))
            
            improved_set, improved_score = simulated_annealing(start_set, time.time() + 5)
            if improved_score > best_overall_score:
                best_overall_score = improved_score
                best_overall_set = improved_set[:]
                
    # Final verification with exact score
    final_score = exact_score(best_overall_set)
    if final_score < best_overall_score:
        # If numpy approximation was optimistic, we still return the set, 
        # but the evaluator will use the exact score.
        # We can't change the set now without more time.
        pass
        
    return best_overall_set

if __name__ == "__main__":
    U = construct()
    print(f"Result: {U}")
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
