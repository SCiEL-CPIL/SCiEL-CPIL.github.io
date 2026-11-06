import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    N = 512
    K = 32
    
    def score_bitset(b):
        """Count distinct d > 0 such that a, a+d, a+2d all in set (bit x-1 for x)."""
        count = 0
        # d ranges 1..255 (since max element 512, 2d <= 511)
        for d in range(1, 256):
            if (b & (b >> d) & (b >> (2 * d))):
                count += 1
        return count
    
    def set_to_bitset(S):
        b = 0
        for x in S:
            b |= (1 << (x - 1))
        return b
    
    def bitset_to_set(b):
        s = []
        for i in range(1, N + 1):
            if (b >> (i - 1)) & 1:
                s.append(i)
        return s
    
    def random_set():
        return random.sample(range(1, N + 1), K)
    
    def try_improve(current_set, current_score, time_limit):
        """Hill-climb: try swaps and add/remove pairs."""
        current_bit = set_to_bitset(current_set)
        current_score = score_bitset(current_bit)
        best_set = list(current_set)
        best_score = current_score
        best_bit = current_bit
        
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < time_limit and no_improve_count < max_no_improve:
            improved = False
            current_set = list(best_set)
            current_bit = best_bit
            current_score = best_score
            
            # Generate candidate moves: swap one element with another
            # Try all pairs (i in set, j not in set)
            in_set = set(current_set)
            not_in_set = [x for x in range(1, N + 1) if x not in in_set]
            
            # Limit candidates for speed
            random.shuffle(not_in_set)
            candidates = not_in_set[:300]
            
            for i in current_set:
                for j in candidates:
                    # Swap i with j
                    new_bit = current_bit & ~(1 << (i - 1))
                    new_bit |= (1 << (j - 1))
                    new_score = score_bitset(new_bit)
                    if new_score > current_score:
                        current_score = new_score
                        current_bit = new_bit
                        current_set.remove(i)
                        current_set.append(j)
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(current_set)
                            best_bit = current_bit
                        break
                if improved:
                    break
            
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0
        
        return best_set, best_score
    
    # Multiple restarts
    best_overall_set = None
    best_overall_score = 0
    
    # First, try a structured initial guess: dense block
    # A block of 32 consecutive numbers: differences 1..31 all work, 
    # but also some larger ones. Let's compute.
    dense_set = list(range(1, 33))
    dense_score = score_bitset(set_to_bitset(dense_set))
    
    if dense_score > best_overall_score:
        best_overall_score = dense_score
        best_overall_set = list(dense_set)
    
    # Try several random restarts with hill-climbing
    num_restarts = 20
    restart_time_budget = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Random initialization
        init_set = random_set()
        
        # Hill-climb with time budget for this restart
        restart_deadline = time.time() + min(restart_time_budget, 15.0)
        improved_set, improved_score = try_improve(init_set, 0, restart_deadline)
        
        if improved_score > best_overall_score:
            best_overall_score = improved_score
            best_overall_set = improved_set
    
    # Final polish: try more aggressive search on the best found
    if best_overall_set and time.time() < deadline - 15.0:
        final_deadline = time.time() + (deadline - time.time()) - 10.0
        final_set, final_score = try_improve(best_overall_set, best_overall_score, final_deadline)
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_overall_set = final_set
    
    # Ensure valid output
    result = sorted(best_overall_set) if best_overall_set else list(range(1, 33))
    assert len(result) == K
    assert all(1 <= x <= N for x in result)
    assert len(set(result)) == K
    
    return result

# EVOLVE-BLOCK-END
