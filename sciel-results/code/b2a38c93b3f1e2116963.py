import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Greedy Initialization
    # Pick largest numbers first that are coprime to all previously picked
    initial_set = []
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in initial_set):
            initial_set.append(a)
    
    best_set = list(initial_set)
    best_score = sum(1.0 / (512 - a) for a in best_set)
    
    # Helper to calculate score
    def get_score(s):
        return sum(1.0 / (512 - a) for a in s)

    # Local Search / Swap Optimization
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = list(best_set)
        current_score = best_score
        
        # Try removing each element P
        for i, P in enumerate(current_set):
            if time.time() > deadline:
                break
                
            # Temporarily remove P
            remaining = current_set[:i] + current_set[i+1:]
            
            # Find candidates: numbers not in remaining, coprime to all in remaining
            # AND that were blocked by P (i.e., gcd(a, P) > 1)
            # Actually, we just need to find numbers a in 1..511 such that:
            # 1. a not in remaining
            # 2. gcd(a, b) == 1 for all b in remaining
            # 3. We want to add a subset of these.
            # Note: If a number was not blocked by P, it would have been in the set already
            # (since greedy is maximal). So any valid candidate a must satisfy gcd(a, P) > 1.
            
            candidates = []
            for a in range(511, 0, -1):
                if a in remaining:
                    continue
                # Check if a is coprime to all in remaining
                is_coprime_to_all = True
                for b in remaining:
                    if gcd(a, b) != 1:
                        is_coprime_to_all = False
                        break
                if is_coprime_to_all:
                    # This a is a valid addition.
                    # It must have been blocked by P in the previous state, 
                    # so gcd(a, P) should be > 1. 
                    # If gcd(a, P) == 1, it means a was coprime to P too, 
                    # so it should have been in the set already. 
                    # Since it's not, this case shouldn't happen for a maximal set,
                    # but due to previous swaps, the set might not be "maximal" in the 
                    # simple greedy sense. So we just add it if it fits.
                    candidates.append(a)
            
            if not candidates:
                continue
                
            # Sort candidates by score descending (largest a first)
            candidates.sort(key=lambda x: 1.0 / (512 - x), reverse=True)
            
            # Greedily pick pairwise coprime subset from candidates
            new_additions = []
            for a in candidates:
                if all(gcd(a, b) == 1 for b in new_additions):
                    new_additions.append(a)
            
            # Calculate new score
            # Remove P, add new_additions
            # We can compute delta: -f(P) + sum(f(a) for a in new_additions)
            delta = -1.0 / (512 - P) + sum(1.0 / (512 - a) for a in new_additions)
            
            if delta > 1e-9:
                # Commit the change
                new_set = remaining + new_additions
                # Ensure distinctness (should be guaranteed)
                new_set = list(set(new_set))
                
                new_score = current_score + delta
                if new_score > best_score:
                    best_set = new_set
                    best_score = new_score
                    improved = True
                    # Restart loop with new best_set
                    break 

    return best_set
