import random
import time
from itertools import combinations

def is_sidon(subset):
    """Check if a tuple/list of integers is a Sidon set (unique pair sums)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def has_sidon_subset(S, size):
    """Check if any subset of S of given size is Sidon."""
    for comb in combinations(S, size):
        if is_sidon(comb):
            return True
    return False

def get_max_sidon_size(S):
    """Find the size of the largest Sidon subset of S."""
    # Check down from 12
    for k in range(12, 2, -1):
        if has_sidon_subset(S, k):
            return k
    if has_sidon_subset(S, 3):
        return 3
    return 2

def construct():
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
    best_set = None
    best_score = float('inf')
    
    # Greedy function
    def greedy_construct(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        current_set = []
        candidates = list(range(0, 129))
        sums = set()
        
        for i in range(12):
            best_x = None
            max_collisions = -1
            best_candidates = []
            
            for x in candidates:
                if x in current_set:
                    continue
                
                new_sums = [2 * x] + [x + c for c in current_set]
                collisions = sum(1 for s in new_sums if s in sums)
                
                # Tie breaking: prefer x that maximizes potential future collisions?
                # For now, just random among max collisions
                if collisions > max_collisions:
                    max_collisions = collisions
                    best_candidates = [x]
                elif collisions == max_collisions:
                    best_candidates.append(x)
            
            if not best_candidates:
                best_x = random.choice(candidates)
            else:
                best_x = random.choice(best_candidates)
            
            current_set.append(best_x)
            candidates.remove(best_x)
            sums.add(2 * best_x)
            for c in current_set:
                if c != best_x:
                    sums.add(best_x + c)
        
        return current_set

    # Local search function
    def local_search(S, max_iters=20):
        S = S.copy()
        current_score = get_max_sidon_size(S)
        
        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            improved = False
            # Try swapping each element
            for i in range(len(S)):
                if time.time() - start_time > time_limit:
                    break
                    
                x = S[i]
                # Try replacing x with some y not in S
                # Sample a few y's to keep it fast
                others = [y for y in range(129) if y not in S]
                random.shuffle(others)
                
                # Check a limited number of swaps
                for y in others[:30]:
                    new_S = S.copy()
                    new_S[i] = y
                    new_score = get_max_sidon_size(new_S)
                    if new_score < current_score:
                        S = new_S
                        current_score = new_score
                        improved = True
                        break # Restart loop with new S
                if improved:
                    break
            if not improved:
                break
        return S, current_score

    # Run multiple trials
    for trial in range(10):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Get initial set
        S_init = greedy_construct(trial)
        
        # Refine
        S_refined, score = local_search(S_init, max_iters=10)
        
        if score < best_score:
            best_score = score
            best_set = S_refined
            
        if best_score == 2:
            # 2 is the theoretical minimum?
            # If we find 2, it's likely optimal.
            break

    if best_set is None:
        best_set = greedy_construct(0)
        
    return best_set
