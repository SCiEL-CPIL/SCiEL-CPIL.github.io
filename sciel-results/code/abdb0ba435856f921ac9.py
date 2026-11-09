import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    n = 512
    max_size = 48
    
    # Precompute powers of 2 as a good starting set
    # Powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256 (9 elements, all <= 512)
    # We can try to add more elements greedily
    
    # Start with a greedy approach
    # Order candidates by some heuristic
    
    def is_valid_set(S):
        """Check if set S is valid: no element is mean of any subset of >=2 other elements."""
        S_list = sorted(S)
        n = len(S_list)
        for i, a in enumerate(S_list):
            others = [x for x in S_list if x != a]
            m = len(others)
            for k in range(2, m + 1):
                for subset in combinations(others, k):
                    if sum(subset) == k * a:
                        return False
        return True
    
    # Greedy construction with incremental check
    def check_addition(S, x):
        """Check if adding x to S keeps the set valid."""
        if x in S:
            return False
        S_new = S | {x}
        S_list = sorted(S_new)
        n = len(S_list)
        
        # Check x is not mean of any subset of S of size >= 2
        S_list_no_x = [a for a in S_list if a != x]
        m = len(S_list_no_x)
        for k in range(2, m + 1):
            for subset in combinations(S_list_no_x, k):
                if sum(subset) == k * x:
                    return False
        
        # Check each a in S is not mean of any subset of (S - {a}) ∪ {x} of size >= 2
        for a in S:
            others = [b for b in S_new if b != a]
            m = len(others)
            for k in range(2, m + 1):
                for subset in combinations(others, k):
                    if sum(subset) == k * a:
                        return False
        return True
    
    # Try greedy with different orderings
    best_set = set()
    
    # Start with powers of 2
    powers_of_2 = [2**i for i in range(0, 10) if 2**i <= 512]
    for p in powers_of_2:
        if len(best_set) < max_size and check_addition(best_set, p):
            best_set.add(p)
    
    # Now try to add more elements
    candidates = list(range(1, 513))
    
    # Try multiple passes with different orderings
    for order in ['ascending', 'descending', 'random']:
        if time.time() > deadline:
            break
        
        if order == 'ascending':
            cand_list = list(range(1, 513))
        elif order == 'descending':
            cand_list = list(range(512, 0, -1))
        else:
            import random
            random.seed(42)
            cand_list = list(range(1, 513))
            random.shuffle(cand_list)
        
        for x in cand_list:
            if time.time() > deadline:
                break
            if len(best_set) >= max_size:
                break
            if x not in best_set and check_addition(best_set, x):
                best_set.add(x)
    
    # Try local search: swap elements
    if time.time() < deadline - 5:
        current_set = set(best_set)
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            improved = False
            # Try removing an element and adding a different one
            for a in list(current_set):
                if time.time() > deadline - 2:
                    break
                current_set.remove(a)
                # Try to add a new element
                for x in range(1, 513):
                    if x not in current_set and check_addition(current_set, x):
                        current_set.add(x)
                        improved = True
                        break
                if len(current_set) >= len(best_set):
                    best_set = set(current_set)
                current_set.add(a)  # restore
                if improved:
                    break
            if not improved:
                break
    
    return sorted(list(best_set))
