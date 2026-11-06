"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 256
    
    # 1. Greedy Initialization to establish a strong initial lower bound
    def greedy_init():
        S = set()
        in_S = [False] * (N + 1)
        for x in range(1, N + 1):
            valid = True
            for a in S:
                # Check 3 cases of 3-AP involving x and a
                # 1. x is middle: a, x, c => c = 2x - a
                c = 2 * x - a
                if c in S:
                    valid = False
                    break
                # 2. a is middle: x, a, c => c = 2a - x
                c = 2 * a - x
                if c in S:
                    valid = False
                    break
                # 3. b is middle: x, b, a => b = (x+a)/2
                s = x + a
                if s % 2 == 0:
                    b = s // 2
                    if b in S:
                        valid = False
                        break
            if valid:
                S.add(x)
                in_S[x] = True
        return S

    best_set = greedy_init()
    best_score = sum(Fraction(1, x) for x in best_set)
    
    # 2. Precomputation for Branch and Bound
    # We process numbers in order of decreasing weight: 1, 2, ..., 256
    indices = list(range(1, N + 1))
    weights = [Fraction(1, i) for i in indices]
    
    # Suffix sums of weights for upper bound pruning
    # suffix[i] = sum(weights[i:])
    suffix = [Fraction(0)] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix[i] = suffix[i + 1] + weights[i]
        
    # 3. Branch and Bound Search
    # State:
    # k: current index in indices
    # current_sum: sum of weights of selected elements so far
    # selected_list: list of selected elements
    # in_selected: boolean array for O(1) membership check
    
    selected_list = []
    in_selected = [False] * (N + 1)
    
    def dfs(k, current_sum):
        nonlocal best_score, best_set
        
        # Time guard
        if time.time() > deadline:
            return
            
        # Pruning: Upper Bound
        # The maximum additional weight we can get from indices k..N-1 is suffix[k]
        upper_bound = current_sum + suffix[k]
        if upper_bound <= best_score:
            return
            
        # Base Case
        if k == N:
            if current_sum > best_score:
                best_score = current_sum
                best_set = set(selected_list)
            return
            
        x = indices[k]
        w = weights[k]
        
        # Check if x can be added to current selection
        # It must not form a 3-AP with any two existing elements
        valid = True
        for a in selected_list:
            # Case 1: x is middle. a, x, c. c = 2x - a.
            c = 2 * x - a
            if 1 <= c <= N and in_selected[c]:
                valid = False
                break
            # Case 2: a is middle. x, a, c. c = 2a - x.
            c = 2 * a - x
            if 1 <= c <= N and in_selected[c]:
                valid = False
                break
            # Case 3: b is middle. x, b, a. b = (x+a)/2.
            s = x + a
            if s % 2 == 0:
                b = s // 2
                if 1 <= b <= N and in_selected[b]:
                    valid = False
                    break
            
        # Branch 1: Include x (if valid)
        # We try Include first because x has high weight (small value)
        if valid:
            selected_list.append(x)
            in_selected[x] = True
            dfs(k + 1, current_sum + w)
            # Backtrack
            selected_list.pop()
            in_selected[x] = False
            
        # Branch 2: Exclude x
        dfs(k + 1, current_sum)

    # Start the search
    dfs(0, Fraction(0, 1))
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
