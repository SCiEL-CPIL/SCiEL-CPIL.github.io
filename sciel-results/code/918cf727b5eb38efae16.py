"""Improved program for the suite task; EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 256
    random.seed(42)
    
    def is_valid_set(S):
        """Check if set S satisfies the property: for every a<b<=c in S, a does not divide b+c."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        for i in range(m):
            a = S_sorted[i]
            for j in range(i + 1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def count_violations(S):
        """Count total number of violating triples."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        count = 0
        for i in range(m):
            a = S_sorted[i]
            for j in range(i + 1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        count += 1
        return count
    
    def violations_involving(S, x):
        """Count violations that involve element x in any role."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        count = 0
        for i in range(m):
            a = S_sorted[i]
            for j in range(i + 1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        if a == x or b == x or c == x:
                            count += 1
        return count
    
    # Strategy 1: Greedy construction - add elements one by one, trying to maximize size
    def greedy_construct(order):
        """Build a valid set by greedily adding elements in the given order."""
        S = []
        S_set = set()
        for x in order:
            if time.time() > deadline:
                break
            S_new = S + [x]
            if is_valid_set(S_new):
                S = S_new
                S_set.add(x)
        return S
    
    # Strategy 2: Start from upper tail, try to add smaller elements
    def tail_plus_add(start_val):
        """Start with [start_val, 256] and try to add smaller elements."""
        S = list(range(start_val, 257))
        S_set = set(S)
        for candidate in range(1, start_val):
            if time.time() > deadline:
                break
            S_new = S + [candidate]
            if is_valid_set(S_new):
                S = S_new
                S_set.add(candidate)
        return S
    
    # Strategy 3: Local search - start from a valid set, try 1-remove-1-add moves
    def local_search(initial_S, max_iters=5000):
        """Improve a valid set using 1-remove-1-add moves."""
        S = list(initial_S)
        S_set = set(S)
        best_size = len(S)
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if no_improve > 200:
                break
            
            improved = False
            # Try removing one element and adding another
            for remove_x in list(S_set):
                if time.time() > deadline:
                    break
                S_without = [x for x in S if x != remove_x]
                S_without_set = S_set - {remove_x}
                
                # Try adding each candidate not in S_without
                for add_x in range(1, 257):
                    if add_x in S_without_set:
                        continue
                    S_new = S_without + [add_x]
                    if is_valid_set(S_new):
                        S = S_new
                        S_set = S_without_set | {add_x}
                        best_size = len(S)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
        
        return S, best_size
    
    # Strategy 4: Start from full set, remove elements that cause most violations
    def removal_heuristic():
        """Start from [1,256] and remove elements causing most violations."""
        S = list(range(1, 257))
        S_set = set(S)
        
        while True:
            if time.time() > deadline:
                break
            # Count violations for each element
            violations = {}
            for x in S_set:
                violations[x] = violations_involving(S, x)
            
            # Find element with most violations
            if not violations or max(violations.values()) == 0:
                break
            
            worst = max(violations, key=violations.get)
            S = [x for x in S if x != worst]
            S_set.remove(worst)
        
        return S
    
    best_S = []
    best_size = 0
    
    # Try multiple strategies
    # 1. Upper tail with different starting points
    for start_val in [200, 190, 180, 175, 170, 165, 160, 155, 150, 140, 130]:
        if time.time() > deadline:
            break
        S = tail_plus_add(start_val)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
    
    # 2. Greedy from different orderings
    # Descending order
    if time.time() < deadline - 10:
        S = greedy_construct(range(256, 0, -1))
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
    
    # Ascending order
    if time.time() < deadline - 10:
        S = greedy_construct(range(1, 257))
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
    
    # 3. Removal heuristic
    if time.time() < deadline - 15:
        S = removal_heuristic()
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
    
    # 4. Local search from best found
    if time.time() < deadline - 5 and best_S:
        S, size = local_search(best_S)
        if size > best_size:
            best_S = S
            best_size = size
    
    # 5. More local search iterations with random restarts
    if time.time() < deadline - 5:
        for restart in range(5):
            if time.time() > deadline - 2:
                break
            # Create a perturbed version
            S_perturbed = list(best_S)
            # Remove a few random elements
            for _ in range(random.randint(1, 3)):
                if S_perturbed:
                    idx = random.randint(0, len(S_perturbed) - 1)
                    S_perturbed.pop(idx)
            
            if is_valid_set(S_perturbed):
                S, size = local_search(S_perturbed, max_iters=2000)
                if size > best_size:
                    best_S = S
                    best_size = size
    
    return sorted(best_S)
# EVOLVE-BLOCK-END
