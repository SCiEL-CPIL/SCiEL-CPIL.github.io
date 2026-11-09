"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Find a maximum-size subset S of {1..192} such that no element a in S
    divides the sum of any 3 distinct OTHER elements b,c,d in S.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    N = 192
    MAX_K = 48

    # Precompute for each a, the set of 3-sums from {1..192}\{a} that are divisible by a.
    # Actually, we need: for a given a, and a set S, check if there exist b,c,d in S\{a}
    # with b+c+d ≡ 0 (mod a).
    
    # Strategy: 
    # 1. Start with a greedy construction (large numbers first, as in seed).
    # 2. Try to improve via local search: swap elements to increase size.
    # 3. Use randomized restarts with different orderings.
    
    def is_valid_addition(candidate_set, a):
        """Check if adding 'a' to candidate_set violates the constraint."""
        others = [v for v in candidate_set if v != a]
        if len(others) < 3:
            return True
        # For the new element 'a': check if a divides sum of any 3 distinct others
        for combo in combinations(others, 3):
            if sum(combo) % a == 0:
                return False
        # For each existing element 'b' in candidate_set: check if b divides sum of any 3 distinct others (including a)
        for b in candidate_set:
            if b == a:
                continue
            others_for_b = [v for v in candidate_set if v != b]
            # others_for_b includes 'a' now
            for combo in combinations(others_for_b, 3):
                if sum(combo) % b == 0:
                    return False
        return True
    
    def is_valid_set(S):
        """Check if set S is valid."""
        S_list = list(S)
        for a in S_list:
            others = [v for v in S_list if v != a]
            if len(others) < 3:
                continue
            for combo in combinations(others, 3):
                if sum(combo) % a == 0:
                    return False
        return True
    
    def greedy(order):
        """Greedy construction with given order of candidates."""
        selected = []
        for x in order:
            if x in selected:
                continue
            # Check if we can add x
            can_add = True
            others = [v for v in selected if v != x]
            # Check x against existing
            for combo in combinations(others, 3):
                if sum(combo) % x == 0:
                    can_add = False
                    break
            if not can_add:
                continue
            # Check existing against x
            for b in selected:
                others_for_b = [v for v in selected if v != b]
                # Add x to the pool for b
                # We need to check if b divides sum of any 3 from (others_for_b + [x])
                # This is O(C(len(others_for_b)+1, 3)) which is expensive
                # Optimization: only check combos that include x, since combos without x were already valid
                # So we check: does b divide (x + c1 + c2) for any c1,c2 in others_for_b?
                for c1, c2 in combinations(others_for_b, 2):
                    if (x + c1 + c2) % b == 0:
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                selected.append(x)
        return selected
    
    def local_search(S, max_iter=1000):
        """Try to improve set S by swapping elements."""
        S_set = set(S)
        best_S = list(S_set)
        best_size = len(best_S)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            # Try to add an element
            for x in range(1, N+1):
                if x in S_set:
                    continue
                new_set = S_set | {x}
                if is_valid_set(new_set):
                    S_set = new_set
                    best_S = list(S_set)
                    best_size += 1
                    improved = True
                    break
            if improved:
                continue
            # Try to swap: remove one, add another
            for remove in list(S_set):
                for x in range(1, N+1):
                    if x in S_set:
                        continue
                    new_set = (S_set - {remove}) | {x}
                    if len(new_set) >= best_size and is_valid_set(new_set):
                        S_set = new_set
                        best_S = list(S_set)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best_S
    
    # Initial greedy: large to small (seed approach)
    best_result = greedy(range(N, 0, -1))
    
    # Try randomized orderings
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        result = greedy(order)
        if len(result) > len(best_result):
            best_result = result
    
    # Local search on best result
    if time.time() < deadline:
        improved = local_search(best_result, max_iter=200)
        if len(improved) > len(best_result):
            best_result = improved
    
    # More randomized restarts with local search
    for trial in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        result = greedy(order)
        result = local_search(result, max_iter=100)
        if len(result) > len(best_result):
            best_result = result
    
    # Final check
    if not is_valid_set(set(best_result)):
        # Fallback to greedy large-to-small
        best_result = greedy(range(N, 0, -1))
    
    return best_result[:MAX_K]
# EVOLVE-BLOCK-END
