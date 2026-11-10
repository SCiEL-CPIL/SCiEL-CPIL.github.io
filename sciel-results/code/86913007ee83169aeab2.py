import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0
    N = 100
    target = 200

    # Bitset for subset sums
    # dp[s] is a bitset of elements used? No, that's for reconstruction.
    # We just need to know if a sum is reachable.
    
    def can_form_sum(elements, target):
        """Check if any subset of elements sums to target."""
        reachable = np.zeros(target + 1, dtype=bool)
        reachable[0] = True
        for x in elements:
            if x > target:
                continue
            # Shift reachable by x
            # new_reachable = reachable | (reachable << x)
            # But we only care up to target
            shifted = np.zeros(target + 1, dtype=bool)
            shifted[x:] = reachable[:target - x + 1]
            reachable = reachable | shifted
            if reachable[target]:
                return True
        return False

    def find_witness(elements, target):
        """Find one subset of elements summing to target."""
        n = len(elements)
        # DP to find one path
        parent = [-1] * (target + 1)
        parent_elem = [-1] * (target + 1)
        reachable = [False] * (target + 1)
        reachable[0] = True
        
        for i, x in enumerate(elements):
            if x > target:
                continue
            for s in range(target - x, -1, -1):
                if reachable[s] and not reachable[s + x]:
                    reachable[s + x] = True
                    parent[s + x] = s
                    parent_elem[s + x] = i
                    if s + x == target:
                        # Reconstruct
                        subset = []
                        cur = target
                        while cur > 0:
                            prev = parent[cur]
                            elem_idx = parent_elem[cur]
                            subset.append(elements[elem_idx])
                            cur = prev
                        return subset
        return None

    # Initial solution: Greedy Largest First
    def greedy_largest():
        selected = []
        # We want to select elements such that no subset sums to 200.
        # Greedy: Add element if it doesn't create a subset sum of 200.
        # To check this, we need to know if 200 - x is a subset sum of currently selected.
        # This is O(N * T * N) if we recompute DP each time.
        # N=100, T=200. 100 * 200 * 100 = 2,000,000 ops. Very fast.
        
        reachable = {0}
        for x in range(N, 0, -1):
            new_sums = {s + x for s in reachable if s + x <= target}
            if target not in new_sums:
                selected.append(x)
                reachable |= new_sums
        return selected

    best_selected = greedy_largest()
    best_size = len(best_selected)
    
    # Greedy Smallest First
    def greedy_smallest():
        selected = []
        reachable = {0}
        for x in range(1, N + 1):
            new_sums = {s + x for s in reachable if s + x <= target}
            if target not in new_sums:
                selected.append(x)
                reachable |= new_sums
        return selected

    sel2 = greedy_smallest()
    if len(sel2) > best_size:
        best_selected = sel2
        best_size = len(best_selected)

    # Hitting Set Approach
    # Start with all elements, remove those in witness subsets.
    
    def solve_hitting_set(seed):
        rng = random.Random(seed)
        remaining = list(range(1, N + 1))
        removed = set()
        
        for _ in range(100): # Max iterations
            if time.time() > deadline - 2:
                break
            if not remaining:
                break
            
            witness = find_witness(remaining, target)
            if witness is None:
                break
            
            # Pick an element from witness to remove
            # Heuristic: Pick the one that appears in the most "potential" witnesses?
            # Simple: Random or Max
            # Let's try removing the largest element in the witness?
            # Or the one with highest "degree" in the subset sum graph?
            # For now, random selection from witness.
            to_remove = rng.choice(witness)
            removed.add(to_remove)
            remaining = [x for x in remaining if x != to_remove]
            
        # Local Search: Try to add back removed elements
        for elem in sorted(removed, reverse=True):
            if time.time() > deadline - 2:
                break
            test_remaining = remaining + [elem]
            # Check if valid
            if not can_form_sum(test_remaining, target):
                remaining.append(elem)
                removed.discard(elem)
        
        return remaining

    for seed in range(50):
        if time.time() > deadline - 5:
            break
        res = solve_hitting_set(seed)
        if len(res) > best_size:
            best_selected = res
            best_size = len(best_selected)

    return best_selected
