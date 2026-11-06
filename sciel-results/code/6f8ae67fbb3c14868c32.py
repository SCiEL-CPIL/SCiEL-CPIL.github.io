"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # Pre-compute conflict scores
    # Score(x) = number of pairs (u, v) in [1, 512] \ {x} such that x = (u+v)/2
    # This is the number of 3-term APs centered at x.
    conflict_scores = {}
    for x in range(1, 513):
        count = 0
        start_u = max(1, 2*x - 512)
        end_u = x - 1
        if start_u <= end_u:
            count = end_u - start_u + 1
        conflict_scores[x] = count

    def _is_valid_set(S):
        """
        Check if S is valid.
        Returns True if no element is the mean of any subset of other elements of size >= 2.
        Uses per-element subset-sum DP with bitsets.
        """
        n = len(S)
        if n < 3:
            return True
        
        S_list = list(S)
        max_val = max(S_list)
        
        for i, a in enumerate(S_list):
            # Build DP for S \ {a}
            n_t = n - 1
            dp = [0] * (n_t + 1)
            dp[0] = 1  # sum 0 with 0 elements
            
            for j, x in enumerate(S_list):
                if j == i:
                    continue
                # Update in reverse
                for k in range(n_t, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << x)
                        
            # Check if k*a is in dp[k] for any k >= 2
            for k in range(2, n_t + 1):
                target = k * a
                if target > k * max_val:
                    break
                if (dp[k] >> target) & 1:
                    return False
                    
        return True

    def _count_conflicts(S):
        """
        Calculate a badness score for each element.
        Score = number of k >= 2 such that k*a is achievable by S\{a}.
        """
        n = len(S)
        if n < 3:
            return {a: 0 for a in S}
            
        S_list = list(S)
        max_val = max(S_list)
        scores = {a: 0 for a in S_list}
        
        for i, a in enumerate(S_list):
            n_t = n - 1
            dp = [0] * (n_t + 1)
            dp[0] = 1
            
            for j, x in enumerate(S_list):
                if j == i:
                    continue
                for k in range(n_t, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << x)
                        
            for k in range(2, n_t + 1):
                target = k * a
                if target > k * max_val:
                    break
                if (dp[k] >> target) & 1:
                    scores[a] += 1
                    
        return scores

    def _repair_set(S):
        """
        Iteratively remove elements until S is valid.
        Returns the valid subset.
        """
        S = list(S)
        while not _is_valid_set(S):
            if time.time() > deadline:
                break
            scores = _count_conflicts(S)
            max_score = max(scores.values())
            if max_score == 0:
                # Should not happen if _is_valid_set returned False, but safety
                break
            candidates = [a for a, s in scores.items() if s == max_score]
            to_remove = random.choice(candidates)
            S.remove(to_remove)
            
        return S

    def _local_search(S, removed_elements, max_iters=100):
        """
        Try to add back removed elements one by one if the set remains valid.
        Also try swaps: remove one from S, add one from removed_elements.
        """
        S = list(S)
        removed = list(removed_elements)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try adding back each removed element
            for i, elem in enumerate(removed):
                if time.time() > deadline:
                    return S
                candidate = S + [elem]
                if _is_valid_set(candidate):
                    S = candidate
                    removed.pop(i)
                    improved = True
                    break
            
            # Try swaps: for each s in S and r in removed, check if S - {s} + {r} is valid
            if not improved and len(removed) > 0:
                # Limit swaps to avoid O(N^2) explosion
                # Only try a few random swaps
                for _ in range(5):
                    if time.time() > deadline:
                        return S
                    si = random.randint(0, len(S)-1)
                    ri = random.randint(0, len(removed)-1)
                    s_elem = S[si]
                    r_elem = removed[ri]
                    
                    new_S = S[:si] + S[si+1:] + [r_elem]
                    if _is_valid_set(new_S):
                        S = new_S
                        removed.pop(ri)
                        improved = True
                        break
                    
        return S

    best_set = []
    best_size = 0

    # Phase 1: Greedy construction with different orderings
    # Heuristic 1: Sort by conflict score ascending (safest first)
    order_safe = sorted(range(1, 513), key=lambda x: conflict_scores[x])
    S = []
    for x in order_safe:
        if time.time() > deadline - 10:
            break
        new_S = S + [x]
        if _is_valid_set(new_S):
            S.append(x)
        if len(S) == 48:
            break
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # Heuristic 2: Sort by conflict score descending (most constrained first?)
    if best_size < 48 and time.time() < deadline - 10:
        order_unsafe = sorted(range(1, 513), key=lambda x: -conflict_scores[x])
        S = []
        for x in order_unsafe:
            if time.time() > deadline - 10:
                break
            new_S = S + [x]
            if _is_valid_set(new_S):
                S.append(x)
            if len(S) == 48:
                break
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # Phase 2: Random restarts with repair
    # This is often better for finding large valid sets in complex constraint spaces
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Generate a random candidate set
        # Size 60-80 is a good sweet spot: large enough to have potential, small enough to repair quickly
        initial_size = random.randint(60, 80)
        candidates = random.sample(range(1, 513), initial_size)
        
        valid_set = _repair_set(candidates)
        
        if len(valid_set) > best_size:
            best_size = len(valid_set)
            best_set = valid_set
            
        if best_size == 48:
            break

    # Phase 3: Local search to improve the best set found
    if best_size < 48 and best_set:
        if time.time() < deadline - 2:
            best_set_set = set(best_set)
            # Prioritize adding elements with low conflict scores?
            # Or just random?
            # Let's sort removed elements by conflict score (ascending) to try "safe" ones first
            removed = [x for x in range(1, 513) if x not in best_set_set]
            removed.sort(key=lambda x: conflict_scores[x])
            
            # Limit the pool for local search to keep it fast
            # If we have many removed elements, just take the first 50
            removed_subset = removed[:50]
            
            improved = _local_search(best_set, removed_subset, max_iters=200)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved

    return best_set

# EVOLVE-BLOCK-END
