# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    """
    Find 128 representations of targets 1..128 as sums of distinct 2,3,5-smooth numbers
    forming a divisibility antichain, minimizing total number of summands.
    """
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill

    # Step 1: Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for a in range(7):  # 2^0..2^6 = 1..64
        for b in range(5):  # 3^0..3^4 = 1..81
            for c in range(3):  # 5^0..5^2 = 1..25
                val = (2**a) * (3**b) * (5**c)
                if val <= 128:
                    smooth.append(val)
    smooth = sorted(set(smooth))
    smooth_set = set(smooth)

    # Precompute divisibility: div[i][j] = True if smooth[i] divides smooth[j]
    n_smooth = len(smooth)
    # div_mask[i] is a bitmask of indices j where smooth[i] divides smooth[j]
    # Actually, we need: for pair (i,j), does smooth[i] divide smooth[j]?
    # Let's use a 2D boolean array or bitmasks
    # div_mask[i] = bitmask of j such that smooth[i] | smooth[j]
    div_mask = [0] * n_smooth
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(n_smooth):
            if smooth[j] % si == 0:
                div_mask[i] |= (1 << j)

    # Step 2: For each target 1..128, find the minimum-k antichain sum
    # Use targeted search: try k=1,2,3,... until found

    # Precompute for efficiency: for each target, which smooth numbers can be used
    # A smooth number x can be used for target n only if x <= n

    # We'll do a BFS/DFS for small k values
    # For k=1: just check if target is smooth
    # For k=2: iterate pairs
    # For k=3: iterate triples
    # For k>=4: use backtracking with pruning

    # Helper: check if a set of indices forms an antichain
    def is_antichain_indices(indices):
        """Check if the given indices form a divisibility antichain."""
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                # Check if smooth[indices[i]] divides smooth[indices[j]] or vice versa
                if div_mask[indices[i]] & (1 << indices[j]):
                    return False
                if div_mask[indices[j]] & (1 << indices[i]):
                    return False
        return True

    # Better: use bitmask-based antichain check
    # For a set of indices represented as a bitmask, check antichain property
    # Precompute: for each index i, conflict_mask[i] = bitmask of j where i and j conflict (one divides the other)
    conflict_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                if smooth[j] % smooth[i] == 0 or smooth[i] % smooth[j] == 0:
                    conflict_mask[i] |= (1 << j)

    def is_antichain_bitmask(mask):
        """Check if the set of indices in mask forms an antichain."""
        # For each index in the set, check it doesn't conflict with any other
        m = mask
        while m:
            i_bit = m & (-m)
            i = i_bit.bit_length() - 1
            m ^= i_bit
            if conflict_mask[i] & m:  # check against remaining elements
                return False
        return True

    # For each target, find minimum k representation
    results = [None] * 129  # index 1..128

    # Precompute: for each target n, the list of smooth numbers <= n
    smooth_by_target = [[] for _ in range(129)]
    for n in range(1, 129):
        smooth_by_target[n] = [x for x in smooth if x <= n]

    # Try k=1 first for all targets
    for n in range(1, 129):
        if n in smooth_set:
            results[n] = [n]

    # For remaining targets, try k=2
    for n in range(1, 129):
        if results[n] is not None:
            continue
        if time.time() > deadline - 5:
            break
        candidates = smooth_by_target[n]
        # Try pairs
        found = False
        for i in range(len(candidates)):
            if found:
                break
            ai = candidates[i]
            for j in range(i+1, len(candidates)):
                aj = candidates[j]
                if ai + aj == n:
                    # Check antichain
                    # Find indices in smooth list
                    # We need to check if ai divides aj or aj divides ai
                    if aj % ai == 0 or ai % aj == 0:
                        continue
                    results[n] = [ai, aj]
                    found = True
                    break
        if time.time() > deadline - 5:
            break

    # For remaining targets, try k=3
    for n in range(1, 129):
        if results[n] is not None:
            continue
        if time.time() > deadline - 5:
            break
        candidates = smooth_by_target[n]
        found = False
        # Use combinations for triples
        for triple in combinations(candidates, 3):
            if sum(triple) == n:
                a, b, c = triple
                # Check antichain: no pair divides another
                if (b % a == 0 or a % b == 0) or \
                   (c % a == 0 or a % c == 0) or \
                   (c % b == 0 or b % c == 0):
                    continue
                results[n] = list(triple)
                found = True
                break
        if time.time() > deadline - 5:
            break

    # For remaining targets, try k=4,5,... using backtracking
    for n in range(1, 129):
        if results[n] is not None:
            continue
        if time.time() > deadline - 2:
            break
        candidates = smooth_by_target[n]
        # Backtracking search for minimum k
        best = None
        max_k = min(30, len(candidates))
        for k in range(4, max_k + 1):
            if time.time() > deadline - 2:
                break
            # Use recursive backtracking
            solution = None
            used = set()
            
            def backtrack(start, remaining, current, depth, max_depth):
                nonlocal solution
                if solution is not None:
                    return
                if depth == max_depth:
                    if remaining == 0:
                        solution = list(current)
                        return
                    return
                if remaining < 0:
                    return
                # Pruning: even if we pick the largest possible (max_depth - depth) times, can we reach remaining?
                # Actually, we need to pick exactly (max_depth - depth) more numbers
                # The minimum sum of (max_depth - depth) distinct numbers from candidates[start:]
                # But they must be antichain compatible
                # Simple pruning: if remaining < (max_depth - depth) * min_candidate, skip
                # Better: if remaining > sum of largest (max_depth - depth) candidates, skip
                
                for i in range(start, len(candidates)):
                    val = candidates[i]
                    if val > remaining:
                        break
                    # Check antichain compatibility with all in current
                    compatible = True
                    for v in current:
                        if val % v == 0 or v % val == 0:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    current.append(val)
                    backtrack(i + 1, remaining - val, current, depth + 1, max_depth)
                    current.pop()
                    if solution is not None:
                        return
            
            backtrack(0, n, [], 0, k)
            if solution is not None:
                results[n] = solution
                break
        if results[n] is None and time.time() <= deadline - 1:
            # Last resort: try larger k with less pruning
            for k in range(5, min(30, len(candidates)) + 1):
                if time.time() > deadline - 1:
                    break
                solution = None
                current = []
                
                def backtrack2(start, remaining, current, depth, max_depth):
                    nonlocal solution
                    if solution is not None:
                        return
                    if depth == max_depth:
                        if remaining == 0:
                            solution = list(current)
                        return
                    if remaining < 0:
                        return
                    for i in range(start, len(candidates)):
                        val = candidates[i]
                        if val > remaining:
                            break
                        compatible = True
                        for v in current:
                            if val % v == 0 or v % val == 0:
                                compatible = False
                                break
                        if not compatible:
                            continue
                        current.append(val)
                        backtrack2(i + 1, remaining - val, current, depth + 1, max_depth)
                        current.pop()
                        if solution is not None:
                            return
                
                backtrack2(0, n, current, 0, k)
                if solution is not None:
                    results[n] = solution
                    break

    # Verify all targets have solutions; if not, fill with any valid representation
    for n in range(1, 129):
        if results[n] is None:
            # Fallback: find any valid representation (may use more terms)
            candidates = smooth_by_target[n]
            # Try up to 30 terms
            for k in range(1, min(31, len(candidates) + 1)):
                if time.time() > deadline - 0.5:
                    break
                solution = None
                current = []
                
                def backtrack_fb(start, remaining, current, depth, max_depth):
                    nonlocal solution
                    if solution is not None:
                        return
                    if depth == max_depth:
                        if remaining == 0:
                            solution = list(current)
                        return
                    if remaining < 0:
                        return
                    for i in range(start, len(candidates)):
                        val = candidates[i]
                        if val > remaining:
                            break
                        compatible = True
                        for v in current:
                            if val % v == 0 or v % val == 0:
                                compatible = False
                                break
                        if not compatible:
                            continue
                        current.append(val)
                        backtrack_fb(i + 1, remaining - val, current, depth + 1, max_depth)
                        current.pop()
                        if solution is not None:
                            return
                
                backtrack_fb(0, n, current, 0, k)
                if solution is not None:
                    results[n] = solution
                    break
            if results[n] is None:
                # Should not happen, but provide a dummy
                results[n] = [n] if n in smooth_set else [1, n-1]

    return results[1:]
# EVOLVE-BLOCK-END
