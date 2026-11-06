import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256
    
    # Precompute conflict tables: for each a, list of (b, c) pairs where a | (b+c), b <= c, a < b
    # This allows O(1) lookup of whether a conflict exists for a given mask
    conflict_pairs = {}
    for a in range(1, n + 1):
        pairs = []
        for b in range(a + 1, n + 1):
            k_min = (2 * b + a - 1) // a
            k_max = (n + b) // a
            for k in range(k_min, k_max + 1):
                c = k * a - b
                if b <= c <= n:
                    pairs.append((b, c))
        conflict_pairs[a] = pairs
    
    def is_valid(S):
        """
        Check if S is a valid set using precomputed conflict pairs.
        """
        if not S:
            return True
        
        # Build mask
        mask = 0
        for x in S:
            mask |= (1 << x)
        
        S_list = sorted(S)
        
        for i, a in enumerate(S_list):
            if a == 1:
                # 1 divides everything. Need at most one element > 1.
                count_gt_1 = sum(1 for x in S_list[i+1:])
                if count_gt_1 >= 2:
                    return False
                continue
            
            # Check all conflict pairs for this a
            for b, c in conflict_pairs[a]:
                if b in S and c in S:
                    return False
        
        return True
    
    def get_conflict(S, x):
        """
        If S U {x} is invalid, find one conflicting triple involving x.
        Returns (a, b, c) or None.
        """
        S_new = S | {x}
        
        S_list = sorted(S_new)
        
        for i, a in enumerate(S_list):
            if a == 1:
                if x == 1:
                    gt_1 = [y for y in S if y > 1]
                    if len(gt_1) >= 2:
                        return (1, gt_1[0], gt_1[1])
                else:
                    if 1 in S:
                        gt_1 = [y for y in S if y > 1]
                        if len(gt_1) >= 1:
                            return (1, gt_1[0], x)
                continue
            
            # Find a conflict involving x
            for b, c in conflict_pairs[a]:
                if b == x or c == x:
                    other = c if b == x else b
                    if other in S:
                        return (a, b, c)
                # Also check if both b and c are in S (x might be a, b, or c)
                if a == x or b == x or c == x:
                    if b in S_new and c in S_new:
                        return (a, b, c)
        
        return None
    
    # Step 1: Find best starting interval
    # Try multiple L values and pick the one that gives the largest valid set after greedy addition
    best_start = None
    best_start_size = 0
    
    for L in range(1, n + 1, 2):  # Step by 2 for speed
        if time.time() > deadline - 15:
            break
        S_test = set(range(L, n + 1))
        if is_valid(S_test):
            # Try to add some small elements
            S_greedy = set(S_test)
            for x in range(1, L):
                if time.time() > deadline - 15:
                    break
                if x in S_greedy:
                    continue
                test = S_greedy | {x}
                if is_valid(test):
                    S_greedy.add(x)
            
            if len(S_greedy) > best_start_size:
                best_start_size = len(S_greedy)
                best_start = S_greedy
    
    if best_start is None:
        # Fallback: just use [n, n]
        best_start = {n}
    
    best_set = best_start
    best_size = len(best_set)
    
    # Step 2: Multi-start greedy with swaps
    candidates = list(range(1, n + 1))
    random.shuffle(candidates)
    
    def greedy_with_swaps(S, candidates, time_limit):
        S = set(S)
        for x in candidates:
            if time.time() > time_limit:
                break
            if x in S:
                continue
            
            test = S | {x}
            if is_valid(test):
                S.add(x)
            else:
                # Try swap: remove one element from a conflict
                if time.time() < time_limit - 1:
                    conflict = get_conflict(S, x)
                    if conflict:
                        a, b, c = conflict
                        # Try removing each element from the conflict
                        for y in [a, b, c]:
                            if y == x or y not in S:
                                continue
                            test_swap = S | {x}
                            test_swap.remove(y)
                            if is_valid(test_swap):
                                S = test_swap
                                break
        return S
    
    # Run multiple attempts with different random orderings
    num_attempts = 30
    for attempt in range(num_attempts):
        if time.time() > deadline - 10:
            break
        
        random.shuffle(candidates)
        S = greedy_with_swaps(best_start, candidates, deadline - 10)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Step 3: Final local search on best_set
    current = set(best_set)
    remaining = [x for x in range(1, n + 1) if x not in current]
    random.shuffle(remaining)
    
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for x in remaining:
            if time.time() > deadline - 3:
                break
            if x in current:
                continue
            test = current | {x}
            if is_valid(test):
                current.add(x)
                improved = True
                best_size = len(current)
                best_set = current
            else:
                # Try swap
                conflict = get_conflict(current, x)
                if conflict:
                    a, b, c = conflict
                    for y in [a, b, c]:
                        if y == x or y not in current:
                            continue
                        test_swap = current | {x}
                        test_swap.remove(y)
                        if is_valid(test_swap):
                            current = test_swap
                            improved = True
                            best_size = len(current)
                            best_set = current
                            break

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
