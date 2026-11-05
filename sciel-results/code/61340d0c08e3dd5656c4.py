"""B_3[1] set construction using bytearray O(1) lookups and recursive backtracking.

Strategy:
  1. Use a bytearray (size 12289) for O(1) triple-sum conflict lookups.
  2. Recursive backtracking that explores candidates in ascending order (greedy path).
  3. Multiple randomized restarts with different candidate orderings.
  4. Local search: swap-based improvement to escape greedy local optima.
"""
import time
import random
import sys

def _is_valid_triples(S):
    """Check if all unordered triple sums (with replacement) are unique."""
    n = len(S)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = S[i] + S[j] + S[k]
                if s in seen:
                    return False
                seen.add(s)
    return True

def _triple_sums_involving(x, S):
    """Generate all triple sums that involve x (with elements from S)."""
    sums = []
    n = len(S)
    # (x, x, x)
    sums.append(3 * x)
    # (x, x, y) and (x, y, y) for y in S
    for y in S:
        sums.append(2 * x + y)
        sums.append(x + 2 * y)
    # (x, y, z) for y <= z in S
    for i in range(n):
        for j in range(i, n):
            sums.append(x + S[i] + S[j])
    return sums

def _backtrack_construct(candidates, deadline, max_size=80):
    """
    Recursive backtracking to construct a B_3[1] set.
    candidates: list of candidate values to try (in order).
    Returns the constructed set.
    """
    S = []
    # bytearray for O(1) sum lookups: max sum = 3*4096 = 12288
    sum_flags = bytearray(12289)
    
    def backtrack(idx, current_size):
        nonlocal S, sum_flags
        if time.time() > deadline:
            return
        if current_size >= max_size:
            return
        if idx >= len(candidates):
            return
        
        x = candidates[idx]
        
        # Check if x can be added
        new_sums = _triple_sums_involving(x, S)
        can_add = True
        for s in new_sums:
            if sum_flags[s]:
                can_add = False
                break
        
        if can_add:
            # Add x
            S.append(x)
            for s in new_sums:
                sum_flags[s] = 1
            
            # Recurse
            backtrack(idx + 1, current_size + 1)
            
            # Backtrack
            S.pop()
            for s in new_sums:
                sum_flags[s] = 0
        
        # Try next candidate
        backtrack(idx + 1, current_size)
    
    backtrack(0, 0)
    return S

def _greedy_construct(candidates, deadline):
    """
    Greedy construction using bytearray for fast conflict checking.
    candidates: list of candidate values to try (in order).
    Returns the constructed set.
    """
    S = []
    sum_flags = bytearray(12289)
    
    for x in candidates:
        if time.time() > deadline:
            break
        
        # Check if x can be added
        new_sums = _triple_sums_involving(x, S)
        can_add = True
        for s in new_sums:
            if sum_flags[s]:
                can_add = False
                break
        
        if can_add:
            S.append(x)
            for s in new_sums:
                sum_flags[s] = 1
    
    return S

def _local_search(S, deadline, rng, max_iters=1000):
    """
    Local search: try to improve S by swapping elements.
    Returns improved S or the same if no improvement.
    """
    S = list(S)
    in_S = set(S)
    sum_flags = bytearray(12289)
    
    # Rebuild sum_flags
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sum_flags[S[i] + S[j] + S[k]] = 1
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try to add a random element not in S
        candidates = [x for x in range(1, 4097) if x not in in_S]
        if not candidates:
            break
        
        x_add = rng.choice(candidates)
        new_sums = _triple_sums_involving(x_add, S)
        can_add = True
        for s in new_sums:
            if sum_flags[s]:
                can_add = False
                break
        
        if can_add:
            S.append(x_add)
            for s in new_sums:
                sum_flags[s] = 1
            in_S.add(x_add)
        else:
            # Try a swap: remove a random element, try to add x_add
            if len(S) > 1:
                idx_remove = rng.randint(0, len(S) - 1)
                x_remove = S[idx_remove]
                
                # Remove x_remove: rebuild sum_flags
                S.pop(idx_remove)
                in_S.discard(x_remove)
                sum_flags = bytearray(12289)
                n = len(S)
                for i in range(n):
                    for j in range(i, n):
                        for k in range(j, n):
                            sum_flags[S[i] + S[j] + S[k]] = 1
                
                new_sums = _triple_sums_involving(x_add, S)
                can_add = True
                for s in new_sums:
                    if sum_flags[s]:
                        can_add = False
                        break
                
                if can_add:
                    S.append(x_add)
                    for s in new_sums:
                        sum_flags[s] = 1
                    in_S.add(x_add)
                else:
                    # Restore x_remove
                    S.append(x_remove)
                    in_S.add(x_remove)
                    sum_flags = bytearray(12289)
                    n = len(S)
                    for i in range(n):
                        for j in range(i, n):
                            for k in range(j, n):
                                sum_flags[S[i] + S[j] + S[k]] = 1
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (90s limit)
    rng = random.Random(42)
    
    best_S = []
    
    # Phase 1: Greedy with ascending order (fast baseline)
    S = _greedy_construct(list(range(1, 4097)), deadline)
    if len(S) > len(best_S):
        best_S = S
    
    # Phase 2: Backtracking with ascending order (explores greedy path first)
    if time.time() < deadline:
        S = _backtrack_construct(list(range(1, 4097)), deadline, max_size=len(best_S) + 5)
        if len(S) > len(best_S):
            best_S = S
    
    # Phase 3: Local search on best_S
    if time.time() < deadline:
        S = _local_search(best_S, deadline, rng, max_iters=500)
        if len(S) > len(best_S):
            best_S = S
    
    # Phase 4: Randomized restarts with greedy + local search
    no_improve = 0
    while time.time() < deadline and no_improve < 20:
        seed = rng.randint(0, 10**9)
        rng2 = random.Random(seed)
        
        # Random candidate ordering
        candidates = list(range(1, 4097))
        rng2.shuffle(candidates)
        S = _greedy_construct(candidates, deadline)
        
        # Local search
        if time.time() < deadline:
            S = _local_search(S, deadline, rng2, max_iters=200)
        
        if len(S) > len(best_S):
            best_S = S
            no_improve = 0
        else:
            no_improve += 1
    
    # Phase 5: Backtracking with random orderings
    no_improve = 0
    while time.time() < deadline and no_improve < 10:
        seed = rng.randint(0, 10**9)
        rng2 = random.Random(seed)
        
        candidates = list(range(1, 4097))
        rng2.shuffle(candidates)
        S = _backtrack_construct(candidates, deadline, max_size=len(best_S) + 5)
        
        if len(S) > len(best_S):
            best_S = S
            no_improve = 0
        else:
            no_improve += 1
    
    # Final verification
    if not _is_valid_triples(best_S):
        # Fallback: rebuild from scratch with a simple greedy
        best_S = _greedy_construct(list(range(1, 4097)), deadline)
    
    return best_S

if __name__ == "__main__":
    s = time.time()
    res = construct()
    print(f"Time: {time.time()-s:.2f}s, Size: {len(res)}")
    if _is_valid_triples(res):
        print("Valid")
    else:
        print("Invalid")
