import random
import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 28  # 2s margin
    
    best_size = float('inf')
    best_set = []
    
    def is_valid_sidon(A, S):
        """Check if A is a valid Sidon set with sum set S."""
        if len(A) != len(set(A)):
            return False
        # Verify S matches all pair sums
        expected_sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                expected_sums.add(A[i] + A[j])
        return S == expected_sums
    
    def compute_blocked(A, S):
        """Return set of elements that cannot be added to A."""
        blocked = set(A)
        for x in range(1, N + 1):
            if x in A:
                continue
            # Check if x can be added
            if 2 * x in S:
                blocked.add(x)
                continue
            ok = True
            for a in A:
                if x + a in S:
                    blocked.add(x)
                    ok = False
                    break
        return blocked
    
    def backtrack_search(A, S, max_depth, deadline, best_size, best_set_ref):
        """Randomized backtracking search with forward checking."""
        if time.time() > deadline:
            return False
        
        if len(A) >= best_size:
            return False
        
        # Find addable elements
        addable = []
        for x in range(1, N + 1):
            if x in A:
                continue
            if 2 * x in S:
                continue
            ok = True
            for a in A:
                if x + a in S:
                    ok = False
                    break
            if ok:
                addable.append(x)
        
        if not addable:
            # Maximal set found
            if len(A) < best_size:
                best_size[0] = len(A)
                best_set_ref[0] = list(A)
                return True
            return False
        
        # Heuristic: try elements that block the most others first
        # Score each candidate by how many other addable elements it blocks
        scored_candidates = []
        for x in addable:
            # Simulate adding x
            new_S = set(S)
            new_S.add(2 * x)
            for a in A:
                new_S.add(x + a)
            
            # Count how many addable elements become blocked
            blocked_count = 0
            for y in addable:
                if y == x:
                    continue
                if 2 * y in new_S:
                    blocked_count += 1
                    continue
                y_blocked = False
                for a in A:
                    if y + a in new_S:
                        y_blocked = True
                        break
                if y + x in new_S:
                    y_blocked = True
                if y_blocked:
                    blocked_count += 1
            
            scored_candidates.append((blocked_count, x))
        
        # Sort by blocked count (descending) with some randomization
        random.shuffle(scored_candidates)
        scored_candidates.sort(key=lambda t: -t[0])
        
        # Try top candidates with some randomization
        num_to_try = min(5, len(scored_candidates))
        for i in range(num_to_try):
            if time.time() > deadline:
                return False
            
            x = scored_candidates[i][1]
            new_A = A + [x]
            new_S = set(S)
            new_S.add(2 * x)
            for a in A:
                new_S.add(x + a)
            
            if backtrack_search(new_A, new_S, max_depth, deadline, best_size, best_set_ref):
                return True
        
        return False
    
    # Multiple restarts with different random seeds
    for seed in range(500):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        
        # Start with empty set
        A = []
        S = set()
        
        best_size_ref = [best_size]
        best_set_ref = [best_set]
        
        # Randomized backtracking
        backtrack_search(A, S, 12, deadline, best_size_ref, best_set_ref)
        
        best_size = best_size_ref[0]
        best_set = best_set_ref[0]
        
        # If we found a very small set, we can stop early
        if best_size <= 8:
            break
    
    return best_set if best_set else [1]
