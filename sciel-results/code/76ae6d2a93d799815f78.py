"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256
import random
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 57  # 57s safety margin for 60s limit
    start_time = time.time()
    
    def is_sidon_valid(s, sums):
        """Check if adding elements to s keeps it Sidon (already assumed valid, just verify)."""
        s_set = set(s)
        for i, a in enumerate(s):
            for j in range(i, len(s)):
                b = s[j]
                s_val = a + b
                # Count how many pairs produce this sum
                count = 0
                for k, c in enumerate(s):
                    for l in range(k, len(s)):
                        d = s[l]
                        if c + d == s_val:
                            count += 1
                if count > 1:
                    return False
        return True
    
    def compute_sums(s):
        """Compute all unordered pair sums a+b where a<=b."""
        sums = set()
        for i, a in enumerate(s):
            sums.add(2 * a)
            for j in range(i + 1, len(s)):
                sums.add(a + s[j])
        return sums
    
    def can_add(x, s, sums):
        """Check if x can be added to s without violating Sidon property."""
        if 2 * x in sums:
            return False
        for a in s:
            if x + a in sums:
                return False
        return True
    
    def get_new_sums(x, s):
        """Get the new sums that would be added if x is added to s."""
        new_sums = {2 * x}
        for a in s:
            new_sums.add(x + a)
        return new_sums
    
    def is_blocked(x, s, sums):
        """Check if x is blocked (cannot be added to s)."""
        if 2 * x in sums:
            return True
        for a in s:
            if x + a in sums:
                return True
        return False
    
    def count_unblocked(s, sums):
        """Count how many elements in [1,N] are unblocked."""
        s_set = set(s)
        count = 0
        for x in range(1, N + 1):
            if x not in s_set and not is_blocked(x, s, sums):
                count += 1
        return count
    
    def get_unblocked_list(s, sums):
        """Get list of unblocked elements not in s."""
        s_set = set(s)
        result = []
        for x in range(1, N + 1):
            if x not in s_set and not is_blocked(x, s, sums):
                result.append(x)
        return result
    
    def blocking_score(x, s, sums, unblocked):
        """Score how many unblocked elements would be blocked if x is added."""
        new_sums = get_new_sums(x, s)
        score = 0
        for y in unblocked:
            if y == x:
                continue
            # Check if y would be blocked by s ∪ {x}
            if 2 * y in sums or 2 * y in new_sums:
                score += 1
            else:
                blocked = False
                for a in s:
                    if y + a in sums:
                        blocked = True
                        break
                if not blocked:
                    if y + x in sums or y + x in new_sums:
                        blocked = True
                if blocked:
                    score += 1
        return score
    
    def make_maximal(s, sums):
        """Make s maximal by greedily adding elements that block the most unblocked elements."""
        s = list(s)
        sums = set(sums)
        
        while True:
            unblocked = get_unblocked_list(s, sums)
            if not unblocked:
                break
            
            # Pick element with highest blocking score
            best_x = None
            best_score = -1
            for x in unblocked:
                sc = blocking_score(x, s, sums, unblocked)
                if sc > best_score:
                    best_score = sc
                    best_x = x
            
            if best_x is None or best_score == 0:
                # Try to add any unblocked element
                if unblocked:
                    best_x = unblocked[0]
                else:
                    break
            
            # Verify we can add best_x
            new_sums = get_new_sums(best_x, s)
            if sums.isdisjoint(new_sums):
                s.append(best_x)
                sums.update(new_sums)
            else:
                break
        
        return s, sums
    
    def verify_maximal(s):
        """Verify s is maximal: no missing element can be added."""
        s_set = set(s)
        sums = compute_sums(s)
        for x in range(1, N + 1):
            if x not in s_set:
                if not is_blocked(x, s, sums):
                    return False
        return True
    
    def local_search_shrink(s):
        """Try to remove elements while maintaining maximality (after re-maximizing)."""
        s = list(s)
        best = list(s)
        best_size = len(s)
        
        for _ in range(20):  # Limited iterations
            improved = False
            for i in range(len(s) - 1, -1, -1):
                if len(s) - 1 >= best_size:
                    continue
                # Try removing s[i]
                s_trial = s[:i] + s[i+1:]
                if len(s_trial) < 3:
                    continue
                sums_trial = compute_sums(s_trial)
                # Check if still Sidon (should be, since subset of Sidon set)
                # Make maximal
                s_max, sums_max = make_maximal(s_trial, sums_trial)
                if verify_maximal(s_max):
                    if len(s_max) < best_size:
                        best = list(s_max)
                        best_size = len(s_max)
                        s = list(s_max)
                        improved = True
                        break
            if not improved:
                break
        
        return best
    
    best_result = None
    best_size = float('inf')
    
    # Strategy 1: Greedy blocking-priority search with multiple random seeds
    for trial in range(200):
        if time.time() > deadline:
            break
        
        random.seed(trial * 7919 + 13)
        
        # Generate initial Sidon set
        target_size = random.randint(6, 14)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        s = []
        sums = set()
        for x in candidates:
            if len(s) >= target_size:
                break
            if can_add(x, s, sums):
                s.append(x)
                sums.update(get_new_sums(x, s))
        
        if len(s) < 3:
            continue
        
        # Make maximal
        s_max, sums_max = make_maximal(s, sums)
        
        if verify_maximal(s_max):
            # Try local search to shrink
            s_shrunk = local_search_shrink(s_max)
            if verify_maximal(s_shrunk):
                if len(s_shrunk) < best_size:
                    best_size = len(s_shrunk)
                    best_result = list(s_shrunk)
                    if best_size <= 10:
                        break
            else:
                if len(s_max) < best_size:
                    best_size = len(s_max)
                    best_result = list(s_max)
                    if best_size <= 10:
                        break
    
    # Strategy 2: Backtracking with branch-and-bound
    if time.time() < deadline - 5:
        def backtrack_search(s, sums, depth, max_depth):
            """Backtracking search for small maximal Sidon sets."""
            nonlocal best_result, best_size
            
            if time.time() > deadline:
                return
            
            if len(s) >= best_size:
                return
            
            if depth >= max_depth:
                # Try to make maximal and check
                s_max, sums_max = make_maximal(s, sums)
                if verify_maximal(s_max) and len(s_max) < best_size:
                    best_size = len(s_max)
                    best_result = list(s_max)
                return
            
            # Get unblocked elements
            unblocked = get_unblocked_list(s, sums)
            if not unblocked:
                # Already maximal
                if len(s) < best_size:
                    best_size = len(s)
                    best_result = list(s)
                return
            
            # Sort by blocking score (descending)
            scored = []
            for x in unblocked:
                sc = blocking_score(x, s, sums, unblocked)
                scored.append((sc, x))
            scored.sort(reverse=True)
            
            # Try top candidates
            for sc, x in scored[:min(8, len(scored))]:
                if time.time() > deadline:
                    return
                new_sums = get_new_sums(x, s)
                if sums.isdisjoint(new_sums):
                    new_sums_all = sums | new_sums
                    backtrack_search(s + [x], new_sums_all, depth + 1, max_depth)
            
            # Also try not adding any more (make maximal from current)
            if depth > 0:
                s_max, sums_max = make_maximal(s, sums)
                if verify_maximal(s_max) and len(s_max) < best_size:
                    best_size = len(s_max)
                    best_result = list(s_max)
        
        # Run a few backtracking searches with different initial seeds
        for bt_trial in range(20):
            if time.time() > deadline - 5:
                break
            
            random.seed(bt_trial * 104729 + 37)
            # Start with a small random seed
            seed_size = random.randint(2, 5)
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            
            s_init = []
            sums_init = set()
            for x in candidates:
                if len(s_init) >= seed_size:
                    break
                if can_add(x, s_init, sums_init):
                    s_init.append(x)
                    sums_init.update(get_new_sums(x, s_init))
            
            if len(s_init) < 2:
                continue
            
            backtrack_search(s_init, sums_init, 0, 12)
    
    # Fallback
    if best_result is None:
        selected, sums = [], set()
        for x in range(1, N + 1):
            if can_add(x, selected, sums):
                selected.append(x)
                sums.update(get_new_sums(x, selected))
        best_result = selected
    
    return best_result
# EVOLVE-BLOCK-END
