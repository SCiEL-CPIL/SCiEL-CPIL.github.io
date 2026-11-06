"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 256

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 27.0
    best_size = N + 1
    best_set = None
    
    def compute_sums(S):
        """Compute all pair sums for a Sidon set."""
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums
    
    def is_maximal_fast(S, sums):
        """Check if S is maximal: no element from [1,N] can be added."""
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if 2 * x in sums:
                continue
            can_add = True
            for a in S:
                if (x + a) in sums:
                    can_add = False
                    break
            if can_add:
                return False
        return True
    
    def count_blocked_fast(S, sums, x):
        """Count how many remaining elements would be blocked if x is added."""
        S_set = set(S)
        new_sums = sums | {2 * x}
        for a in S:
            new_sums.add(x + a)
        
        blocked = 0
        for y in range(1, N + 1):
            if y == x or y in S_set:
                continue
            if 2 * y in new_sums:
                blocked += 1
                continue
            y_blocked = False
            for a in S:
                if (y + a) in new_sums:
                    y_blocked = True
                    break
            if y_blocked:
                blocked += 1
        return blocked
    
    def local_search_remove(S, sums):
        """Try to shrink a maximal Sidon set by removing elements and re-maximizing."""
        nonlocal best_size, best_set
        
        S = list(S)
        sums = set(sums)
        S_set = set(S)
        
        # Try removing each element and see if we can get a smaller maximal set
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline - 2.0:
                break
            
            x = S[i]
            # Try removing x
            S_new = S[:i] + S[i+1:]
            sums_new = compute_sums(S_new)
            
            # Now try to re-maximize from this smaller set
            S_new_set = set(S_new)
            changed = True
            while changed:
                changed = False
                for y in range(1, N + 1):
                    if y in S_new_set:
                        continue
                    if 2 * y in sums_new:
                        continue
                    can_add = True
                    for a in S_new:
                        if (y + a) in sums_new:
                            can_add = False
                            break
                    if can_add:
                        S_new.append(y)
                        S_new_set.add(y)
                        sums_new.add(2 * y)
                        for a in S_new[:-1]:
                            sums_new.add(y + a)
                        changed = True
                        break
            
            if len(S_new) < len(S):
                S = S_new
                sums = sums_new
                S_set = set(S)
                if len(S) < best_size:
                    best_size = len(S)
                    best_set = list(S)
        
        return S
    
    def backtrack_search(S, sums, depth):
        """Backtracking search to find minimal maximal Sidon set."""
        nonlocal best_size, best_set
        
        if time.time() > deadline - 1.5:
            return False
        
        # Check if we've found a better solution
        if len(S) >= best_size:
            return False
        
        # Check if all remaining elements are blocked (maximal)
        if is_maximal_fast(S, sums):
            # We have a maximal Sidon set
            if len(S) < best_size:
                best_size = len(S)
                best_set = list(S)
                # Try local search to shrink further
                local_search_remove(S, sums)
                return True
            return False
        
        # Find all addable elements
        S_set = set(S)
        candidates = []
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if 2 * x in sums:
                continue
            can_add = True
            for a in S:
                if (x + a) in sums:
                    can_add = False
                    break
            if can_add:
                # Count how many elements this would block
                blocked = count_blocked_fast(S, sums, x)
                candidates.append((blocked, x))
        
        if not candidates:
            return False
        
        # Sort by blocking power (descending)
        candidates.sort(reverse=True)
        
        # Limit branching factor for performance
        max_branch = min(len(candidates), 6)
        
        for blocked, x in candidates[:max_branch]:
            # Add x to S
            S.append(x)
            old_sums = set(sums)
            sums.add(2 * x)
            for a in S[:-1]:
                sums.add(x + a)
            
            # Recurse
            if backtrack_search(S, sums, depth + 1):
                S.pop()
                sums.clear()
                sums.update(old_sums)
                return True
            
            S.pop()
            sums.clear()
            sums.update(old_sums)
        
        return False
    
    def greedy_initial(strategy='blocking'):
        """Greedy construction with different strategies."""
        S = []
        S_set = set()
        sums = set()
        remaining = list(range(1, N + 1))
        
        while remaining:
            best_x = None
            best_score = -1
            for x in remaining:
                if 2 * x in sums:
                    continue
                can_add = True
                for a in S:
                    if (x + a) in sums:
                        can_add = False
                        break
                if not can_add:
                    continue
                
                if strategy == 'blocking':
                    # Count how many elements this would block
                    new_sums = sums | {2 * x}
                    for a in S:
                        new_sums.add(x + a)
                    
                    blocked = 0
                    for y in remaining:
                        if y == x:
                            continue
                        if 2 * y in new_sums:
                            blocked += 1
                            continue
                        y_blocked = False
                        for a in S:
                            if (y + a) in new_sums:
                                y_blocked = True
                                break
                        if y_blocked:
                            blocked += 1
                    score = blocked
                elif strategy == 'efficiency':
                    # Count blocked / (1 + len(S)) to favor early high-impact picks
                    new_sums = sums | {2 * x}
                    for a in S:
                        new_sums.add(x + a)
                    
                    blocked = 0
                    for y in remaining:
                        if y == x:
                            continue
                        if 2 * y in new_sums:
                            blocked += 1
                            continue
                        y_blocked = False
                        for a in S:
                            if (y + a) in new_sums:
                                y_blocked = True
                                break
                        if y_blocked:
                            blocked += 1
                    score = blocked / (1 + len(S))
                else:
                    score = x  # ascending
                
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
            
            S.append(best_x)
            S_set.add(best_x)
            sums.add(2 * best_x)
            for a in S[:-1]:
                sums.add(best_x + a)
            remaining.remove(best_x)
        
        return S
    
    # Try multiple greedy strategies
    for strategy in ['blocking', 'efficiency']:
        if time.time() > deadline - 5.0:
            break
        initial_set = greedy_initial(strategy)
        if initial_set:
            if len(initial_set) < best_size:
                best_size = len(initial_set)
                best_set = list(initial_set)
                # Try local search
                local_search_remove(initial_set, compute_sums(initial_set))
    
    # Run backtracking search with different random seeds
    for seed in range(15):
        if time.time() > deadline - 4.0:
            break
        
        random.seed(seed * 137 + 42)
        
        S = []
        sums = set()
        
        # Add a few random elements first to break symmetry
        num_initial = random.randint(2, 5)
        for _ in range(num_initial):
            candidates = [x for x in range(1, N + 1) if x not in S]
            if not candidates:
                break
            x = random.choice(candidates)
            
            if 2 * x in sums:
                continue
            can_add = True
            for a in S:
                if (x + a) in sums:
                    can_add = False
                    break
            if not can_add:
                continue
            
            S.append(x)
            sums.add(2 * x)
            for a in S[:-1]:
                sums.add(x + a)
        
        backtrack_search(S, sums, 0)
    
    if best_set is not None:
        return best_set
    else:
        return greedy_initial('blocking')
# EVOLVE-BLOCK-END
