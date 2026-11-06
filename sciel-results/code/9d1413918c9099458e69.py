import time
import random
from fractions import Fraction

DEADLINE = 55.0  # Safety margin for 60s limit
_START = None

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE

def _can_form_target(target, vals):
    """
    Check if target can be formed as a subset sum of vals (nonempty subset).
    vals is a list of Fractions.
    Returns True if found, False otherwise.
    """
    if not vals:
        return False
    
    # Filter out values greater than target? No, we can skip them in DFS.
    # Sort descending for better pruning
    vals = sorted([v for v in vals if v <= target], reverse=True)
    if not vals:
        return False
        
    n = len(vals)
    total = sum(vals)
    if total < target:
        return False
    
    # Precompute suffix sums for pruning
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + vals[i]
        
    # DFS with pruning
    def dfs(idx, remaining):
        if remaining == 0:
            return True
        if idx >= n:
            return False
        if remaining < 0:
            return False
        if suffix_sums[idx] < remaining:
            return False
        
        # If the current value is exactly the remaining, we found it
        if vals[idx] == remaining:
            return True
            
        # If the current value is greater than remaining, we must skip it
        if vals[idx] > remaining:
            return dfs(idx + 1, remaining)
            
        # Option 1: Take vals[idx]
        if dfs(idx + 1, remaining - vals[idx]):
            return True
            
        # Option 2: Skip vals[idx]
        if dfs(idx + 1, remaining):
            return True
            
        return False

    return dfs(0, target)

def get_violation(S):
    """
    Returns (x, subset_indices) if a violation exists, else (None, None).
    x is the element whose reciprocal is formed.
    subset_indices are indices in S of the elements forming the sum.
    """
    S_set = set(S)
    n = len(S)
    for i, x in enumerate(S):
        target = Fraction(1, x)
        others = [Fraction(1, y) for j, y in enumerate(S) if j != i]
        # We need to map back to indices if found, but for now just check existence
        # To be efficient, we can just check existence.
        if _can_form_target(target, others):
            # We found a violation. To fix it, we need to know which elements were involved.
            # For the removal strategy, we just need to know x and that SOME subset of others works.
            # We don't strictly need the subset to remove, we can just pick one from the "conflict group".
            # However, removing x is always a valid fix. Removing any y in the subset is also a valid fix.
            # Let's return x and a dummy subset marker for now, or better, re-run to find subset?
            # No, let's just return x. The caller can decide to remove x or try to find a better candidate.
            # Actually, to be smart, we should find the subset.
            # But _can_form_target doesn't return the subset.
            # Let's modify _can_form_target to return the subset or just use a heuristic.
            # For simplicity, let's assume we remove x or a random other element.
            # Better: return x and the list of others. The caller can pick one.
            return x, others
    return None, None

def is_valid(S):
    """Check if set S is reciprocal-subsum-free."""
    if not S:
        return True
    S_set = set(S)
    for x in S:
        target = Fraction(1, x)
        others = [Fraction(1, y) for y in S_set if y != x]
        if _can_form_target(target, others):
            return False
    return True

def construct():
    global _START
    _START = time.time()
    
    elements = list(range(1, 25))
    best = []
    best_size = 0
    
    # Strategy 1: Greedy Removal from Full Set
    # Start with all elements, remove those causing conflicts.
    S = set(elements)
    while not is_valid(S):
        if _deadline_hit():
            break
        x, others = get_violation(S)
        if x is None:
            break # Should not happen if is_valid said false, but safe
        
        # Decide what to remove.
        # Candidates: x or any y in the subset that formed the sum.
        # Since we don't have the exact subset, let's try removing x first.
        # If that leads to a small set, we might try removing a "large" element from others later.
        # For now, remove x.
        S.remove(x)
        
    if len(S) > best_size:
        best = list(S)
        best_size = len(S)
        
    # Strategy 2: Local Search to Add Back Elements
    # Try to add elements back to S to increase size.
    S = set(best)
    improved = True
    while improved and not _deadline_hit():
        improved = False
        # Sort elements to try in a specific order (e.g., largest first, or random)
        missing = [x for x in elements if x not in S]
        # Shuffle to avoid bias
        random.shuffle(missing)
        for x in missing:
            if _deadline_hit():
                break
            S_test = S | {x}
            if is_valid(S_test):
                S = S_test
                if len(S) > best_size:
                    best = list(S)
                    best_size = len(S)
                improved = True
                break # Restart loop with new S
                
    # Strategy 3: Randomized Swaps
    # If we are stuck, try swapping elements to find a different configuration that allows more additions.
    random.seed(42)
    S = set(best)
    for _ in range(500):
        if _deadline_hit():
            break
        if len(S) == 0:
            S = set(elements[:10])
            continue
            
        # Try to swap a random element in S with a random element not in S
        if len(S) < 24:
            in_S = list(S)
            not_in_S = [x for x in elements if x not in S]
            if not_in_S:
                a = random.choice(in_S)
                b = random.choice(not_in_S)
                S_test = (S - {a}) | {b}
                if is_valid(S_test):
                    S = S_test
                    if len(S) > best_size:
                        best = list(S)
                        best_size = len(S)
                        
        # Also try just adding if possible (in case previous loop missed it due to order)
        if len(S) < 24:
            not_in_S = [x for x in elements if x not in S]
            if not_in_S:
                # Try a random few
                for _ in range(5):
                    if not_in_S:
                        x = random.choice(not_in_S)
                        S_test = S | {x}
                        if is_valid(S_test):
                            S = S_test
                            if len(S) > best_size:
                                best = list(S)
                                best_size = len(S)
                            not_in_S.remove(x)
                            break

    # Final Polish: Try to add any remaining
    if not _deadline_hit():
        S = set(best)
        improved = True
        while improved and not _deadline_hit():
            improved = False
            missing = [x for x in elements if x not in S]
            for x in missing:
                S_test = S | {x}
                if is_valid(S_test):
                    S = S_test
                    if len(S) > best_size:
                        best = list(S)
                        best_size = len(S)
                    improved = True
                    break
                    
    return best

if __name__ == "__main__":
    print(construct())
