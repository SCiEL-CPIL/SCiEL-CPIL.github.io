import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # We need a Sidon set containing 0, with all positive differences distinct,
    # and every d in 1..30 occurring as a difference.
    # Minimize the largest element.
    
    # A Sidon set (B_2 sequence) has all pairwise sums a_i + a_j (i <= j) distinct,
    # equivalently all positive differences a_j - a_i (i < j) are distinct.
    # Wait: "all positive differences distinct" means if a_j - a_i = a_l - a_k with (i,j) != (k,l), that's not allowed.
    # This is exactly the Sidon property.
    
    # We need to find a set S = {0, x1, x2, ..., xk} with k+1 <= 32 elements,
    # such that:
    # 1. All elements are in 0..10^6
    # 2. All positive differences are distinct (Sidon)
    # 3. For every d in 1..30, there exist a, b in S with b - a = d
    # 4. Minimize max(S)
    
    # Strategy: Use a greedy/backtracking approach with bitsets for efficiency.
    # Key insight: We can model this as finding a set that covers all differences 1..30
    # while maintaining the Sidon property.
    
    # Let's use a randomized greedy approach with local search.
    # Represent the set as a sorted list. Use a set of differences for quick lookup.
    
    def compute_diffs(S):
        """Compute all positive differences of sorted set S."""
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(S[j] - S[i])
        return diffs
    
    def is_valid_sidon(S):
        """Check if S is a valid Sidon set (all positive differences distinct)."""
        if len(S) != len(set(S)):
            return False
        S = sorted(S)
        diffs = []
        n = len(S)
        for i in range(n):
            for j in range(i+1, n):
                diffs.append(S[j] - S[i])
        return len(diffs) == len(set(diffs))
    
    def covers_all_diffs(S, max_d=30):
        """Check if all differences 1..max_d are covered."""
        diffs = compute_diffs(S)
        return all(d in diffs for d in range(1, max_d + 1))
    
    # Greedy construction: start with {0}, add elements to cover missing differences
    # while maintaining Sidon property.
    
    def greedy_construct():
        S = [0]
        covered = set()
        
        for d in range(1, 31):
            if d in covered:
                continue
            
            # Try to find a small x such that S ∪ {x} is Sidon and covers d
            # x could be d (if 0 is in S) or some other value
            # Actually, to cover difference d, we need two elements a, b in S ∪ {x}
            # with b - a = d.
            
            # Try adding x = d (uses 0 and d)
            # Or try adding x such that x - s = d for some s in S, i.e., x = s + d
            # Or x + d = s for some s in S, i.e., x = s - d (if s > d)
            
            candidates = set()
            for s in S:
                candidates.add(s + d)  # new element s+d, difference with s is d
                if s > d:
                    candidates.add(s - d)  # new element s-d, difference with s is d
            
            # Also try x = d directly (difference from 0)
            candidates.add(d)
            
            # Sort candidates by value (prefer smaller to minimize diameter)
            candidates = sorted(candidates)
            
            placed = False
            for x in candidates:
                if x < 0 or x > 10**6:
                    continue
                if x in S:
                    continue
                
                new_S = S + [x]
                if is_valid_sidon(new_S):
                    S = new_S
                    # Update covered differences
                    new_diffs = compute_diffs(S)
                    covered = new_diffs
                    placed = True
                    break
            
            if not placed:
                # Fallback: try larger values
                x = max(S) + d + 1
                while x < 10**6:
                    new_S = S + [x]
                    if is_valid_sidon(new_S):
                        S = new_S
                        covered = compute_diffs(S)
                        break
                    x += 1
                else:
                    return None
        
        return S
    
    # Try greedy first
    best_S = greedy_construct()
    if best_S is not None:
        best_max = max(best_S)
    else:
        best_max = 10**6
    
    # Now try to improve using local search / simulated annealing
    # Objective: minimize max(S) subject to Sidon + covering 1..30
    
    if time.time() > deadline - 10:
        return sorted(best_S) if best_S else [0]
    
    # Local search: try to reduce the maximum element
    # Replace the largest element with a smaller one if possible
    
    def try_improve(S, time_limit):
        """Try to improve S by replacing elements."""
        S = sorted(S)
        best = S
        best_max = max(S)
        t0 = time.time()
        
        # Try replacing each element (except 0) with a smaller value
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            S = sorted(best)
            n = len(S)
            
            for i in range(1, n):  # skip 0
                old_val = S[i]
                # Try replacing with smaller values
                for new_val in range(old_val - 1, 0, -1):
                    if new_val in S:
                        continue
                    
                    new_S = S[:i] + [new_val] + S[i+1:]
                    new_S = sorted(new_S)
                    
                    if not is_valid_sidon(new_S):
                        continue
                    
                    if not covers_all_diffs(new_S):
                        continue
                    
                    if max(new_S) < best_max:
                        best = new_S
                        best_max = max(new_S)
                        improved = True
                        break
        
        return best
    
    # Also try a more aggressive search: random restarts with greedy
    if time.time() < deadline - 5:
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            # Randomized greedy: shuffle the order of covering differences
            S = [0]
            covered = set()
            order = list(range(1, 31))
            random.shuffle(order)
            
            success = True
            for d in order:
                if d in covered:
                    continue
                
                candidates = set()
                for s in S:
                    candidates.add(s + d)
                    if s > d:
                        candidates.add(s - d)
                candidates.add(d)
                candidates = sorted(candidates)
                
                placed = False
                for x in candidates:
                    if x < 0 or x > 10**6:
                        continue
                    if x in S:
                        continue
                    
                    new_S = S + [x]
                    if is_valid_sidon(new_S):
                        S = new_S
                        covered = compute_diffs(S)
                        placed = True
                        break
                
                if not placed:
                    success = False
                    break
            
            if success and max(S) < best_max:
                best_S = S
                best_max = max(S)
    
    # Final improvement pass
    if time.time() < deadline - 2 and best_S:
        best_S = try_improve(best_S, deadline - 2)
    
    return sorted(best_S) if best_S else [0]

# EVOLVE-BLOCK-END
