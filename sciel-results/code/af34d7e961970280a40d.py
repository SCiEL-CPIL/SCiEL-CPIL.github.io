"""Construct large sets with cardinality-separated subset sums via local search."""
import random
import time
from collections import defaultdict

def construct():
    """
    Constructs a set of distinct integers in 1..256 such that subset sums
    with different cardinalities are disjoint.
    
    Uses local search to find a large valid set, falling back to a known
    valid arithmetic progression {226..256} of size 31.
    """
    def verify(S):
        """Check if set S satisfies the cardinality-separated condition."""
        S = sorted(S)
        n = len(S)
        # For each cardinality k, compute all subset sums
        # Use the fact that we only need to check separation between
        # adjacent cardinalities (min of k+1 > max of k)
        # Actually we need: for all k != j, sums don't overlap.
        # Sufficient condition: max_sum(k) < min_sum(k+1) for all k
        # This is necessary but not sufficient for general sets,
        # but for our purposes we check the actual sum sets.
        
        # Compute min and max sum for each cardinality
        # min sum of k elements = sum of k smallest
        # max sum of k elements = sum of k largest
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + S[i]
        
        # Check the sufficient condition: max_sum(k) < min_sum(k+1)
        # max_sum(k) = sum of k largest = prefix[n] - prefix[n-k]
        # min_sum(k) = sum of k smallest = prefix[k]
        for k in range(1, n):
            max_k = prefix[n] - prefix[n - k]
            min_k1 = prefix[k + 1]
            if max_k >= min_k1:
                return False
        return True
    
    def verify_exact(S):
        """Exact check using subset sum sets (slower but definitive)."""
        S = sorted(S)
        n = len(S)
        # sums[k] = set of all sums of k-element subsets
        sums = [set() for _ in range(n + 1)]
        sums[0].add(0)
        for x in S:
            for k in range(n - 1, -1, -1):
                for s in list(sums[k]):
                    sums[k + 1].add(s + x)
        # Check disjointness between different cardinalities
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if sums[i] & sums[j]:
                    return False
        return True
    
    # Fallback: known valid AP
    fallback = list(range(226, 257))  # 31 elements
    best = fallback
    best_size = 31
    
    # Local search to try to find larger sets
    deadline = time.time() + 45.0  # Leave buffer for evaluation
    
    # Try different starting points and sizes
    random.seed(42)
    
    # Strategy: Start with a candidate size, use local search
    # The key constraint is that the sum ranges for different cardinalities
    # must be disjoint. For a set S of size n, the tightest constraint is
    # usually between adjacent cardinalities at the extremes.
    
    # Let's try to find sets of size 32, 33, etc.
    for target_size in range(32, 40):
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        for restart in range(200):
            if time.time() > deadline:
                break
            
            # Generate a random starting set
            # Bias toward larger values to keep sum ranges separated
            # Use a perturbed AP-like structure
            base = random.randint(200, 240)
            step = random.randint(1, 3)
            
            S = []
            val = base
            while len(S) < target_size and val <= 256:
                S.append(val)
                val += step + random.randint(0, 2)
            
            if len(S) < target_size:
                # Fill with random values
                remaining = [v for v in range(1, 257) if v not in S]
                random.shuffle(remaining)
                S.extend(remaining[:target_size - len(S)])
            
            if len(S) != target_size:
                continue
            
            # Local search: try to improve
            current = S[:]
            current_valid = verify(current)
            
            if current_valid:
                if len(current) > best_size:
                    best = current[:]
                    best_size = len(current)
                continue
            
            # Try to fix by adjusting elements
            for step_i in range(500):
                if time.time() > deadline:
                    break
                
                # Pick a random element to modify
                idx = random.randint(0, len(current) - 1)
                old_val = current[idx]
                
                # Try a new value
                new_val = random.randint(1, 256)
                if new_val in current:
                    continue
                
                current[idx] = new_val
                if verify(current):
                    if len(current) > best_size:
                        best = current[:]
                        best_size = len(current)
                        # Try to extend
                        break
                    else:
                        # Same size, keep searching for better
                        pass
                else:
                    # Revert
                    current[idx] = old_val
            
            if best_size >= target_size:
                break
    
    # Final verification of best
    if not verify(best):
        # Double-check with exact method
        if not verify_exact(best):
            best = fallback
    
    return list(best)

# EVOLVE-BLOCK-END
