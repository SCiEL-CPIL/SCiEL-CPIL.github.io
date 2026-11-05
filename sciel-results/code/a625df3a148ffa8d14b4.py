"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 25.0  # Safety margin before 30s limit
    
    # Helper: check if adding x to set S (with sum set sums) keeps Sidon property
    def can_add(x, S, sums):
        # Check 2x
        if 2 * x in sums:
            return False
        # Check x + a for a in S
        for a in S:
            if x + a in sums:
                return False
        return True

    def get_sums(S):
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums.add(S[i] + S[j])
        return sums

    def is_maximal_fast(S):
        """Check if S is inclusion-maximal with early termination."""
        if not S:
            return False
        sums = get_sums(S)
        S_set = set(S)
        # Try to find any addable element
        for x in range(1, N + 1):
            if x in S_set:
                continue
            # Quick check: if 2*x is in sums, x cannot be added
            if 2 * x in sums:
                continue
            # Check if x + a is in sums for any a in S
            addable = True
            for a in S:
                if x + a in sums:
                    addable = False
                    break
            if addable:
                return False  # x can be added, so not maximal
        return True

    def greedy_construct(order):
        S = []
        sums = set()
        for x in order:
            if can_add(x, S, sums):
                S.append(x)
                # Update sums
                sums.add(2 * x)
                for a in S[:-1]:  # S[:-1] is S without the newly added x
                    sums.add(x + a)
        return S

    def prune_aggressive(S):
        """Try to remove elements while maintaining maximality, trying multiple strategies."""
        S = list(S)
        if not S:
            return S
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(S)
            if n <= 1:
                break
            
            # Strategy 1: Try removing each element, prefer removing elements that
            # have fewer "conflicts" (i.e., their removal is more likely to preserve maximality)
            # Heuristic: elements with smaller values tend to create more conflicts when removed
            # because they can pair with more other elements
            indices = list(range(n))
            
            # Try removals in order of increasing value (smaller elements first)
            # as they might be "less essential" for maximality
            indices_sorted = sorted(indices, key=lambda i: S[i])
            
            for idx in indices_sorted:
                if time.time() > deadline:
                    break
                new_S = S[:idx] + S[idx+1:]
                if not new_S:
                    continue
                if is_maximal_fast(new_S):
                    S = new_S
                    improved = True
                    break  # Restart with new S
            
            if not improved and n > 2:
                # Try a different order: remove largest elements first
                indices_sorted2 = sorted(indices, key=lambda i: -S[i])
                for idx in indices_sorted2:
                    if time.time() > deadline:
                        break
                    new_S = S[:idx] + S[idx+1:]
                    if not new_S:
                        continue
                    if is_maximal_fast(new_S):
                        S = new_S
                        improved = True
                        break
        
        return S

    def local_search(S):
        """Perform local search: try single-element swaps to reduce size."""
        S = list(S)
        S_set = set(S)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            n = len(S)
            if n <= 1:
                break
            
            # Try removing each element
            for i in range(n):
                if time.time() > deadline:
                    break
                new_S = S[:i] + S[i+1:]
                if not new_S:
                    continue
                if is_maximal_fast(new_S):
                    S = new_S
                    S_set = set(S)
                    improved = True
                    break
        
        return S

    # Phase 1: Deterministic Descending Greedy
    order_desc = list(range(N, 0, -1))
    S_init = greedy_construct(order_desc)
    S_init = prune_aggressive(S_init)
    best_set = S_init
    best_size = len(S_init)
    
    # Phase 2: Randomized Restarts with aggressive pruning
    rng = random.Random(42)
    
    # Precompute some useful structures
    all_elements = list(range(1, N + 1))
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Try different random permutations
        order = list(all_elements)
        rng.shuffle(order)
        
        S = greedy_construct(order)
        
        # Aggressive pruning
        S_pruned = prune_aggressive(S)
        
        # Additional local search
        S_local = local_search(S_pruned)
        
        if len(S_local) < best_size:
            best_size = len(S_local)
            best_set = S_local
    
    # Final verification
    if not is_maximal_fast(best_set):
        # Fallback: use the initial greedy result which should be maximal
        best_set = greedy_construct(order_desc)
    
    return best_set

# EVOLVE-BLOCK-END
