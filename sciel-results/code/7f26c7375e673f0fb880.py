"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 27.0  # Safety margin before 30s limit
    
    # Precompute for fast operations
    MAX_SUM = 2 * N  # 512
    
    def get_sums_bitset(S):
        """Compute the sum bitset for set S."""
        sums = 0
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums |= (1 << (S[i] + S[j]))
        return sums

    def get_blocked_elements(S, sums_bitset):
        """Compute the set of elements blocked by S.
        An element y is blocked if 2y is in sums or y+s is in sums for some s in S.
        This is equivalent to y in (sums - S) or 2y in sums.
        """
        blocked = 0
        # Check 2y in sums
        # If bit k is set in sums, then k/2 is blocked if k is even.
        # We can iterate through set bits or just check all y in 1..N
        # For speed, we can compute the set of differences.
        # But N is small (256), so O(N) check is fine.
        for y in range(1, N + 1):
            if (sums_bitset >> (2 * y)) & 1:
                blocked |= (1 << y)
                continue
            for s in S:
                if (sums_bitset >> (y + s)) & 1:
                    blocked |= (1 << y)
                    break
        return blocked

    def is_maximal_fast(S, sums_bitset=None):
        """Check if S is inclusion-maximal using bitset for sums."""
        if not S:
            return False
        if sums_bitset is None:
            sums_bitset = get_sums_bitset(S)
        S_mask = 0
        for s in S:
            S_mask |= (1 << s)
        
        # All elements in [1, N] not in S must be blocked
        # Create a mask of all elements 1..N
        all_mask = 0
        for i in range(1, N + 1):
            all_mask |= (1 << i)
        
        missing_mask = all_mask & ~S_mask
        if missing_mask == 0:
            return True
            
        blocked_mask = get_blocked_elements(S, sums_bitset)
        
        # Check if missing_mask is a subset of blocked_mask
        if (missing_mask & ~blocked_mask) != 0:
            return False
        return True

    def can_add_fast(x, sums_bitset, S):
        """Check if x can be added to S given sums as a bitset."""
        if (sums_bitset >> (2 * x)) & 1:
            return False
        for a in S:
            if (sums_bitset >> (x + a)) & 1:
                return False
        return True

    def greedy_coverage():
        """Greedy construction maximizing coverage (blocking)."""
        S = []
        sums = 0
        S_set = set()
        
        while True:
            if time.time() > deadline:
                break
            
            best_x = -1
            best_gain = -1
            
            for y in range(1, N + 1):
                if y in S_set:
                    continue
                if not can_add_fast(y, sums, S):
                    continue
                
                # Calculate how many new elements this would block
                new_sums = sums | (1 << (2 * y))
                for s in S:
                    new_sums |= (1 << (y + s))
                
                # Count new blocked elements
                gain = 0
                for z in range(1, N + 1):
                    if z in S_set:
                        continue
                    # Was z addable before?
                    was_addable = can_add_fast(z, sums, S)
                    if not was_addable:
                        continue
                    # Is z addable after adding y?
                    if can_add_fast(z, new_sums, S + [y]):
                        continue
                    gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = y
            
            if best_x == -1:
                break
            
            S.append(best_x)
            S_set.add(best_x)
            sums |= (1 << (2 * best_x))
            for s in S[:-1]:
                sums |= (1 << (best_x + s))
            
            if is_maximal_fast(S, sums):
                break
        
        return S

    def greedy_construct(order):
        """Greedy construction with given order."""
        S = []
        sums = 0  # bitset
        for x in order:
            if can_add_fast(x, sums, S):
                S.append(x)
                sums |= (1 << (2 * x))
                for a in S[:-1]:
                    sums |= (1 << (x + a))
        return S

    def prune_aggressive(S):
        """Aggressively prune: try removing each element, check maximality."""
        S = list(S)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                return S
            # Try removing each element
            for i in range(len(S) - 1, -1, -1):
                if time.time() > deadline:
                    return S
                new_S = S[:i] + S[i+1:]
                if not new_S:
                    continue
                new_sums = get_sums_bitset(new_S)
                if is_maximal_fast(new_S, new_sums):
                    S = new_S
                    improved = True
                    break  # Restart with new indices
        return S

    def local_search(S, max_iters=100):
        """Local search: try removing elements to reduce size while maintaining maximality."""
        S = list(S)
        best = S
        best_size = len(S)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            # Try removing a random element
            if len(S) <= 1:
                break
            i = random.Random(len(S) * 7 + _).randint(0, len(S) - 1)
            new_S = S[:i] + S[i+1:]
            if new_S:
                new_sums = get_sums_bitset(new_S)
                if is_maximal_fast(new_S, new_sums):
                    S = new_S
                    if len(S) < best_size:
                        best_size = len(S)
                        best = list(S)
        
        return best

    # Phase 0: Try greedy coverage first
    if time.time() < deadline - 5.0:
        S_cov = greedy_coverage()
        S_cov = prune_aggressive(S_cov)
        best_set = S_cov
        best_size = len(S_cov)
    else:
        best_set = None
        best_size = N + 1
    
    # Phase 1: Multiple deterministic orders
    orders = [
        list(range(N, 0, -1)),  # descending
        list(range(1, N + 1)),  # ascending
        list(range(N // 2, 0, -1)) + list(range(N // 2 + 1, N + 1)),
        list(range(1, N + 1, 2)) + list(range(2, N + 1, 2)),
        list(range(2, N + 1, 2)) + list(range(1, N + 1, 2)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        S = greedy_construct(order)
        S = prune_aggressive(S)
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
    
    # Phase 2: Randomized restarts with local search
    rng = random.Random(42)
    all_elements = list(range(1, N + 1))
    
    iterations = 0
    while time.time() < deadline - 2.0:  # Leave 2s for final checks
        iterations += 1
        # Various random order strategies
        r = rng.random()
        if r < 0.25:
            order = list(all_elements)
            rng.shuffle(order)
        elif r < 0.4:
            # Bias toward larger numbers
            order = sorted(all_elements, key=lambda x: -x + rng.gauss(0, 80))
        elif r < 0.55:
            # Bias toward smaller numbers
            order = sorted(all_elements, key=lambda x: x + rng.gauss(0, 80))
        else:
            # Random with occasional reversal
            order = list(all_elements)
            rng.shuffle(order)
            if rng.random() < 0.3:
                order.reverse()
        
        S = greedy_construct(order)
        S = prune_aggressive(S)
        
        # Local search on the pruned set
        if len(S) < best_size:
            S_ls = local_search(S, max_iters=30)
            if len(S_ls) < best_size:
                best_size = len(S_ls)
                best_set = S_ls
        elif len(S) == best_size:
            # Even if same size, local search might find better structure
            S_ls = local_search(S, max_iters=15)
            if len(S_ls) < best_size:
                best_size = len(S_ls)
                best_set = S_ls
    
    # Final: try more aggressive local search on best found
    if best_set is None:
        best_set = greedy_construct(list(range(N, 0, -1)))
    
    # One more round of local search with more iterations
    if time.time() < deadline - 1.0:
        final_ls = local_search(best_set, max_iters=80)
        if len(final_ls) < len(best_set):
            best_set = final_ls
    
    # Final verification
    if not is_maximal_fast(best_set):
        # Fallback to a known valid construction
        best_set = greedy_construct(list(range(N, 0, -1)))
    
    return best_set

# EVOLVE-BLOCK-END
