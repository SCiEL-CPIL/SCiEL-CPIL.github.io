"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random
import numpy as np

N = 256
MAX_SUM = 2 * N

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 25.0
    rng = random.Random(42)
    
    # Bitmask-based Sidon check and greedy construction
    # sums is an integer where bit s is 1 if sum s is present
    
    def greedy_bitmask(order):
        S = []
        sums = 0
        for x in order:
            # Check 2*x
            if (sums >> (2*x)) & 1:
                continue
            # Check x+a for all a in S
            conflict = False
            for a in S:
                if (sums >> (x+a)) & 1:
                    conflict = True
                    break
            if not conflict:
                S.append(x)
                sums |= 1 << (2*x)
                for a in S[:-1]:
                    sums |= 1 << (x + a)
        return S

    def is_maximal_numpy(S):
        """Check if S is inclusion-maximal using NumPy for speed."""
        if not S:
            return False
        
        S_arr = np.array(S)
        sums = np.zeros(MAX_SUM + 1, dtype=bool)
        
        # Compute sums
        n = len(S)
        for i in range(n):
            si = S[i]
            sums[2*si] = True
            for j in range(i+1, n):
                sums[si + S[j]] = True
        
        S_set = set(S)
        
        # Check if any missing x can be added
        # x can be added if 2*x is not in sums AND no x+a is in sums
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if sums[2*x]:
                continue
            # Check if any x+a is in sums
            # Vectorized check: sums[x + S_arr]
            if not np.any(sums[x + S_arr]):
                return False
        return True

    def prune_numpy(S):
        """Remove elements while maintaining maximality."""
        S = list(S)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            n = len(S)
            # Try removing from the end first (arbitrary order, could be random)
            # Shuffling the order of removal attempts might help escape local minima?
            # But we just want to find ANY removable element.
            for i in range(n - 1, -1, -1):
                if time.time() > deadline:
                    return S
                new_S = S[:i] + S[i+1:]
                if not new_S:
                    continue
                if is_maximal_numpy(new_S):
                    S = new_S
                    changed = True
                    break
        return S

    # Phase 1: Initial greedy
    best_S = greedy_bitmask(range(1, N + 1))
    best_S = prune_numpy(best_S)
    best_size = len(best_S)
    
    base_order = list(range(1, N + 1))
    
    # Phase 2: Randomized restarts
    for trial in range(100000):
        if time.time() > deadline:
            break
        
        order = base_order[:]
        rng.shuffle(order)
        
        # Apply various heuristics
        t = trial % 10
        if t == 1:
            order = order[::-1]
        elif t == 2:
            # Bias towards high numbers
            order.sort(reverse=True)
            # Partial shuffle
            half = len(order) // 2
            for i in range(half - 1, 0, -1):
                j = rng.randint(0, i)
                order[i], order[j] = order[j], order[i]
        elif t == 3:
            # Alternate low/high
            low = [x for x in order if x <= N // 2]
            high = [x for x in order if x > N // 2]
            new_order = []
            li, hi = 0, 0
            for _ in range(N):
                if li < len(low):
                    new_order.append(low[li]); li += 1
                if hi < len(high):
                    new_order.append(high[hi]); hi += 1
            order = new_order
        elif t == 4:
            # Random hash ordering
            seed_val = trial * 6151
            order = sorted(range(1, N + 1), key=lambda x: (x * seed_val) % (N + 1))
        elif t == 5:
            # Reverse hash
            seed_val = trial * 8191
            order = sorted(range(1, N + 1), key=lambda x: (x * seed_val) % (N + 1), reverse=True)
            
        S = greedy_bitmask(order)
        if S:
            S_pruned = prune_numpy(S)
            if len(S_pruned) < best_size:
                best_size = len(S_pruned)
                best_S = S_pruned

    # Phase 3: Final aggressive pruning on best
    S = prune_numpy(best_S)
    if len(S) < best_size:
        best_S = S

    return best_S

# EVOLVE-BLOCK-END
