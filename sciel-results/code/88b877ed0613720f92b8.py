import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin under 150s kill

    N = 1000
    MAX_SUM = 2 * N  # max sum is 2*N

    def compute_counts(mask):
        """Compute sum counts for all sums 2..2N using convolution."""
        m = mask.astype(np.int32)
        c = np.convolve(m, m, mode='full')  # length 2N+1
        s = np.arange(2 * N + 1)
        diag = np.zeros(2 * N + 1, dtype=np.int32)
        even = (s % 2 == 0)
        half = s // 2
        valid = even & (half <= N)
        diag[valid] = m[half[valid]]
        counts = (c + diag) // 2
        # counts[s] = number of pairs (a,b) with a<=b, a+b=s
        # We care about sums 2..2N
        return counts[2:2*N+1]  # index 0 corresponds to sum=2

    def delta_add(counts, mask, x):
        """Return new counts after adding x to mask. Returns (new_counts, valid)."""
        new_counts = counts.copy()
        # Adding x: for each y in mask with y <= x, sum x+y gets +1
        # For each y in mask with y > x, sum x+y gets +1 (already counted via symmetry? No.)
        # Actually: representations a+b with a<=b.
        # When we add x, new representations are:
        #   (x, y) for y in mask where x <= y -> sum x+y
        #   (y, x) for y in mask where y < x -> sum x+y (but this is same as (x,y) with x>y, not allowed)
        # Wait: a<=b. So new pairs involving x:
        #   (x, y) where y in mask and x <= y
        #   (y, x) where y in mask and y < x -> but this violates a<=b unless y=x
        # Actually no: if y < x, then the pair would be (y, x) with y < x, which is valid (a=y, b=x, a<=b).
        # So all pairs (min(x,y), max(x,y)) for y in mask.
        # If y == x: (x,x) is one representation.
        # If y != x: (min,max) is one representation.
        # So adding x creates |mask| new representations, one for each y in mask.
        # The sums are x+y for each y in mask.
        
        # But wait: we need to be careful. The count for sum s is the number of pairs (a,b) with a<=b, a+b=s.
        # When we add x, for each existing y in mask:
        #   if x <= y: new pair (x, y), sum = x+y
        #   if x > y: new pair (y, x), sum = x+y
        # So in all cases, sum x+y gets +1.
        # Additionally, if x is already... no, x is not in mask.
        
        # So: for each y in mask, counts[x+y - 2] += 1
        # (subtract 2 because our counts array is indexed from sum=2)
        
        present = np.where(mask)[0]
        present = present[present >= 1]
        for y in present:
            s = x + y
            if 2 <= s <= 2*N:
                new_counts[s - 2] += 1
        
        valid = np.max(new_counts) <= 2
        return new_counts, valid

    def delta_remove(counts, mask, x):
        """Return new counts after removing x from mask."""
        new_counts = counts.copy()
        present = np.where(mask)[0]
        present = present[present >= 1]
        present = present[present != x]  # exclude x itself
        for y in present:
            s = x + y
            if 2 <= s <= 2*N:
                new_counts[s - 2] -= 1
        # Also remove the pair (x, x) if x was in mask
        # When x was in mask, (x,x) contributed to sum 2x
        s = 2 * x
        if 2 <= s <= 2*N:
            new_counts[s - 2] -= 1
        return new_counts

    def is_valid(counts):
        return np.max(counts) <= 2

    def greedy_random_order(rng, time_budget):
        """Greedy construction with random ordering."""
        mask = np.zeros(N + 1, dtype=bool)
        counts = np.zeros(2 * N - 1, dtype=np.int32)  # sums 2..2N
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        size = 0
        for x in candidates:
            if time.time() > time_budget:
                break
            new_counts, valid = delta_add(counts, mask, x)
            if valid:
                mask[x] = True
                counts = new_counts
                size += 1
        return mask, counts, size

    def local_search(mask, counts, size, rng, time_budget):
        """Local search with swap and add/remove moves."""
        best_mask = mask.copy()
        best_counts = counts.copy()
        best_size = size
        
        while time.time() < time_budget:
            improved = False
            
            # Try to add
            absent = np.where(~mask)[0]
            absent = absent[absent >= 1]
            if len(absent) > 0:
                # Try a few random absent elements
                num_tries = min(20, len(absent))
                indices = rng.sample(range(len(absent)), num_tries)
                for idx in indices:
                    x = absent[idx]
                    new_counts, valid = delta_add(counts, mask, x)
                    if valid:
                        mask[x] = True
                        counts = new_counts
                        size += 1
                        improved = True
                        break
            
            # Try swaps
            present = np.where(mask)[0]
            present = present[present >= 1]
            if len(present) > 0 and len(absent) > 0:
                num_tries = min(30, len(present) * len(absent) // 10)
                for _ in range(num_tries):
                    a_idx = rng.randrange(len(present))
                    b_idx = rng.randrange(len(absent))
                    a = present[a_idx]
                    b = absent[b_idx]
                    
                    # Swap: remove a, add b
                    # First remove a
                    counts_no_a = delta_remove(counts, mask, a)
                    mask[a] = False
                    size -= 1
                    # Then add b
                    new_counts, valid = delta_add(counts_no_a, mask, b)
                    if valid:
                        mask[b] = True
                        counts = new_counts
                        size += 1
                        improved = True
                        break
                    else:
                        # Revert
                        mask[a] = True
                        size += 1
                        counts = counts_no_a  # revert to state before removing a
            
            if not improved:
                # Try removing an element to enable future adds
                present = np.where(mask)[0]
                present = present[present >= 1]
                if len(present) > 0:
                    # Try removing a random element
                    num_tries = min(10, len(present))
                    for _ in range(num_tries):
                        a_idx = rng.randrange(len(present))
                        a = present[a_idx]
                        new_counts = delta_remove(counts, mask, a)
                        # Check if this enables any adds
                        mask[a] = False
                        size -= 1
                        counts = new_counts
                        
                        # Try to add some absent elements
                        absent = np.where(~mask)[0]
                        absent = absent[absent >= 1]
                        added = False
                        if len(absent) > 0:
                            num_add_tries = min(20, len(absent))
                            indices = rng.sample(range(len(absent)), num_add_tries)
                            for idx in indices:
                                x = absent[idx]
                                new_counts2, valid2 = delta_add(counts, mask, x)
                                if valid2:
                                    mask[x] = True
                                    counts = new_counts2
                                    size += 1
                                    added = True
                                    break
                        
                        if added and size > best_size:
                            best_mask = mask.copy()
                            best_counts = counts.copy()
                            best_size = size
                            improved = True
                            break
                        elif added and size == best_size:
                            # Keep exploring
                            best_mask = mask.copy()
                            best_counts = counts.copy()
                            # Don't update best_size, just keep current
                            break
                        else:
                            # Revert
                            mask[a] = True
                            size += 1
                            # Recompute counts (expensive but rare)
                            counts = compute_counts(mask)
                            if size > best_size:
                                best_mask = mask.copy()
                                best_counts = counts.copy()
                                best_size = size
        
        return best_mask, best_counts, best_size

    # Main: multi-restart
    best_size = 0
    best_mask = np.zeros(N + 1, dtype=bool)
    
    # First, deterministic greedy as baseline
    mask = np.zeros(N + 1, dtype=bool)
    counts = np.zeros(2 * N - 1, dtype=np.int32)
    size = 0
    for x in range(1, N + 1):
        new_counts, valid = delta_add(counts, mask, x)
        if valid:
            mask[x] = True
            counts = new_counts
            size += 1
    best_size = size
    best_mask = mask.copy()
    
    # Multi-restart with random greedy + local search
    rng = random.Random(42)
    num_restarts = 50  # We'll do as many as time allows
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random greedy
        mask, counts, size = greedy_random_order(rng, deadline - 10)
        
        # Local search
        mask, counts, size = local_search(mask, counts, size, rng, deadline - 5)
        
        if size > best_size:
            best_size = size
            best_mask = mask.copy()
    
    # Final result
    result = [int(x) for x in np.where(best_mask)[0] if x >= 1]
    return result
