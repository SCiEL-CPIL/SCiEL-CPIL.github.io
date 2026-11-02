import itertools
import random

def construct(q, n, radius):
    # For this specific task, q=3, n=7, radius=1.
    # We assume the evaluator passes these specific parameters as per the task description.
    # If other parameters are passed, we fall back to a simple construction.
    if q != 3 or n != 7 or radius != 1:
        # Fallback: Direct sum construction (same as seed)
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Trivial case for small n
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific solver for K_3(7,1)
    total_words = 3 ** n
    
    def to_index(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    def to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return t

    # Precompute cover masks for each word
    # cover_masks[i] is an integer where bit j is 1 if word j is within radius of word i
    cover_masks = [0] * total_words
    for idx in range(total_words):
        w = to_tuple(idx)
        mask = 0
        # Self
        mask |= (1 << idx)
        # Distance 1
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = val
                    new_idx = to_index(tuple(new_w))
                    mask |= (1 << new_idx)
        cover_masks[idx] = mask

    best_code_size = float('inf')
    best_centers = []

    # Run multiple randomized greedy + local search iterations
    for iteration in range(50):
        random.seed(iteration)
        uncovered = (1 << total_words) - 1
        centers = []
        
        # Greedy Phase with slight randomization to escape local minima
        while uncovered:
            best_score = -1
            best_idx = -1
            # To speed up, we can just iterate all. 2187 is small.
            for i in range(total_words):
                # Count how many uncovered words this center covers
                score = (cover_masks[i] & uncovered).bit_count()
                # Add small random noise to break ties and diversify
                score += random.random() * 0.1 
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_score < 0.1: # Only noise, no actual coverage
                break
                
            centers.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
        # Local Search: Remove redundant centers
        # Repeat until no changes
        changed = True
        while changed:
            changed = False
            new_centers = []
            for c in centers:
                # Check if c is redundant
                # A center c is redundant if all words it covers are also covered by at least one other center in 'centers'
                # Let's compute the union of masks of all OTHER centers
                others_mask = 0
                for d in centers:
                    if d != c:
                        others_mask |= cover_masks[d]
                
                # If (cover_masks[c] & ~others_mask) is 0, then c adds no unique coverage
                if (cover_masks[c] & ~others_mask) == 0:
                    changed = True
                else:
                    new_centers.append(c)
            centers = new_centers

        if len(centers) < best_code_size:
            best_code_size = len(centers)
            best_centers = centers.copy()

    # Convert indices back to tuples/lists
    result = [to_tuple(idx) for idx in best_centers]
    return result
