import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85  # safety margin
    best_peak = float('inf')
    best_A = None
    
    N = 128
    target_start = 128
    target_end = 256
    num_targets = target_end - target_start + 1 # 129
    
    # Precompute indices for convolution
    # v is length 129 (0 to 128)
    # conv = np.convolve(v, v) -> length 257 (0 to 256)
    
    def get_counts_from_mask(mask):
        """
        mask: numpy boolean array of length 129
        returns: numpy array of counts for sums 128..256
        """
        v = mask.astype(np.int32)
        conv = np.convolve(v, v, mode='full')
        # conv[s] is number of ordered pairs (a,b) in A such that a+b=s
        # We want unordered pairs with replacement.
        # Let U[s] be unordered count.
        # conv[s] = sum_{a+b=s} v[a]v[b]
        # If a != b, pair {a,b} contributes 2 to conv[s].
        # If a == b, pair {a,a} contributes 1 to conv[s].
        # Let D[s] be 1 if s is even and s/2 in A, else 0.
        # conv[s] = 2 * (U[s] - D[s]) + D[s] = 2*U[s] - D[s]
        # U[s] = (conv[s] + D[s]) / 2
        
        # Extract relevant sums 128 to 256
        # conv indices correspond to sums 0 to 256
        # We need conv[128:257]
        
        # Calculate D[s] for s in 128..256
        # s is even? s/2 in [0, 128]?
        # s/2 is in A iff mask[s//2] is True (and s is even)
        
        counts = np.zeros(num_targets, dtype=np.int32)
        for i in range(num_targets):
            s = target_start + i
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= 128 and mask[half]:
                    d = 1
                else:
                    d = 0
            else:
                d = 0
            # conv[s] is at index s in the full convolution
            counts[i] = (conv[s] + d) // 2
        return counts

    def is_valid(counts):
        return np.all(counts > 0)

    def get_peak(counts):
        return int(np.max(counts))

    # Initial construction
    # Try to create a set that covers all sums.
    # A simple heuristic: include 0 and 128.
    # Fill rest with random elements.
    
    # To ensure validity, we can start with a "dense" set and remove elements?
    # Or start with sparse and add?
    # Let's try random restarts with local search.
    
    for restart in range(500):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Size k around 40-60?
        # Smaller k means larger average multiplicity if valid, but we want to minimize max.
        # If k is too small, might not be valid.
        # Let's try k=50
        
        k = random.randint(40, 60)
        
        # Initialize mask
        mask = np.zeros(129, dtype=bool)
        # Force 0 and 128 to help with endpoints?
        # 0+128=128. 128+128=256.
        # This guarantees endpoints are covered if we have 0 and 128.
        mask[0] = True
        mask[128] = True
        current_size = 2
        
        # Add random elements
        candidates = list(range(1, 128))
        random.shuffle(candidates)
        for c in candidates:
            if current_size >= k:
                break
            mask[c] = True
            current_size += 1
        
        A_list = [i for i in range(129) if mask[i]]
        
        counts = get_counts_from_mask(mask)
        
        if not is_valid(counts):
            # Try to fix by adding elements? Or just skip this restart.
            # Adding elements increases multiplicity.
            # Let's try a quick repair: find uncovered sums, add elements that cover them.
            # This is complex. Let's just rely on more restarts or better init.
            # A better init: Greedy construction.
            # Start with empty.
            # While there are uncovered sums:
            #   Pick a random pair (a,b) that covers a missing sum?
            #   Or pick a random a, add it.
            # Let's try a greedy init for this restart.
            mask = np.zeros(129, dtype=bool)
            counts = np.zeros(num_targets, dtype=np.int32)
            # We need to cover all 129 sums.
            # Greedy:
            # 1. Add 0, 128.
            mask[0] = True
            mask[128] = True
            # Recalculate counts? Too slow for greedy loop if done naively.
            # Let's stick to random init + local search.
            # If invalid, we can try to add elements to fix coverage, accepting higher peak temporarily.
            
            # Simple repair:
            # Identify uncovered sums.
            uncovered = [target_start + i for i in range(num_targets) if counts[i] == 0]
            if uncovered:
                # Add random elements to cover them
                attempts = 0
                while uncovered and attempts < 100:
                    attempts += 1
                    # Pick a random sum from uncovered
                    s = random.choice(uncovered)
                    # Pick a random a in [0, 128]
                    a = random.randint(0, 128)
                    b = s - a
                    if 0 <= b <= 128:
                        # Add a and b if not present
                        if not mask[a]:
                            mask[a] = True
                        if not mask[b]:
                            mask[b] = True
                        # Recalculate counts (expensive, but only during repair)
                        counts = get_counts_from_mask(mask)
                        # Update uncovered
                        uncovered = [target_start + i for i in range(num_targets) if counts[i] == 0]
                        
                if uncovered:
                    # Still invalid, skip this restart
                    continue
        
        peak = get_peak(counts)
        
        if peak < best_peak:
            best_peak = peak
            best_A = [i for i in range(129) if mask[i]]
            
        # Local Search: Simulated Annealing
        # Move: Swap a in A with b not in A
        # Or: Add b not in A (if we allow size to grow, but we want small size? No, size is not the objective, peak is. But larger size usually allows lower peak if balanced, but increases total pairs. Actually, smaller size is better for minimizing max multiplicity? 
        # If we have k elements, total pairs ~ k^2/2. Distributed over 129 sums, avg is k^2/258.
        # To minimize max, we want k small enough that avg is low, but large enough to cover all sums.
        # k=40 -> avg ~ 6. k=50 -> avg ~ 10.
        # So we should try to keep k small.
        # If we add elements, k increases, avg increases.
        # So we should primarily use swaps (keep k constant) or removals (if valid).
        
        current_peak = peak
        current_counts = counts
        current_mask = mask.copy()
        current_A = [i for i in range(129) if current_mask[i]]
        
        # Simulated Annealing parameters
        T = 10.0
        cooling = 0.995
        min_T = 0.1
        
        while T > min_T and time.time() < deadline:
            # Pick random element to remove
            if not current_A:
                break
            idx = random.randint(0, len(current_A) - 1)
            a = current_A[idx]
            
            # Pick random element to add
            b = random.randint(0, 128)
            if b == a:
                continue
            if current_mask[b]:
                continue
            
            # Apply swap
            current_mask[a] = False
            current_mask[b] = True
            current_A[idx] = b
            current_A.sort() # Keep sorted for consistency? Not strictly needed for mask.
            
            new_counts = get_counts_from_mask(current_mask)
            
            if not is_valid(new_counts):
                # Penalty for invalidity
                # Revert
                current_mask[b] = False
                current_mask[a] = True
                current_A[idx] = a
                T *= cooling
                continue
            
            new_peak = get_peak(new_counts)
            
            delta = new_peak - current_peak
            
            if delta < 0:
                # Accept
                current_peak = new_peak
                current_counts = new_counts
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = current_A.copy()
            else:
                # Accept with probability exp(-delta / T)
                if random.random() < np.exp(-delta / T):
                    current_peak = new_peak
                    current_counts = new_counts
                else:
                    # Revert
                    current_mask[b] = False
                    current_mask[a] = True
                    current_A[idx] = a
            
            T *= cooling
            
            # Occasionally try to remove an element to reduce size
            # If we remove an element and it's still valid, peak might decrease or stay same, but size decreases.
            # Smaller size is generally better for peak.
            if random.random() < 0.1 and len(current_A) > 30:
                # Try removing a random element
                r_idx = random.randint(0, len(current_A) - 1)
                r_a = current_A[r_idx]
                
                current_mask[r_a] = False
                current_A.pop(r_idx)
                
                new_counts = get_counts_from_mask(current_mask)
                
                if is_valid(new_counts):
                    new_peak = get_peak(new_counts)
                    if new_peak <= current_peak:
                        current_peak = new_peak
                        current_counts = new_counts
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A = current_A.copy()
                    else:
                        # Revert
                        current_mask[r_a] = True
                        current_A.append(r_a)
                        current_A.sort()
                else:
                    # Revert
                    current_mask[r_a] = True
                    current_A.append(r_a)
                    current_A.sort()

    if best_A is None:
        # Fallback: return a known valid set, e.g. all numbers
        return list(range(0, 129))
    
    return best_A
