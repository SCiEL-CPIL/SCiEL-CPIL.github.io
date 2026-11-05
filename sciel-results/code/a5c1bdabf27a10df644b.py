import math
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit

    # Precompute weights
    # w[a] = 1 / (512 - a) for a in 1..511
    weights = [0.0] * 512
    for a in range(1, 512):
        weights[a] = 1.0 / (512 - a)

    # Precompute coprime matrix
    # coprime[a][b] is True if gcd(a, b) == 1
    coprime = [[False] * 512 for _ in range(512)]
    for a in range(1, 512):
        coprime[a][a] = True
        for b in range(a + 1, 512):
            c = (math.gcd(a, b) == 1)
            coprime[a][b] = c
            coprime[b][a] = c

    # Candidates sorted by weight (descending)
    # We only care about 1..511
    candidates = list(range(511, 0, -1))
    
    def is_compatible(candidate, current_set):
        return all(coprime[candidate][x] for x in current_set)

    # 1. Greedy Initialization
    # Start with empty set, add best compatible candidate
    current_set = []
    for a in candidates:
        if is_compatible(a, current_set):
            current_set.append(a)
            
    best_set = current_set[:]
    best_score = sum(weights[a] for a in current_set)

    def calculate_score(s):
        return sum(weights[a] for a in s)

    def greedy_fill(remaining_set, max_iter=512):
        """
        Given a set of 'remaining' numbers that are already selected and pairwise coprime,
        greedily add the best possible candidates from the global pool that are coprime to all in 'remaining_set'.
        Returns the new extended set.
        """
        s = list(remaining_set)
        s_set = set(s)
        for b in candidates:
            if b in s_set:
                continue
            if is_compatible(b, s):
                s.append(b)
                s_set.add(b)
        return s

    # 2. Local Search
    improved = True
    while improved:
        improved = False
        
        # Step 2a: Try 1-Remove and Re-fill
        # This allows for significant structural changes
        for i, a in enumerate(best_set):
            # Check time
            if time.time() > deadline:
                return best_set
            
            remaining = [x for j, x in enumerate(best_set) if j != i]
            new_set = greedy_fill(remaining)
            
            new_score = calculate_score(new_set)
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_set = new_set
                improved = True
                break

        # Step 2b: Try 1-Swap
        # Replace one element with another
        for i, a in enumerate(best_set):
            if time.time() > deadline:
                return best_set
                
            remaining = [x for j, x in enumerate(best_set) if j != i]
            rem_set = set(remaining)
            
            best_candidate = None
            best_candidate_score = best_score
            
            # Try replacing a with b
            # We only need to check b if it's better than a (weight-wise) or if it unlocks better options?
            # Actually, since we are doing a swap, we just look for any b not in rem_set such that 
            # score(remaining + [b]) > score(best_set)
            # score(remaining + [b]) = score(remaining) + weights[b]
            # score(best_set) = score(remaining) + weights[a]
            # So we need weights[b] > weights[a] AND b is coprime to remaining.
            
            # Optimization: only consider b with weights[b] > weights[a]
            for b in candidates:
                if weights[b] <= weights[a]:
                    continue
                if b in rem_set or b == a:
                    continue
                if is_compatible(b, remaining):
                    # This swap is valid
                    # Score increase is weights[b] - weights[a]
                    # We want the max increase, so we pick the first valid b in descending weight order?
                    # Yes, candidates is sorted by weight descending.
                    # So the first valid b found will have the highest weight.
                    best_candidate = b
                    best_candidate_score = best_score + (weights[b] - weights[a])
                    break # Since candidates are sorted, first valid is best

            if best_candidate is not None:
                best_set = [best_candidate if x == a else x for x in best_set]
                best_score = best_candidate_score
                improved = True
                break

    return best_set
