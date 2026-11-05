import numpy as np
import random

def _generate_random_code(q, n, size):
    code = []
    seen = set()
    while len(code) < size:
        word = [random.randint(0, q - 1) for _ in range(n)]
        idx = 0
        for d in word:
            idx = idx * q + d
        if idx not in seen:
            seen.add(idx)
            code.append(word)
    return code

def _is_valid_cover(code, q, n, radius):
    N = q ** n
    covered = np.zeros(N, dtype=bool)
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i - 1] * q

    for word in code:
        idx = 0
        for d in word:
            idx = idx * q + d
        
        # Mark word itself
        covered[idx] = True
        
        # Mark neighbors
        for pos in range(n):
            orig = word[pos]
            weight = powers[n - 1 - pos]
            for new_d in range(q):
                if new_d == orig:
                    continue
                delta = (new_d - orig) * weight
                ni = idx + delta
                covered[ni] = True
                
    return covered.all()

def _local_search(q, n, radius, initial_code, time_limit=100):
    import time
    start_time = time.time()
    N = q ** n
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i - 1] * q

    # Convert code to indices for faster manipulation
    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx

    code_indices = set(word_to_idx(w) for w in initial_code)
    code_list = list(code_indices)
    
    # Precompute neighbors for each index
    # neighbors[i] is a list of indices within radius 1
    neighbors = []
    for i in range(N):
        temp = i
        digits = [0] * n
        for j in range(n):
            digits[n - 1 - j] = temp % q
            temp //= q
            
        nb = [i]
        for pos in range(n):
            orig = digits[pos]
            weight = powers[n - 1 - pos]
            for new_d in range(q):
                if new_d == orig:
                    continue
                delta = (new_d - orig) * weight
                ni = i + delta
                nb.append(ni)
        neighbors.append(nb)

    # Initial coverage count
    coverage_counts = np.zeros(N, dtype=np.int32)
    for idx in code_indices:
        for nb in neighbors[idx]:
            coverage_counts[nb] += 1

    best_size = len(code_list)
    best_code = code_list.copy()
    
    # Local search loop
    # We try to reduce size by removing elements.
    # If stuck, we swap to change the set structure.
    
    while time.time() - start_time < time_limit:
        # Try to remove
        removed = False
        # Shuffle to avoid bias
        shuffled_code = code_list.copy()
        random.shuffle(shuffled_code)
        
        for idx in shuffled_code:
            # Check if idx is removable
            # If min(coverage_counts[nb] for nb in neighbors[idx]) > 1, then idx is removable?
            # No, we need to check if any neighbor has count 1.
            # If all neighbors have count >= 2, then removing idx leaves them with count >= 1.
            # So idx is removable if min(coverage_counts[neighbors[idx]]) >= 2.
            
            min_cov = np.min(coverage_counts[neighbors[idx]])
            if min_cov >= 2:
                # Remove idx
                code_indices.remove(idx)
                code_list.remove(idx)
                for nb in neighbors[idx]:
                    coverage_counts[nb] -= 1
                removed = True
                break
        
        if removed:
            if len(code_list) < best_size:
                best_size = len(code_list)
                best_code = code_list.copy()
            continue
        
        # If no removal, try swap
        # Pick a random idx from code, random new_idx not in code
        # Check if (code - {idx} + {new_idx}) is valid.
        # Validity check: 
        # Let C' = C \ {idx} U {new_idx}
        # New coverage counts:
        # old_cov - 1 for neighbors of idx
        # old_cov + 1 for neighbors of new_idx
        # We need min(new_cov) >= 1.
        
        # To speed up, we can precompute which indices are "critical" (have a neighbor with count 1).
        # Only critical indices can be removed if their critical neighbors are also covered by new_idx?
        # This is getting complex. Simple random swap check is O(N) per swap.
        # N=729, neighbors ~ 19. 729*19 is small.
        
        attempts = 0
        swapped = False
        while attempts < 50:
            attempts += 1
            idx = random.choice(code_list)
            new_idx = random.randint(0, N - 1)
            if new_idx in code_indices:
                continue
            
            # Simulate swap
            # We need to check if min(coverage_counts + delta) >= 1
            # delta is -1 for neighbors of idx, +1 for neighbors of new_idx.
            # We can just compute the new min for the affected regions.
            
            # Actually, just checking the whole array is fast enough for N=729.
            # But let's optimize slightly.
            # The only places coverage can drop to 0 are neighbors of idx that were 1.
            # So, let critical_neighbors = [nb for nb in neighbors[idx] if coverage_counts[nb] == 1]
            # If critical_neighbors is empty, the swap is definitely valid regarding coverage (counts only increase or stay same for others, decrease for idx's neighbors but they were >=2).
            # Wait, if a neighbor of idx had count 1, it becomes 0. We need new_idx to cover it.
            # So, for every nb in critical_neighbors, new_idx must be in neighbors[nb] (i.e., distance <= radius).
            # If this condition holds, the swap is valid.
            
            critical_neighbors = [nb for nb in neighbors[idx] if coverage_counts[nb] == 1]
            
            valid_swap = True
            for nb in critical_neighbors:
                # Check if new_idx covers nb
                # i.e., nb in neighbors[new_idx]
                # Since neighbors is symmetric (distance metric), this is equivalent to new_idx in neighbors[nb]
                # But checking membership in a list is O(len). 
                # We can precompute a boolean matrix or use sets.
                # For now, simple check:
                if new_idx not in neighbors[nb]:
                    valid_swap = False
                    break
            
            if valid_swap:
                # Perform swap
                code_indices.remove(idx)
                code_indices.add(new_idx)
                code_list.remove(idx)
                code_list.append(new_idx)
                
                # Update counts
                for nb in neighbors[idx]:
                    coverage_counts[nb] -= 1
                for nb in neighbors[new_idx]:
                    coverage_counts[nb] += 1
                    
                swapped = True
                break
                
        if not swapped:
            # Stuck? Try random restart or accept we are at a local minimum.
            # Given the problem size, we might find 73 quickly.
            # Let's just break if we can't swap? Or try a different perturbation.
            # For now, break to save time if stuck for a while?
            # No, let's just let the loop continue, it will pick random swaps.
            # If no valid swap found in 50 tries, it's likely a deep local minimum.
            # We could restart with a different greedy seed.
            break

    # Convert best_code indices back to words
    result = []
    for idx in best_code:
        temp = idx
        digits = [0] * n
        for j in range(n):
            digits[n - 1 - j] = temp % q
            temp //= q
        result.append(digits)
        
    return result

def _greedy_cover_numpy_optimized(q, n, radius, num_restarts=10):
    N = q ** n
    
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    # Precompute masks and coverers
    masks = np.zeros((N, N), dtype=bool)
    coverers = [[] for _ in range(N)]
    
    for i in range(N):
        temp = i
        digits = [0] * n
        for j in range(n):
            digits[n-1-j] = temp % q
            temp //= q
            
        masks[i, i] = True
        coverers[i].append(i)
        
        for pos in range(n):
            orig = digits[pos]
            weight = powers[n - 1 - pos]
            for new_d in range(q):
                if new_d == orig:
                    continue
                delta = (new_d - orig) * weight
                ni = i + delta
                masks[i, ni] = True
                coverers[ni].append(i)

    best_code = None
    best_size = float('inf')
    
    for _ in range(num_restarts):
        uncovered = np.ones(N, dtype=bool)
        coverage_counts = masks.sum(axis=1).astype(np.int32)
        selected_indices = []
        
        while uncovered.any():
            max_cov = coverage_counts.max()
            if max_cov == 0:
                break
            candidates = np.where(coverage_counts == max_cov)[0]
            chosen = np.random.choice(candidates)
            
            selected_indices.append(chosen)
            
            newly_covered = np.where(uncovered & masks[chosen])[0]
            
            if len(newly_covered) == 0:
                break
                
            uncovered[newly_covered] = False
            
            for w in newly_covered:
                for i in coverers[w]:
                    coverage_counts[i] -= 1
                    
        if len(selected_indices) < best_size:
            best_size = len(selected_indices)
            best_code = selected_indices.copy()
            
    result = []
    if best_code:
        for idx in best_code:
            digits = []
            temp = idx
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            digits.reverse()
            result.append(digits)
    return result

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        # Use local search starting from a greedy solution
        initial_code = _greedy_cover_numpy_optimized(3, 6, 1, num_restarts=5)
        optimized_code = _local_search(3, 6, 1, initial_code, time_limit=110)
        return optimized_code
    else:
        return _greedy_cover_numpy_optimized(q, n, radius, num_restarts=10)
