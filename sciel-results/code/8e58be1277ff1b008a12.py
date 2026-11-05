"""Additive bases with few representations - structural design using greedy set cover."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))
    
    def compute_peak(A_set):
        """Compute the peak representation multiplicity for all target sums."""
        A_list = sorted(A_set)
        m = len(A_list)
        if m == 0:
            return float('inf')
        
        # Count representations for each target sum
        # For unordered pairs (a,b) with a <= b
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        
        for i in range(m):
            a = A_list[i]
            for j in range(i, m):
                b = A_list[j]
                s = a + b
                if min_sum <= s <= max_sum:
                    counts[s] += 1
        
        # Check coverage
        target_counts = counts[min_sum:max_sum+1]
        if np.any(target_counts == 0):
            return float('inf')
        
        return int(np.max(target_counts))
    
    def compute_peak_fast(A_set):
        """Faster peak computation using convolution-like approach."""
        A_list = sorted(A_set)
        m = len(A_list)
        if m == 0:
            return float('inf')
        
        # Use numpy to count pairs
        # For each a in A, count how many b in A satisfy a+b = s for each s
        # But we need unordered pairs with a <= b
        
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        
        for a in A_list:
            # For each a, find all b >= a such that a+b is in range
            # b ranges from a to max_sum - a, and b must be in A
            # We can use binary search or just iterate
            
            # Actually, let's just do double loop but with numpy indexing
            pass
        
        # Simpler: just do the double loop
        for i in range(m):
            a = A_list[i]
            for j in range(i, m):
                b = A_list[j]
                s = a + b
                if min_sum <= s <= max_sum:
                    counts[s] += 1
        
        target_counts = counts[min_sum:max_sum+1]
        if np.any(target_counts == 0):
            return float('inf')
        
        return int(np.max(target_counts))
    
    def greedy_set_cover():
        """Greedy approach: start with empty set, add elements to cover uncovered sums while minimizing peak."""
        A = set()
        uncovered = set(target_sums)
        
        # Precompute: for each possible element x in [0,128], which target sums can it help cover?
        # A sum s can be written as x + y where x,y in A
        # When we add x to A, the new sums covered are x+y for y in A (including y=x)
        # plus we can also cover sums with future elements
        
        # Better approach: for each x, compute what sums x could potentially form
        # with existing A elements
        
        while uncovered:
            if time.time() > deadline - 5.0:
                break
            
            # For each candidate x in [0,128] not in A, compute:
            # 1. How many currently uncovered sums would be covered (x+y for y in A, y>=x or x+y where y in A)
            # 2. The resulting peak if we add x
            
            best_x = None
            best_score = None
            best_gain = 0
            best_peak = float('inf')
            
            candidates = [x for x in range(0, 129) if x not in A]
            
            for x in candidates:
                # Sums that x can form with existing A
                new_sums = set()
                for y in A:
                    s = x + y
                    if min_sum <= s <= max_sum:
                        new_sums.add(s)
                # Also x+x
                s_xx = x + x
                if min_sum <= s_xx <= max_sum:
                    new_sums.add(s_xx)
                
                gain = len(new_sums & uncovered)
                if gain == 0:
                    continue
                
                # Compute resulting peak
                A_test = A | {x}
                peak = compute_peak(A_test)
                
                # Score: prefer more gain, then lower peak
                if best_score is None or gain > best_gain or (gain == best_gain and peak < best_peak):
                    best_score = (gain, -peak)
                    best_gain = gain
                    best_peak = peak
                    best_x = x
            
            if best_x is None:
                # No candidate helps, try any candidate
                if candidates:
                    best_x = random.choice(candidates)
                else:
                    break
            
            A.add(best_x)
            # Update uncovered
            for y in list(A):
                s = best_x + y
                if min_sum <= s <= max_sum:
                    uncovered.discard(s)
            s_xx = best_x + best_x
            if min_sum <= s_xx <= max_sum:
                uncovered.discard(s_xx)
        
        return A
    
    def local_search(A_set, max_iter=500):
        """Local search: try to improve by swapping elements."""
        A = set(A_set)
        current_peak = compute_peak(A)
        
        if current_peak == float('inf'):
            return A, current_peak
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            # Try removing an element and adding another
            for x in list(A):
                if time.time() > deadline - 5.0:
                    break
                
                A_without_x = A - {x}
                peak_without = compute_peak(A_without_x)
                
                if peak_without == float('inf'):
                    continue
                
                # Try adding various y
                for y in range(0, 129):
                    if y in A_without_x:
                        continue
                    
                    A_new = A_without_x | {y}
                    peak_new = compute_peak(A_new)
                    
                    if peak_new < current_peak:
                        A = A_new
                        current_peak = peak_new
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return A, current_peak
    
    # Strategy 1: Greedy set cover
    A_greedy = greedy_set_cover()
    peak_greedy = compute_peak(A_greedy)
    
    best_A = A_greedy
    best_peak = peak_greedy
    
    # Strategy 2: Local search from greedy result
    if time.time() < deadline - 10.0:
        A_ls, peak_ls = local_search(A_greedy, max_iter=200)
        if peak_ls < best_peak:
            best_A = A_ls
            best_peak = peak_ls
    
    # Strategy 3: Try structured sets (arithmetic progressions, etc.)
    if time.time() < deadline - 10.0:
        # Try various structured sets
        structured_candidates = []
        
        # Arithmetic progressions
        for start in range(0, 129):
            for step in range(1, 130):
                A_struct = set()
                x = start
                while x <= 128:
                    A_struct.add(x)
                    x += step
                if len(A_struct) >= 2:
                    structured_candidates.append(A_struct)
        
        # Random sets of various sizes
        for size in range(10, 60):
            for _ in range(20):
                A_rand = set(random.sample(range(0, 129), size))
                structured_candidates.append(A_rand)
        
        for A_cand in structured_candidates:
            if time.time() > deadline - 5.0:
                break
            peak = compute_peak(A_cand)
            if peak < best_peak:
                best_A = A_cand
                best_peak = peak
        
        # Local search from best structured candidate
        if time.time() < deadline - 10.0:
            A_ls2, peak_ls2 = local_search(best_A, max_iter=100)
            if peak_ls2 < best_peak:
                best_A = A_ls2
                best_peak = peak_ls2
    
    # Ensure validity
    final_peak = compute_peak(best_A)
    if final_peak == float('inf'):
        # Fallback to a known valid set
        best_A = set(range(64, 129))
    
    return sorted(best_A)

# EVOLVE-BLOCK-END
