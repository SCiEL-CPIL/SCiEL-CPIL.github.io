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
        
        counts = np.zeros(257, dtype=np.int32)
        
        A_arr = np.array(A_list)
        for i in range(m):
            for j in range(i, m):
                s = A_arr[i] + A_arr[j]
                if 128 <= s <= 256:
                    counts[s] += 1
        
        for s in target_sums:
            if counts[s] == 0:
                return float('inf')
        
        return int(np.max(counts[128:257]))
    
    def greedy_set_cover_with_peak_constraint(max_peak):
        """
        Greedy set cover: find minimal A such that all target sums are covered
        and no sum exceeds max_peak representations.
        Returns A as a set, or None if impossible.
        """
        # Precompute for each element x in [0,128], which target sums it can help cover
        # and what the pair partners would be
        # A sum s can be formed by (a, s-a) where a in A and s-a in A
        # We need to select A such that for every s in [128,256], there exists a in A with s-a in A
        
        # Strategy: iteratively add elements that cover the most uncovered sums
        # while respecting the peak constraint
        
        uncovered = set(target_sums)
        A = set()
        counts = np.zeros(257, dtype=np.int32)
        
        # Precompute all possible pairs
        all_pairs = {}  # sum -> list of (a, b) pairs
        for a in range(0, 129):
            for b in range(a, 129):
                s = a + b
                if min_sum <= s <= max_sum:
                    if s not in all_pairs:
                        all_pairs[s] = []
                    all_pairs[s].append((a, b))
        
        for _ in range(200):  # max iterations
            if not uncovered:
                break
            
            # For each candidate element x, compute how many uncovered sums it would help cover
            # and check if adding it violates the peak constraint
            best_x = None
            best_gain = 0
            best_new_uncovered = None
            best_new_counts = None
            
            for x in range(0, 129):
                if x in A:
                    continue
                
                # Simulate adding x
                new_A = A | {x}
                new_counts = counts.copy()
                new_uncovered = set(uncovered)
                
                # Update counts for pairs involving x
                for y in new_A:
                    if y < x:
                        continue  # already counted when y was added
                    s = x + y
                    if min_sum <= s <= max_sum:
                        new_counts[s] += 1
                        if new_counts[s] > max_peak:
                            # Violates peak constraint
                            break
                        if s in new_uncovered and new_counts[s] >= 1:
                            new_uncovered.discard(s)
                else:
                    # No violation
                    gain = len(uncovered) - len(new_uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_new_uncovered = new_uncovered
                        best_new_counts = new_counts
            
            if best_x is None or best_gain == 0:
                return None  # Cannot make progress
            
            A.add(best_x)
            uncovered = best_new_uncovered
            counts = best_new_counts
        
        if uncovered:
            return None
        
        return A
    
    # Try greedy set cover with various peak constraints
    best_A = None
    best_peak = float('inf')
    
    for max_peak in range(2, 15):
        A = greedy_set_cover_with_peak_constraint(max_peak)
        if A is not None:
            peak = compute_peak(A)
            if peak < best_peak:
                best_peak = peak
                best_A = A.copy()
                break  # Found a valid solution with this peak
    
    # If greedy didn't find anything good, fall back to structured candidates
    if best_A is None or best_peak > 10:
        candidates = []
        
        # Structure 1: Two blocks
        for k in range(10, 40):
            for m in range(100, 129):
                A = set(range(0, k+1)) | set(range(m-k, m+1))
                A = A & set(range(0, 129))
                if len(A) > 0:
                    candidates.append(A)
        
        # Structure 2: Sparse structured
        for start in range(0, 20):
            A = set()
            for i in range(40):
                val = (start + i*i) % 129
                A.add(val)
            candidates.append(A)
        
        candidates.append(set(range(64, 129)))
        
        for A in candidates:
            peak = compute_peak(A)
            if peak < best_peak:
                best_peak = peak
                best_A = A.copy()
    
    # Local search to improve
    if best_A is not None:
        current_A = best_A.copy()
        current_peak = best_peak
        
        random.seed(42)
        
        max_iters = 100000
        iter_count = 0
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline and iter_count < max_iters and no_improve < max_no_improve:
            iter_count += 1
            
            move_type = random.random()
            
            if move_type < 0.3:
                candidates_add = [x for x in range(0, 129) if x not in current_A]
                if not candidates_add:
                    continue
                new_elem = random.choice(candidates_add)
                new_A = current_A.copy()
                new_A.add(new_elem)
                
            elif move_type < 0.6:
                if len(current_A) <= 5:
                    continue
                elems = list(current_A)
                new_elem = random.choice(elems)
                new_A = current_A.copy()
                new_A.discard(new_elem)
                
            else:
                if len(current_A) <= 5:
                    continue
                elems = list(current_A)
                remove_elem = random.choice(elems)
                candidates_add = [x for x in range(0, 129) if x not in current_A]
                if not candidates_add:
                    continue
                add_elem = random.choice(candidates_add)
                new_A = current_A.copy()
                new_A.discard(remove_elem)
                new_A.add(add_elem)
            
            new_peak = compute_peak(new_A)
            
            if new_peak < current_peak:
                current_A = new_A
                current_peak = new_peak
                no_improve = 0
                best_A = current_A.copy()
                best_peak = current_peak
            else:
                no_improve += 1
            
            if iter_count % 1000 == 0 and time.time() > deadline - 10:
                break
    
    # Final check
    if best_A is None or len(best_A) == 0:
        return list(range(64, 129))
    
    return sorted(best_A)
