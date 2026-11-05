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
        # Use numpy for speed
        # For each pair (i,j) with i<=j, sum = A[i]+A[j]
        # We need counts for sums in [128, 256]
        
        # Efficient approach: for each a in A, for each b in A with b>=a
        # sum = a+b, if 128 <= sum <= 256, increment count[sum]
        
        counts = np.zeros(257, dtype=np.int32)  # indices 0-256
        
        A_arr = np.array(A_list)
        # Create all pairs
        # This is O(m^2) which is fine for m ~ 40-60
        for i in range(m):
            for j in range(i, m):
                s = A_arr[i] + A_arr[j]
                if 128 <= s <= 256:
                    counts[s] += 1
        
        # Check coverage
        for s in target_sums:
            if counts[s] == 0:
                return float('inf')  # Not covered
        
        return int(np.max(counts[128:257]))
    
    # Initial solution: try a structured set
    # Use two blocks to control representation counts
    # A = {0,1,...,k} U {m-k, m-k+1, ..., m} where m ~ 128
    
    # Let's try several initial structures
    candidates = []
    
    # Structure 1: Two blocks
    for k in range(10, 40):
        for m in range(100, 129):
            A = set(range(0, k+1)) | set(range(m-k, m+1))
            A = A & set(range(0, 129))
            if len(A) > 0:
                candidates.append(A)
    
    # Structure 2: Sparse structured (like Sidon)
    # Use quadratic residues or similar
    for start in range(0, 20):
        A = set()
        for i in range(40):
            val = (start + i*i) % 129
            A.add(val)
        candidates.append(A)
    
    # Also include the seed solution
    candidates.append(set(range(64, 129)))
    
    # Evaluate all candidates and pick the best
    best_A = None
    best_peak = float('inf')
    
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
        
        # Try different move types
        max_iters = 100000
        iter_count = 0
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline and iter_count < max_iters and no_improve < max_no_improve:
            iter_count += 1
            
            # Choose a move
            move_type = random.random()
            
            if move_type < 0.3:
                # Add a random element
                candidates_add = [x for x in range(0, 129) if x not in current_A]
                if not candidates_add:
                    continue
                new_elem = random.choice(candidates_add)
                new_A = current_A.copy()
                new_A.add(new_elem)
                
            elif move_type < 0.6:
                # Remove a random element
                if len(current_A) <= 5:
                    continue
                elems = list(current_A)
                new_elem = random.choice(elems)
                new_A = current_A.copy()
                new_A.discard(new_elem)
                
            else:
                # Swap: remove one, add one
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
            
            # Periodic check for timeout
            if iter_count % 1000 == 0 and time.time() > deadline - 10:
                break
    
    # Final check
    if best_A is None or len(best_A) == 0:
        return list(range(64, 129))
    
    return sorted(best_A)
