import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 128
    min_sum = 128
    max_sum = 256

    def compute_counts(A_set):
        """
        Compute the unordered pair sum counts for a given set A.
        Returns a numpy array of counts for sums 0..256.
        """
        counts = np.zeros(257, dtype=np.int32)
        A_list = sorted(A_set)
        for i, a in enumerate(A_list):
            for b in A_list[i:]:
                s = a + b
                if s <= 256:
                    counts[s] += 1
        return counts

    def random_valid_set():
        """
        Generate a random set A that covers all target sums.
        """
        for _ in range(200):
            size = random.randint(25, 45)
            A = set(random.sample(range(129), size))
            counts = compute_counts(A)
            if np.min(counts[min_sum:max_sum+1]) > 0:
                return A, counts
        # Fallback to a dense set if random fails
        A = set(range(64, 129))
        return A, compute_counts(A)

    # Initialize with a few random restarts to find a good starting point
    best_A = None
    best_counts = None
    best_peak = float('inf')
    
    for _ in range(20):
        A, counts = random_valid_set()
        peak = int(np.max(counts[min_sum:max_sum+1]))
        if peak < best_peak:
            best_peak = peak
            best_A = A.copy()
            best_counts = counts.copy()
            
    if best_A is None:
        best_A, best_counts = random_valid_set()
        best_peak = int(np.max(best_counts[min_sum:max_sum+1]))

    current_A = best_A.copy()
    current_counts = best_counts.copy()
    current_peak = best_peak

    def try_improve(A, counts, peak):
        """
        Try to find a swap that reduces the peak multiplicity.
        Returns (new_A, new_counts, new_peak) or None.
        """
        A_list = list(A)
        not_in_A = [x for x in range(129) if x not in A]
        
        if not not_in_A:
            return None
            
        best_new_peak = peak
        best_new_A = None
        best_new_counts = None
        
        # Try a few random swaps
        # We limit the number of trials to keep it fast
        num_trials = 30
        for _ in range(num_trials):
            x = random.choice(A_list)
            y = random.choice(not_in_A)
            
            new_counts = counts.copy()
            
            # Remove x
            # Sums: x+x, x+z (z in A, z!=x)
            s_2x = 2*x
            if s_2x <= 256:
                new_counts[s_2x] -= 1
            for z in A_list:
                if z == x: continue
                s = x + z
                if s <= 256:
                    new_counts[s] -= 1
                    
            # Add y
            s_2y = 2*y
            if s_2y <= 256:
                new_counts[s_2y] += 1
            for z in A_list:
                if z == x: continue
                s = y + z
                if s <= 256:
                    new_counts[s] += 1
            
            # Check validity: no zeros in target range
            if np.min(new_counts[min_sum:max_sum+1]) < 0:
                continue
            if np.any(new_counts[min_sum:max_sum+1] == 0):
                continue
                
            new_peak = int(np.max(new_counts[min_sum:max_sum+1]))
            
            if new_peak < best_new_peak:
                best_new_peak = new_peak
                best_new_A = A.copy()
                best_new_A.discard(x)
                best_new_A.add(y)
                best_new_counts = new_counts.copy()
                
            if best_new_peak == 1:
                break

        if best_new_A:
            return best_new_A, best_new_counts, best_new_peak
        return None

    # Main Local Search Loop
    while time.time() < deadline:
        res = try_improve(current_A, current_counts, current_peak)
        if res:
            current_A, current_counts, current_peak = res
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = current_A.copy()
                best_counts = current_counts.copy()
        else:
            # Perturbation: Randomly change k elements to escape local optima
            k = random.randint(3, 7)
            if len(current_A) > k:
                to_remove = random.sample(list(current_A), k)
                for x in to_remove:
                    current_A.discard(x)
                
                not_in_A = [x for x in range(129) if x not in current_A]
                if not_in_A:
                    to_add = random.sample(not_in_A, min(k, len(not_in_A)))
                    for x in to_add:
                        current_A.add(x)
                
                current_counts = compute_counts(current_A)
                
                # Check validity, if invalid, revert to best
                if np.any(current_counts[min_sum:max_sum+1] == 0):
                    current_A = best_A.copy()
                    current_counts = best_counts.copy()
                    current_peak = best_peak
                    continue
                current_peak = int(np.max(current_counts[min_sum:max_sum+1]))

    return sorted(best_A)
