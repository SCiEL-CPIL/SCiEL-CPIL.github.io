import numpy as np
import itertools
import time

D = 11

def generate_candidates(max_norm_sq):
    """
    Generate all nonzero integer vectors in R^D with norm squared <= max_norm_sq.
    """
    candidates = []
    # We need to generate integer vectors.
    # Since D=11 is small and max_norm_sq is small (<= 10), we can use recursion or itertools.
    # However, brute force over ranges is too slow.
    # We can generate by weight.
    
    # Helper to add permutations of a tuple with signs
    def add_vectors_with_values(values):
        # values is a list of non-zero magnitudes
        # We need to place them in D dimensions and assign signs.
        # This is getting complex to do generically.
        # Let's just iterate over all possible combinations of coordinates?
        # No, that's huge.
        
        pass

    # Better: Generate by weight k
    # For k=1 to max_weight
    # Max weight is sqrt(max_norm_sq)
    max_w = int(np.sqrt(max_norm_sq))
    
    # Precompute combinations of indices
    # This is still heavy.
    
    # Alternative: 
    # Since max_norm_sq is small, the number of non-zero elements is small.
    # Let's use a recursive generator.
    
    def gen(idx, remaining_norm_sq, current_vec):
        if idx == D:
            if remaining_norm_sq == 0 and any(x != 0 for x in current_vec):
                candidates.append(tuple(current_vec))
            return
        
        # Option 1: current_vec[idx] = 0
        gen(idx + 1, remaining_norm_sq, current_vec)
        
        # Option 2: current_vec[idx] = +/- val
        # val^2 <= remaining_norm_sq
        for val in range(1, int(np.sqrt(remaining_norm_sq)) + 1):
            # +val
            current_vec[idx] = val
            gen(idx + 1, remaining_norm_sq - val*val, current_vec)
            # -val
            current_vec[idx] = -val
            gen(idx + 1, remaining_norm_sq - val*val, current_vec)
            # Reset
            current_vec[idx] = 0
            
    gen(0, max_norm_sq, [0]*D)
    
    # Remove duplicates (shouldn't be any due to recursion logic, but just in case)
    unique_candidates = list(set(candidates))
    return np.array(unique_candidates, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_vectors = None
    best_count = 0
    
    # Try different max norm squared values
    # K=2 is the seed.
    # K=4 gave ~480 in previous attempts.
    # Let's try K=2, 3, 4, 5, 6, 9.
    target_norm_sqs = [2, 3, 4, 5, 6, 8, 9, 10]
    
    for K in target_norm_sqs:
        if time.time() > deadline:
            break
            
        try:
            candidates = generate_candidates(K)
            if len(candidates) == 0:
                continue
                
            # Greedy with random restarts
            # Number of restarts depends on time available
            num_restarts = 50
            
            for r in range(num_restarts):
                if time.time() > deadline:
                    break
                    
                # Shuffle candidates
                np.random.shuffle(candidates)
                
                selected = []
                selected_matrix = []
                
                for v in candidates:
                    # Check if v can be added
                    if not selected_matrix:
                        selected.append(v)
                        selected_matrix.append(v)
                        continue
                        
                    # Vectorized distance check
                    # dist_sq = np.sum((np.array(selected_matrix) - v)**2, axis=1)
                    # This creates a large array every time.
                    # Better to maintain the matrix and update.
                    
                    # For speed, let's use a loop if N is small, or numpy if large.
                    # N ~ 1000-10000.
                    # Let's use numpy.
                    S = np.array(selected_matrix)
                    dists = np.sum((S - v)**2, axis=1)
                    
                    if np.min(dists) >= K:
                        selected.append(v)
                        selected_matrix.append(v)
                        
                if len(selected) > best_count:
                    best_count = len(selected)
                    best_vectors = np.array(selected, dtype=np.int64)
                    
        except Exception as e:
            pass
            
    if best_vectors is None:
        # Fallback to seed logic if something went wrong
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return best_vectors
