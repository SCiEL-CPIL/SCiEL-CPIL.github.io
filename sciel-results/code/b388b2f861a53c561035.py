import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    T_min = 128
    T_max = 256
    N_max = 129
    R = T_max - T_min + 1  # 129
    
    def compute_counts(A_list):
        """
        Compute representation counts for all sums in [T_min, T_max].
        Uses numpy for speed.
        """
        if not A_list:
            return np.zeros(T_max + 1, dtype=np.int32)
        
        A_arr = np.array(A_list, dtype=np.int32)
        # Convolution of A with itself gives counts of ordered pairs (a,b) with a+b=s
        # We want unordered pairs with replacement.
        # Let C_ordered[s] be the number of pairs (a,b) in A x A such that a+b=s.
        # The number of unordered pairs {a,b} with a,b in A (a<=b) such that a+b=s is:
        # If a != b: C_ordered[s] / 2 (since (a,b) and (b,a) are distinct in convolution)
        # If a == b: C_ordered[s] includes (a,a) once.
        # Actually, convolution counts:
        # sum_{a in A} sum_{b in A} [a+b=s]
        # This is the number of ordered pairs.
        # Let k = number of a in A such that 2a = s.
        # Then the number of unordered pairs is (C_ordered[s] + k) / 2.
        # Why?
        # C_ordered[s] = sum_{a!=b, a+b=s} 1 + sum_{a=a, 2a=s} 1
        # C_ordered[s] = 2 * N_distinct + N_equal
        # Unordered count = N_distinct + N_equal
        # N_distinct = (C_ordered[s] - N_equal) / 2
        # Unordered = (C_ordered[s] - N_equal)/2 + N_equal = (C_ordered[s] + N_equal)/2
        
        # Convolve A with A
        # Length of A is n. Max sum is 2*128=256.
        # np.convolve(A_arr, A_arr) where A_arr is 0/1 indicator?
        # Yes, create indicator vector of length 129.
        ind = np.zeros(N_max, dtype=np.int32)
        ind[A_arr] = 1
        
        conv = np.convolve(ind, ind, mode='full')
        # conv[s] is the count of ordered pairs (a,b) with a+b=s.
        
        # Calculate N_equal for each s in [T_min, T_max]
        # N_equal[s] is 1 if s/2 is an integer and s/2 in A, else 0.
        # We can just compute this for the target range.
        
        counts = np.zeros(T_max + 1, dtype=np.int32)
        for s in range(T_min, T_max + 1):
            if s >= len(conv):
                break
            c_ord = conv[s]
            # Check if s/2 is in A
            if s % 2 == 0:
                half = s // 2
                if half < N_max and ind[half] == 1:
                    n_eq = 1
                else:
                    n_eq = 0
            else:
                n_eq = 0
            
            counts[s] = (c_ord + n_eq) // 2
        
        return counts

    def get_max_count(counts):
        return int(np.max(counts[T_min:T_max+1]))

    def is_valid(counts):
        return np.all(counts[T_min:T_max+1] > 0)

    best_A = None
    best_max = float('inf')
    
    def try_improve(A_initial, label=""):
        nonlocal best_A, best_max
        A_set = set(A_initial)
        A_list = list(A_set)
        counts = compute_counts(A_list)
        
        if not is_valid(counts):
            # Repair: add elements to cover gaps
            # Simple greedy repair
            for _ in range(50):
                gaps = np.where(counts[T_min:T_max+1] == 0)[0] + T_min
                if len(gaps) == 0:
                    break
                
                best_y = None
                best_gain = -1
                # Sample candidates
                candidates = random.sample(range(0, N_max), min(30, N_max))
                for y in candidates:
                    if y in A_set:
                        continue
                    gain = 0
                    for s in gaps:
                        a = s - y
                        if 0 <= a < N_max and (a in A_set or a == y):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_y = y
                
                if best_y is not None:
                    A_set.add(best_y)
                    A_list = list(A_set)
                    counts = compute_counts(A_list)
                else:
                    break
            
            if not is_valid(counts):
                return

        current_max = get_max_count(counts)
        if current_max < best_max:
            best_max = current_max
            best_A = set(A_list)

        # Local Search
        A_cur = set(A_list)
        counts_cur = counts
        max_cur = current_max
        
        no_improve_count = 0
        while time.time() < deadline - 2.0:
            # Pick random swap
            A_list_cur = list(A_cur)
            x = random.choice(A_list_cur)
            y = random.randint(0, N_max - 1)
            if y in A_cur:
                continue
            
            # Simulate
            A_new = A_cur - {x}
            A_new.add(y)
            A_new_list = list(A_new)
            c_new = compute_counts(A_new_list)
            
            if not is_valid(c_new):
                continue
            
            m_new = get_max_count(c_new)
            
            if m_new < max_cur:
                A_cur = A_new
                counts_cur = c_new
                max_cur = m_new
                no_improve_count = 0
                if max_cur < best_max:
                    best_max = max_cur
                    best_A = set(A_cur)
            elif m_new == max_cur:
                # Accept with prob 0.1 to escape local minima
                if random.random() < 0.1:
                    A_cur = A_new
                    counts_cur = c_new
                    # max_cur unchanged
            else:
                no_improve_count += 1
                if no_improve_count > 500:
                    break
            
    # 1. Try Seed
    A_seed = list(range(64, 129))
    try_improve(A_seed, "seed")
    
    # 2. Try Random Sizes
    if time.time() < deadline - 10:
        for N in [40, 45, 50, 55, 60]:
            for trial in range(2):
                if time.time() > deadline - 5:
                    break
                random.seed(trial * 100 + N)
                A_rand = random.sample(range(0, N_max), N)
                try_improve(A_rand, f"rand_{N}_{trial}")
                
    if best_A is None:
        return list(range(64, 129))
        
    return list(best_A)
