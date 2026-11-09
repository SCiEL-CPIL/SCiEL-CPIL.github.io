import time
import random
import numpy as np
from scipy.signal import fftconvolve

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for 90s limit
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128

    def compute_counts(A):
        """
        Compute representation counts for all sums using FFT convolution.
        For unordered pairs with equal summands permitted:
        - If a != b: sum a+b appears once
        - If a == b: sum 2a appears once
        The convolution of indicator vector with itself gives ordered counts.
        For unordered: count[s] = (ordered_count[s] + (1 if s is even and s/2 in A else 0)) // 2
        Actually, let's think more carefully.
        
        Let f[i] = 1 if i in A, else 0.
        Convolution (f*f)[s] = sum_i f[i]*f[s-i] = number of ordered pairs (i,j) with i+j=s and i,j in A.
        
        For unordered pairs {i,j} with i<=j:
        - If i != j: the ordered pair (i,j) and (j,i) both contribute, so ordered count includes 2 for each such pair.
        - If i == j: only one ordered pair (i,i), contributes 1 to ordered count.
        
        So unordered_count[s] = (ordered_count[s] + diag_count[s]) // 2
        where diag_count[s] = 1 if s is even and s/2 in A, else 0.
        
        Wait, let me re-derive:
        ordered_count[s] = sum_{i+j=s, i!=j} 1 + sum_{i+j=s, i==j} 1
                         = 2 * (number of unordered pairs with i!=j and i+j=s) + (1 if s/2 in A and s even)
        
        unordered_count[s] = (number of unordered pairs with i!=j and i+j=s) + (1 if s/2 in A and s even)
                           = (ordered_count[s] - diag[s]) / 2 + diag[s]
                           = (ordered_count[s] + diag[s]) / 2
        where diag[s] = 1 if s even and s/2 in A, else 0.
        """
        if not A:
            return np.zeros(RANGE_SIZE, dtype=np.int32)
        
        A_arr = np.array(sorted(A), dtype=np.int32)
        # Create indicator vector
        f = np.zeros(MAX_VAL + 1, dtype=np.float64)
        f[A_arr] = 1.0
        
        # Convolve with itself
        conv = fftconvolve(f, f, mode='full')
        # conv[s] = ordered count for sum s
        
        # Extract our range
        ordered = conv[N_LOW:N_HIGH+1].astype(np.int32)
        
        # Compute diagonal contribution
        diag = np.zeros(RANGE_SIZE, dtype=np.int32)
        for idx in range(RANGE_SIZE):
            s = N_LOW + idx
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= MAX_VAL and half in A:
                    diag[idx] = 1
        
        # Unordered count
        counts = (ordered + diag) // 2
        
        return counts

    def evaluate(A):
        """
        Evaluates the set A.
        Returns (peak, is_covered).
        """
        if not A:
            return float('inf'), False
            
        counts = compute_counts(A)
        
        if not np.all(counts > 0):
            return float('inf'), False
            
        return int(np.max(counts)), True

    def greedy_peel():
        """
        Start from full set [0,128] and greedily remove elements
        that don't increase the peak or break coverage.
        """
        A = set(range(0, MAX_VAL + 1))
        
        # Try removing elements in random order
        elements = list(range(0, MAX_VAL + 1))
        random.shuffle(elements)
        
        for elem in elements:
            if time.time() > deadline - 15:
                break
            if elem not in A:
                continue
            
            A_test = A - {elem}
            if len(A_test) < 2:
                continue
            
            peak_test, covered_test = evaluate(A_test)
            if covered_test:
                A = A_test
        
        return A

    best_peak = float('inf')
    best_A = None

    # Phase 1: Greedy peeling initialization
    A_init = greedy_peel()
    peak_init, covered_init = evaluate(A_init)
    if covered_init:
        best_peak = peak_init
        best_A = list(A_init)
    
    if time.time() > deadline - 20:
        if best_A is not None:
            return best_A
        return list(range(0, MAX_VAL + 1))

    # Phase 2: Targeted local search from the peeled set
    current_A = set(A_init)
    current_peak = peak_init
    
    # Try to further reduce by swapping elements
    # Focus on elements that are "expensive" (contribute to high-multiplicity sums)
    
    def get_expensive_elements(A):
        """Return elements that participate in many high-multiplicity sums."""
        counts = compute_counts(A)
        peak = int(np.max(counts))
        # Find sums with count == peak
        high_sums = set(N_LOW + i for i in range(RANGE_SIZE) if counts[i] >= peak - 1)
        
        # For each element, count how many high sums it participates in
        elem_score = {}
        for a in A:
            score = 0
            for b in A:
                if a + b in high_sums:
                    score += 1
            elem_score[a] = score
        
        return elem_score

    # Local search: try swapping expensive elements with random ones
    T = 1.0
    cooling_rate = 0.9995
    min_T = 0.05
    
    while time.time() < deadline - 5 and T > min_T:
        # Get expensive elements
        elem_scores = get_expensive_elements(current_A)
        sorted_elems = sorted(elem_scores.keys(), key=lambda x: elem_scores[x], reverse=True)
        
        # Pick an expensive element to potentially remove
        if not sorted_elems:
            break
        
        # Bias towards expensive elements
        r = random.random()
        if r < 0.6:
            # Pick from top 30% expensive
            top_n = max(1, len(sorted_elems) // 3)
            to_remove = random.choice(sorted_elems[:top_n])
        else:
            to_remove = random.choice(sorted_elems)
        
        # Pick a random element to add
        to_add = random.randint(0, MAX_VAL)
        if to_add == to_remove:
            T *= cooling_rate
            continue
        
        new_A = set(current_A)
        new_A.remove(to_remove)
        new_A.add(to_add)
        
        new_peak, is_covered_new = evaluate(new_A)
        
        if is_covered_new:
            if new_peak < current_peak:
                current_A = new_A
                current_peak = new_peak
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_A = list(new_A)
            else:
                # Accept worse with some probability
                delta = new_peak - current_peak
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_A = new_A
                    current_peak = new_peak
        else:
            # Not covered: reject unless we were also not covered before
            # (but we start covered, so just reject)
            pass
        
        T *= cooling_rate

    # Phase 3: Final polish - try removing elements one by one
    if time.time() < deadline - 3:
        A_polish = set(best_A)
        # Try removing each element if possible
        for elem in sorted(A_polish):
            if time.time() > deadline - 3:
                break
            A_test = A_polish - {elem}
            if len(A_test) < 2:
                continue
            peak_test, covered_test = evaluate(A_test)
            if covered_test and peak_test <= best_peak:
                A_polish = A_test
                if peak_test < best_peak:
                    best_peak = peak_test
                    best_A = list(A_polish)

    if best_A is None:
        best_A = list(range(0, MAX_VAL + 1))

    return best_A
