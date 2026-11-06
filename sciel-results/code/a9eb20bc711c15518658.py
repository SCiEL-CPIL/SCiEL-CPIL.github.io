import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin

    # 1. Precompute APs as a sparse-like structure using numpy
    ap_indices = []
    
    # Generate all APs of length >= 2
    for d in range(1, N + 1):
        for start in range(0, N - d + 1):
            terms = list(range(start, N, d))
            if len(terms) > 1:
                ap_indices.append(terms)
                
    num_aps = len(ap_indices)
    
    # Build incidence matrix: M[j, i] = 1 if index i is in AP j
    # Use a dense numpy array since num_aps * N is manageable
    M = np.zeros((num_aps, N), dtype=np.int8)
    for j, terms in enumerate(ap_indices):
        for i in terms:
            M[j, i] = 1
    
    # For each index i, precompute which APs contain it
    aps_containing = [np.nonzero(M[:, i])[0] for i in range(N)]
    
    # Helper to compute sums from signs vector
    def compute_sums(signs_np):
        """signs_np is a numpy array of shape (N,) with values ±1"""
        return M.dot(signs_np)
    
    def get_max_abs(ap_sums):
        """Return max absolute value from ap_sums numpy array"""
        return int(np.max(np.abs(ap_sums)))
    
    def greedy_init():
        """Greedy initialization: place each sign to minimize current max AP sum"""
        signs = np.zeros(N, dtype=np.int8)
        ap_sums = np.zeros(num_aps, dtype=np.int8)
        
        for i in range(N):
            # Try both signs, pick the one that gives lower max abs
            best_sign = 1
            best_max = N + 1
            
            for s in [1, -1]:
                # Compute new ap_sums
                new_sums = ap_sums.copy()
                for ap_idx in aps_containing[i]:
                    new_sums[ap_idx] += s
                m = get_max_abs(new_sums)
                if m < best_max:
                    best_max = m
                    best_sign = s
            
            signs[i] = best_sign
            for ap_idx in aps_containing[i]:
                ap_sums[ap_idx] += best_sign
        
        return signs, ap_sums
    
    def try_2flip(signs, ap_sums, i, j, current_max):
        """
        Try flipping signs at positions i and j simultaneously.
        Returns (accepted, new_max)
        """
        if i == j:
            return False, current_max
            
        old_si = signs[i]
        old_sj = signs[j]
        
        # Deltas for each position
        delta_i = -2 * old_si  # change in sign at i
        delta_j = -2 * old_sj  # change in sign at j
        
        # Find affected APs
        affected_aps = set(aps_containing[i].tolist()) | set(aps_containing[j].tolist())
        
        # Store old values for revert
        old_values = {}
        
        # Apply the 2-flip
        for ap_idx in affected_aps:
            old_sum = ap_sums[ap_idx]
            new_sum = old_sum
            if M[ap_idx, i] == 1:
                new_sum += delta_i
            if M[ap_idx, j] == 1:
                new_sum += delta_j
            old_values[ap_idx] = old_sum
            ap_sums[ap_idx] = new_sum
        
        # Compute new max
        new_max = get_max_abs(ap_sums)
        
        # Decide acceptance
        accepted = False
        if new_max < current_max:
            accepted = True
        elif new_max == current_max:
            if random.random() < 0.3:
                accepted = True
        
        if not accepted:
            # Revert
            for ap_idx, old_sum in old_values.items():
                ap_sums[ap_idx] = old_sum
            return False, current_max
        
        # Apply sign changes
        signs[i] = -old_si
        signs[j] = -old_sj
        return True, new_max
    
    def try_single_flip(signs, ap_sums, i, current_max):
        """
        Try flipping sign at position i.
        Returns (accepted, new_max)
        """
        old_si = signs[i]
        delta_i = -2 * old_si
        
        affected_aps = aps_containing[i]
        
        # Store old values
        old_values = {}
        
        for ap_idx in affected_aps:
            old_sum = ap_sums[ap_idx]
            ap_sums[ap_idx] = old_sum + delta_i
            old_values[ap_idx] = old_sum
        
        new_max = get_max_abs(ap_sums)
        
        accepted = False
        if new_max < current_max:
            accepted = True
        elif new_max == current_max:
            if random.random() < 0.4:
                accepted = True
        
        if not accepted:
            for ap_idx, old_sum in old_values.items():
                ap_sums[ap_idx] = old_sum
            return False, current_max
        
        signs[i] = -old_si
        return True, new_max
    
    best_signs = None
    best_val = N + 1
    
    # Multi-restart with different strategies
    while time.time() < deadline - 3:
        # Alternate between random init and greedy init
        use_greedy = random.random() < 0.3
        
        if use_greedy:
            signs, ap_sums = greedy_init()
        else:
            signs = np.random.choice([-1, 1], size=N).astype(np.int8)
            ap_sums = compute_sums(signs)
        
        current_max = get_max_abs(ap_sums)
        
        if current_max < best_val:
            best_val = current_max
            best_signs = signs.copy()
            if best_val <= 2:
                break
        
        # Local search: mix of single and 2-flip moves
        steps = 0
        max_steps = 30000
        
        while steps < max_steps and time.time() < deadline - 3:
            # Choose move type: 70% single flip, 30% 2-flip
            if random.random() < 0.7:
                i = random.randint(0, N - 1)
                accepted, new_max = try_single_flip(signs, ap_sums, i, current_max)
            else:
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if i == j:
                    steps += 1
                    continue
                accepted, new_max = try_2flip(signs, ap_sums, i, j, current_max)
            
            if accepted:
                current_max = new_max
                if current_max < best_val:
                    best_val = current_max
                    best_signs = signs.copy()
                    if best_val <= 2:
                        break
            
            steps += 1
        
        if best_val <= 2:
            break

    if best_signs is None:
        return [1] * N
        
    return best_signs.tolist()
