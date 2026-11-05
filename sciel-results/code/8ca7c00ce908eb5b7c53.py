import time
import random
import numpy as np
from scipy import signal

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    num_targets = len(target_sums)
    
    # Precompute mapping from sum value to index in target_sums
    sum_to_idx = np.full(257, -1, dtype=np.int32)
    for i, s in enumerate(target_sums):
        sum_to_idx[s] = i

    def get_counts_convolution(A_binary):
        """Compute representation counts using convolution.
        
        A_binary is a binary array of length 129.
        The pair-sum multiplicity for sum s is:
          - For s = 2*x: count of (x,x) = A[x]
          - For s = x+y with x != y: count of (x,y) and (y,x), but we want unordered
          
        Actually, let's think carefully. For unordered pairs with equal summands allowed:
        - (x, x): 1 representation if x in A
        - (x, y) with x < y: 1 representation if both x, y in A
        
        The total count for sum s is:
        sum_{x+y=s} [x in A and y in A] where we count (x,y) once for x<=y
        
        This is equivalent to:
        For each x from 0 to s//2:
          if x <= s-x and x in A and (s-x) in A:
            count += 1
        
        Using convolution:
        Let f = A_binary (length 129)
        The full convolution f * f gives, for index k:
          sum_{i+j=k} f[i] * f[j]
        
        For k = s, this counts:
          - (x, s-x) for all x where 0 <= x <= s and 0 <= s-x <= 128
          - This includes both (x, s-x) and (s-x, x) when x != s-x
          - And (s/2, s/2) once when s is even
        
        So the convolution count for sum s is:
          conv[s] = sum_{x: 0<=x<=128, 0<=s-x<=128} A[x] * A[s-x]
        
        For unordered pairs:
          - If s is even: conv[s] = 1 + 2 * (number of unordered pairs (x, s-x) with x < s-x)
            Wait no. Let me reconsider.
          
          conv[s] = sum_{x} A[x] * A[s-x]
          
          For x < s-x (i.e., x < s/2): contributes A[x]*A[s-x] which counts the pair (x, s-x)
          For x > s-x (i.e., x > s/2): contributes A[x]*A[s-x] which counts the same pair (s-x, x)
          For x = s-x (i.e., x = s/2, s even): contributes A[s/2]^2 = A[s/2]
          
          So conv[s] = 2 * (sum_{x < s/2} A[x]*A[s-x]) + (A[s/2] if s even else 0)
          
          The unordered pair count is:
            sum_{x < s/2} A[x]*A[s-x] + (A[s/2] if s even else 0)
          
          So: unordered_count[s] = (conv[s] + (A[s/2] if s even else 0)) / 2
          
          Wait, that's not quite right either. Let me be more careful.
          
          Let U(s) = number of unordered pairs {x,y} with x,y in A, x+y=s, x<=y
          
          For x < y: the pair {x,y} contributes 1 to U(s), and 2 to conv[s] (from (x,y) and (y,x))
          For x = y: the pair {x,x} contributes 1 to U(s), and 1 to conv[s] (from (x,x))
          
          So: conv[s] = 2 * (number of pairs with x < y) + (number of pairs with x = y)
          And: U(s) = (number of pairs with x < y) + (number of pairs with x = y)
          
          Therefore: U(s) = (conv[s] + (number of pairs with x = y)) / 2
          
          Number of pairs with x = y and x+y=s is 1 if s is even and s/2 in A, else 0.
          
          So: U(s) = (conv[s] + (1 if s%2==0 and A[s//2] else 0)) // 2
        
        """
        # Compute full convolution
        conv = signal.convolve(A_binary, A_binary, mode='full')
        # conv has length 257, conv[k] corresponds to sum k
        
        counts = np.zeros(num_targets, dtype=np.int32)
        for i, s in enumerate(target_sums):
            if s >= len(conv):
                continue
            c = int(conv[s])
            # Add back the diagonal term if s is even
            if s % 2 == 0:
                mid = s // 2
                if 0 <= mid <= 128 and A_binary[mid]:
                    c += 1
            counts[i] = c // 2
        return counts

    def get_counts_direct(A_set):
        """Compute representation counts for target sums given set A (direct method)."""
        counts = np.zeros(num_targets, dtype=np.int32)
        A_list = sorted(A_set)
        m = len(A_list)
        for i in range(m):
            ai = A_list[i]
            # Pair (ai, ai)
            s = 2 * ai
            idx = sum_to_idx[s]
            if idx >= 0:
                counts[idx] += 1
            # Pairs (ai, aj) for j > i
            for j in range(i + 1, m):
                aj = A_list[j]
                s = ai + aj
                idx = sum_to_idx[s]
                if idx >= 0:
                    counts[idx] += 1
        return counts

    def calculate_peak(counts):
        """Return the maximum multiplicity in the target interval."""
        if np.any(counts == 0):
            return 1000 + np.sum(counts == 0)
        return int(np.max(counts))

    def get_cost(counts):
        if np.any(counts == 0):
            return 1000 + np.sum(counts == 0)
        return int(np.max(counts))

    def initial_solution():
        """Create a valid solution using greedy coverage."""
        A = set()
        size_init = 40
        A = set(random.sample(range(129), size_init))
        
        A_binary = np.zeros(129, dtype=np.int32)
        for x in A:
            A_binary[x] = 1
        
        counts = get_counts_convolution(A_binary)
        if np.any(counts == 0):
            candidates = [x for x in range(129) if x not in A]
            random.shuffle(candidates)
            
            for x in candidates:
                if not np.any(counts == 0):
                    break
                
                # Check if adding x helps
                A_binary[x] = 1
                new_counts = get_counts_convolution(A_binary)
                A_binary[x] = 0
                
                if np.sum(new_counts == 0) < np.sum(counts == 0):
                    A.add(x)
                    A_binary[x] = 1
                    counts = new_counts
        
        return A, calculate_peak(get_counts_convolution(A_binary))

    # Find a good starting point
    best_A = set()
    best_peak = 10000
    current_A = set()
    current_peak = 10000
    
    # Try a few random restarts for initialization
    for _ in range(15):
        if time.time() > deadline - 30.0:
            break
        A_try, peak_try = initial_solution()
        if peak_try < best_peak:
            best_peak = peak_try
            best_A = set(A_try)
            current_A = A_try
            current_peak = peak_try
            
    # If still invalid (shouldn't happen), fallback
    if best_peak > 1000:
        current_A = set(range(129))
        current_peak = 10000
        best_A = set(current_A)
        best_peak = 10000

    # Build binary representation
    current_binary = np.zeros(129, dtype=np.int32)
    for x in current_A:
        current_binary[x] = 1
    
    current_counts = get_counts_convolution(current_binary)
    current_cost = get_cost(current_counts)
    
    if current_cost < best_peak:
        best_peak = current_cost
        best_A = set(current_A)
    
    # Iterative Reweighting + Local Search
    # Phase 1: Iterative reweighting to balance multiplicities
    # Phase 2: Fine-tuning with local search
    
    T_start = 5.0
    T_end = 0.005
    alpha = 0.999
    
    all_elements = list(range(129))
    
    # Phase 1: Reweighting iterations
    reweight_steps = 0
    max_reweight_steps = 5000
    
    while time.time() < deadline - 20.0 and reweight_steps < max_reweight_steps:
        reweight_steps += 1
        
        # Compute current counts
        counts = get_counts_convolution(current_binary)
        cost = get_cost(counts)
        
        if cost < best_peak:
            best_peak = cost
            best_A = set(x for x in range(129) if current_binary[x])
        
        # If valid, try to reduce peak
        if not np.any(counts == 0):
            peak = int(np.max(counts))
            if peak <= 2:
                # Already very good, break early
                break
            
            # Identify over-represented sums (count >= peak)
            over_mask = counts >= peak
            over_sums = target_sums[over_mask]
            
            # For each element in A, compute its "load" on over-represented sums
            # Element x contributes to sums 2*x, and x+a for a in A, a != x
            # We want to remove elements that contribute most to over-represented sums
            
            # Compute contribution of each element to over-represented sums
            contrib = np.zeros(129, dtype=np.int32)
            for x in range(129):
                if not current_binary[x]:
                    continue
                # (x, x)
                s = 2 * x
                idx = sum_to_idx[s]
                if idx >= 0 and over_mask[idx]:
                    contrib[x] += 1
                # (x, a) for a in A, a != x
                for a in range(129):
                    if a == x or not current_binary[a]:
                        continue
                    s = x + a
                    idx = sum_to_idx[s]
                    if idx >= 0 and over_mask[idx]:
                        contrib[x] += 1
            
            # Find element with highest contribution to over-represented sums
            if len(current_A) > 30:
                in_A = [x for x in range(129) if current_binary[x]]
                if in_A:
                    max_contrib = max(contrib[x] for x in in_A)
                    if max_contrib > 0:
                        # Try removing the worst contributor
                        worst_x = max(in_A, key=lambda x: contrib[x])
                        
                        # Check if removal keeps validity
                        current_binary[worst_x] = 0
                        new_counts = get_counts_convolution(current_binary)
                        new_cost = get_cost(new_counts)
                        
                        if new_cost <= cost:
                            # Accept removal
                            current_A.discard(worst_x)
                            current_counts = new_counts
                            current_cost = new_cost
                            if new_cost < best_peak:
                                best_peak = new_cost
                                best_A = set(x for x in range(129) if current_binary[x])
                        else:
                            # Restore
                            current_binary[worst_x] = 1
        
        # Also try adding elements that cover under-represented sums
        if np.any(counts == 0):
            under_mask = counts == 0
            under_sums = target_sums[under_mask]
            
            # Find candidate elements that cover the most under-represented sums
            best_gain = 0
            best_x = -1
            for x in range(129):
                if current_binary[x]:
                    continue
                # Count how many under-represented sums this would cover
                gain = 0
                # (x, x)
                s = 2 * x
                idx = sum_to_idx[s]
                if idx >= 0 and under_mask[idx]:
                    gain += 1
                # (x, a) for a in A
                for a in range(129):
                    if not current_binary[a]:
                        continue
                    s = x + a
                    idx = sum_to_idx[s]
                    if idx >= 0 and under_mask[idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x >= 0 and best_gain > 0:
                current_binary[best_x] = 1
                current_A.add(best_x)
                current_counts = get_counts_convolution(current_binary)
                current_cost = get_cost(current_counts)
        
        # Occasionally do a swap to escape local optima
        if reweight_steps % 50 == 0 and len(current_A) > 25:
            # Random swap
            in_A = list(current_A)
            if len(in_A) > 1:
                x = random.choice(in_A)
                out_A = [y for y in range(129) if not current_binary[y]]
                if out_A:
                    y = random.choice(out_A)
                    
                    # Try the swap
                    current_binary[x] = 0
                    current_binary[y] = 1
                    new_counts = get_counts_convolution(current_binary)
                    new_cost = get_cost(new_counts)
                    
                    # Accept with simulated annealing probability
                    if new_cost < current_cost:
                        current_A.discard(x)
                        current_A.add(y)
                        current_counts = new_counts
                        current_cost = new_cost
                        if new_cost < best_peak:
                            best_peak = new_cost
                            best_A = set(current_A)
                    else:
                        # Restore
                        current_binary[x] = 1
                        current_binary[y] = 0
    
    # Phase 2: Fine-tuning with local search
    T = T_start
    steps = 0
    max_steps = 10000
    
    while time.time() < deadline - 2.0 and T > T_end and steps < max_steps:
        steps += 1
        if steps % 100 == 0:
            T *= alpha
        
        # Choose move type
        move_type = random.random()
        new_binary = current_binary.copy()
        
        if move_type < 0.2: # Add
            candidates = [x for x in all_elements if not new_binary[x]]
            if candidates:
                x = random.choice(candidates)
                new_binary[x] = 1
        elif move_type < 0.4: # Remove
            in_A = [x for x in range(129) if new_binary[x]]
            if len(in_A) > 25:
                x = random.choice(in_A)
                new_binary[x] = 0
        elif move_type < 0.7: # Swap
            in_A = [x for x in range(129) if new_binary[x]]
            out_A = [x for x in range(129) if not new_binary[x]]
            if in_A and out_A:
                x = random.choice(in_A)
                y = random.choice(out_A)
                new_binary[x] = 0
                new_binary[y] = 1
        else: # Double swap (remove one, add two) or (remove two, add one)
            in_A = [x for x in range(129) if new_binary[x]]
            out_A = [x for x in range(129) if not new_binary[x]]
            if len(in_A) > 2 and out_A:
                x1 = random.choice(in_A)
                x2 = random.choice([x for x in in_A if x != x1])
                y = random.choice(out_A)
                new_binary[x1] = 0
                new_binary[x2] = 0
                new_binary[y] = 1
            elif in_A and len(out_A) >= 2:
                x = random.choice(in_A)
                y1 = random.choice(out_A)
                y2 = random.choice([x for x in out_A if x != y1])
                new_binary[x] = 0
                new_binary[y1] = 1
                new_binary[y2] = 1
        
        if np.array_equal(new_binary, current_binary):
            continue
        
        new_counts = get_counts_convolution(new_binary)
        new_cost = get_cost(new_counts)
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0:
            current_binary = new_binary
            current_counts = new_counts
            current_cost = new_cost
            # Update set representation
            current_A = set(x for x in range(129) if current_binary[x])
            if current_cost < best_peak:
                best_peak = current_cost
                best_A = set(current_A)
        else:
            prob = np.exp(-delta_cost / T)
            if random.random() < prob:
                current_binary = new_binary
                current_counts = new_counts
                current_cost = new_cost
                current_A = set(x for x in range(129) if current_binary[x])
        
        T *= alpha
    
    # Final check
    if current_cost < best_peak:
        best_peak = current_cost
        best_A = set(x for x in range(129) if current_binary[x])
    
    # Verify best_A
    final_binary = np.zeros(129, dtype=np.int32)
    for x in best_A:
        final_binary[x] = 1
    final_counts = get_counts_convolution(final_binary)
    if np.any(final_counts == 0):
        # Fallback
        return list(range(129))
    
    return list(best_A)
