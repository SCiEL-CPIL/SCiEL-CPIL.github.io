import time

def construct():
    start_time = time.time()
    deadline = 58.0  # Safety margin before 90s kill
    
    def is_valid(A):
        """Check if set A has disjoint sum sets for different cardinalities."""
        n = len(A)
        if n == 0:
            return True
        # Compute sum sets for each cardinality
        S = [set() for _ in range(n + 1)]
        S[0] = {0}
        for x in A:
            for k in range(n, 0, -1):
                if S[k-1]:
                    S[k].update(s + x for s in S[k-1])
        # Check disjointness
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if S[i] & S[j]:
                    return False
        return True
    
    def is_valid_fast(A):
        """Faster check using interval bounds first, then exact if needed."""
        n = len(A)
        if n == 0:
            return True
        A_sorted = sorted(A)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + A_sorted[i]
        
        # Check interval disjointness
        for k in range(1, n):
            min_k = prefix[k]
            max_k = prefix[n] - prefix[n-k]
            min_next = prefix[k+1]
            if max_k >= min_next:
                # Intervals overlap, need exact check
                return is_valid(A)
        return True
    
    best_size = 0
    best_set = []
    
    # Try contiguous ranges of different lengths and starting points
    # Start with the known good range
    for start in range(1, 257):
        if time.time() - start_time > deadline:
            break
        # Try extending from start
        for end in range(start, 257):
            if time.time() - start_time > deadline:
                break
            A = list(range(start, end + 1))
            n = len(A)
            if n <= best_size:
                continue
            if is_valid_fast(A):
                best_size = n
                best_set = A
    
    # Also try non-contiguous sets via greedy
    def run_greedy(order):
        S = [set() for _ in range(61)]
        S[0] = {0}
        A = []
        current_size = 0
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            
            valid = True
            for j in range(1, current_size + 2):
                if not S[j-1]:
                    continue
                for s in S[j-1]:
                    candidate = s + x
                    for i in range(current_size + 1):
                        if i == j:
                            continue
                        if candidate in S[i]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                A.append(x)
                current_size += 1
                for k in range(current_size, 0, -1):
                    S[k].update(s + x for s in S[k-1])
        
        return A
    
    # Try greedy with different orders
    orders = [
        list(range(1, 257)),
        list(range(256, 0, -1)),
    ]
    
    for order in orders:
        if time.time() - start_time > deadline:
            break
        candidate = run_greedy(order)
        if len(candidate) > best_size:
            best_size = len(candidate)
            best_set = candidate
    
    # Return as list of ints
    return [int(x) for x in best_set]
