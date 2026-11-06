import time
import random

def construct():
    n = 512
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    # Precompute conflicts
    # conflicts[x] is a list of bitmasks. If (S & mask) == mask, x cannot be added.
    # x is 1-indexed in problem, 0-indexed in bit operations.
    # Bit i corresponds to element i+1.
    
    conflicts = [[] for _ in range(n)]
    
    for x in range(1, n + 1):
        i = x - 1
        # Case 1: x is 1st term: x, x+d, x+2d, x+3d
        # Need x+d, x+2d, x+3d in S
        max_d1 = (n - x) // 3
        for d in range(1, max_d1 + 1):
            y1 = x + d
            y2 = x + 2 * d
            y3 = x + 3 * d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
        # Case 2: x is 2nd term: x-d, x, x+d, x+2d
        # Need x-d, x+d, x+2d in S
        max_d2 = min(x - 1, (n - x) // 2)
        for d in range(1, max_d2 + 1):
            y1 = x - d
            y2 = x + d
            y3 = x + 2 * d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
        # Case 3: x is 3rd term: x-2d, x-d, x, x+d
        # Need x-2d, x-d, x+d in S
        max_d3 = min((x - 1) // 2, n - x)
        for d in range(1, max_d3 + 1):
            y1 = x - 2 * d
            y2 = x - d
            y3 = x + d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
        # Case 4: x is 4th term: x-3d, x-2d, x-d, x
        # Need x-3d, x-2d, x-d in S
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            y1 = x - 3 * d
            y2 = x - 2 * d
            y3 = x - d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
    def can_add(x_idx, S):
        """Check if element at index x_idx (0-based) can be added to bitset S."""
        for mask in conflicts[x_idx]:
            if (S & mask) == mask:
                return False
        return True

    def saturate(S, order=None):
        """Add all possible elements to S. If order is provided, use that order."""
        if order is None:
            order = range(n)
        for i in order:
            if not (S & (1 << i)):
                if can_add(i, S):
                    S |= (1 << i)
        return S

    def local_search(S):
        """Improve S using local search with kicks."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Saturate
            S = saturate(S)
            
            # 2. Try to kick
            set_bits = []
            temp_S = S
            for i in range(n):
                if temp_S & (1):
                    set_bits.append(i)
                temp_S >>= 1
                
            if not set_bits:
                break

            # Try removing 1 random element
            rem_idx = random.choice(set_bits)
            S_new = S & ~(1 << rem_idx)
            S_candidate = saturate(S_new)
            
            if S_candidate.bit_count() > S.bit_count():
                S = S_candidate
                improved = True
            else:
                # Try removing 2 random elements
                if len(set_bits) >= 2:
                    rem1, rem2 = random.sample(set_bits, 2)
                    S_new = S & ~(1 << rem1) & ~(1 << rem2)
                    S_candidate = saturate(S_new)
                    
                    if S_candidate.bit_count() > S.bit_count():
                        S = S_candidate
                        improved = True
                    else:
                        # Try removing 3 random elements
                        if len(set_bits) >= 3:
                            rem1, rem2, rem3 = random.sample(set_bits, 3)
                            S_new = S & ~(1 << rem1) & ~(1 << rem2) & ~(1 << rem3)
                            S_candidate = saturate(S_new)
                            
                            if S_candidate.bit_count() > S.bit_count():
                                S = S_candidate
                                improved = True
                            else:
                                # If we couldn't improve with kicks, break to avoid infinite loop
                                # or try a different random kick
                                pass
                                
        return S

    def get_base3_set():
        """Generate the base-3 3-AP-free set (digits 0, 2) in [1, n]."""
        S = 0
        for x in range(1, n + 1):
            valid = True
            num = x
            while num > 0:
                if num % 3 == 1:
                    valid = False
                    break
                num //= 3
            if valid:
                S |= (1 << (x - 1))
        return S

    best_S = 0
    best_size = 0
    
    # 1. Start with Base-3 set and expand
    S_base = get_base3_set()
    S_base = saturate(S_base)
    S_base = local_search(S_base)
    if S_base.bit_count() > best_size:
        best_S = S_base
        best_size = S_base.bit_count()
        
    # 2. Multiple Random Greedy Restarts
    while time.time() < deadline:
        order = list(range(n))
        random.shuffle(order)
        
        S = saturate(0, order)
        S = local_search(S)
        
        if S.bit_count() > best_size:
            best_S = S
            best_size = S.bit_count()
            
    # Convert bitset to sorted list
    result = []
    for i in range(n):
        if best_S & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
