import itertools
import time

def construct(n):
    if n <= 1:
        return "1"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n+1)))
    num_perms = len(perms)
    
    # Map permutation to index for faster lookup? 
    # Or just use the tuples.
    
    # Standard order generation
    # The standard order is the order in which (n-1)-perms appear in S_{n-1}
    # We can simulate the standard construction to get the order.
    
    # Function to get standard order of perms for n
    def get_standard_order(n):
        if n == 1:
            return [(1,)]
        
        # Get S_{n-1}
        word = [1]
        for m in range(2, n):
            perms_prev, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_prev.append(window)
            
            new = []
            for p in perms_prev:
                block = list(p) + [m] + list(p)
                # Find overlap
                overlap = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if new[len(new)-o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        
        # Now extract the order of (n-1) perms from S_{n-1}
        order = []
        seen = set()
        for i in range(len(word) - (n-1) + 1):
            window = tuple(word[i:i + n-1])
            if window not in seen:
                seen.add(window)
                order.append(window)
        return order

    std_order = get_standard_order(n)
    
    # Convert to indices or just keep tuples
    # We will work with the list of tuples
    
    # Precompute overlap function
    def overlap(a, b):
        # a, b are tuples of length n-1
        # max overlap is n-2
        max_ov = n - 2
        for k in range(max_ov, -1, -1):
            if a[-k:] == b[:k]:
                return k
        return 0

    # Initialize order
    order = list(std_order)
    
    # Local search: adjacent swaps
    # Max time 110s
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < 110:
        improved = False
        for i in range(len(order) - 1):
            if (time.time() - start_time) > 110:
                break
            A = order[i-1] if i > 0 else None
            B = order[i]
            C = order[i+1]
            D = order[i+2] if i+2 < len(order) else None
            
            # Current cost: ov(A,B) + ov(B,C)
            # New cost: ov(A,C) + ov(C,B)
            
            cost_curr = 0
            if A is not None:
                cost_curr += overlap(A, B)
            cost_curr += overlap(B, C)
            
            cost_new = 0
            if A is not None:
                cost_new += overlap(A, C)
            cost_new += overlap(C, B)
            
            if cost_new > cost_curr:
                order[i], order[i+1] = order[i+1], order[i]
                improved = True
    
    # Construct the word
    if n == 1:
        return "1"
        
    word = []
    for i, p in enumerate(order):
        block = list(p) + [n] + list(p)
        if i == 0:
            word = block
        else:
            # Overlap with previous
            prev_p = order[i-1]
            # The previous block ends with prev_p
            # The current block starts with p
            # So we need overlap between prev_p and p
            ov = overlap(prev_p, p)
            # But wait, the previous block is prev_p n prev_p.
            # It ends with prev_p.
            # The current block is p n p.
            # It starts with p.
            # So the overlap is indeed overlap(prev_p, p).
            word += block[ov:]
            
    return "".join(map(str, word))
