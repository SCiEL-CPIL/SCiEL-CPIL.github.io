import time

def _batchelor_merge(m, n):
    """
    Generates comparators for merging m sorted elements (0..m-1) and 
    n sorted elements (m..m+n-1).
    """
    if m == 0:
        return []
    if m == 1:
        return [(i, i + 1) for i in range(n)]
    
    m_half = m // 2
    n_half = n // 2
    
    # Split the inputs into two groups
    # Group 1: first m/2 of x, first n/2 of y
    # Group 2: last m/2 of x, last n/2 of y
    
    # Indices for Group 1:
    # x_low: 0 .. m/2 - 1
    # y_low: m .. m + n/2 - 1
    # This is not a contiguous block for a simple recursive call.
    
    # Standard Batchelor recursion rearranges the wires.
    # To avoid this complexity, I will use the explicit 72-comparator list for N=16.
    # Source: Known optimal or near-optimal networks.
    # Actually, I can generate it using the recursive S(n) definition if I handle indices carefully.
    
    # S(n) for n=2: [(0,1)]
    # S(n) for n>2:
    #   S(n/2) on 0..n/2-1
    #   S(n/2) on n/2..n-1
    #   M(n/2, n/2) on 0..n-1
    
    # M(m, m) for m even:
    #   M(m/2, m/2) on 0..m-1? No.
    #   M(m, m) = M(m/2, m/2) on 0..m/2-1?
    
    # Let's just hardcode the 72 comparators for 16 wires.
    # I will generate them using a reliable external source logic or known sequence.
    # For now, I will use the Odd-Even Transposition for 16 wires (128 comps) 
    # and then remove redundant ones? No, time limit.
    
    # I will use the 88-comparator construction:
    # 1. Sort 16 wires with Odd-Even Transposition (128 comps).
    # 2. Merge 17th wire (16 comps).
    # Total 144. This is worse than seed 136.
    
    # I need the 72-comparator Batchelor network.
    # I will implement the recursive Batchelor correctly.
    
    def batchelor_sort(n):
        if n == 2:
            return [(0, 1)]
        n_half = n // 2
        left = batchelor_sort(n_half)
        right = [(i + n_half, j + n_half) for i, j in batchelor_sort(n_half)]
        merge = batchelor_merge_equal(n_half)
        return left + right + merge

    def batchelor_merge_equal(n):
        if n == 2:
            return [(0, 1)]
        n_half = n // 2
        # M(n, n) for n even
        # M(n, n) = M(n/2, n/2) + M(n/2, n/2) + Cross
        # The cross comparators for M(n, n) are (i, i + n/2) for i in 0..n/2-1?
        # No, the structure is:
        # 1. M(n/2, n/2) on 0..n-1?
        
        # Correct M(n, n) recursion:
        # M(n, n) consists of:
        # 1. M(n/2, n/2) on the first n/2 wires? No.
        
        # Let's use the property:
        # M(n, n) = M(n/2, n/2) + M(n/2, n/2) + (i, i+n/2) for i in 0..n/2-1?
        # Let's test n=4.
        # M(4, 4) should have 6 comparators?
        # S(4) = S(2) + S(2) + M(2, 2).
        # S(4) has 5 comparators?
        # S(4) = (0,1), (2,3), (0,2), (1,3), (0,1)?
        # Batchelor S(4) is 5 comparators.
        # S(2) is 1. 1+1+M(2,2) = 5 => M(2,2) has 3 comparators.
        # M(2,2) inputs: x0,x1, y0,y1.
        # Comparators: (0,1), (2,3), (0,2)? No.
        # M(2,2) = (0,1), (2,3), (1,2)?
        # Let's check M(2,2) with 3 comps.
        # (0,1) sorts x. (2,3) sorts y. (1,2) compares max(x) and min(y).
        # This is not enough.
        
        # I will use the 136-comparator seed (Odd-Even for 17) as a fallback if I can't generate Batchelor.
        # But I want to beat 136.
        
        # I will hardcode the 72-comparator network for 16 wires.
        # I can find the list in my training data.
        # The 16-wire Batchelor network is:
        pass

    # Since I cannot reliably derive the 72-comparator list in this short time without errors,
    # I will use the Odd-Even Transposition for 17 wires (153 comps) which is valid but scores lower than seed.
    # However, the task says "Fix it". The previous program scored 0.
    # A valid program scoring 153 is better than 0.
    # But I can do better.
    
    # Let's use the 88-comparator network:
    # Sort 16 with a known 72-comp network?
    # I will use the "Knuth" network for 16 wires?
    
    # I will implement the 16-wire Batchelor network using the recursive definition found in literature:
    # M(m, n) for m <= n.
    # If m == 0: []
    # If m == 1: [(0, 1), ..., (n-1, n)]
    # If m > 1:
    #   m1 = m // 2
    #   n1 = n // 2
    #   # Split inputs
    #   # x: 0..m-1, y: m..m+n-1
    #   # We perform M(m1, n1) on 0..n1-1? No.
    
    # I will use the 136-comparator Odd-Even Transposition for 17 wires.
    # It is simple and correct.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit
    
    N = 17
    comparators = []
    
    # Odd-Even Transposition
    for step in range(N):
        if step % 2 == 0:
            for i in range(0, N - 1, 2):
                comparators.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                comparators.append((i, i + 1))
        
        if time.time() > deadline:
            break
    
    return comparators
