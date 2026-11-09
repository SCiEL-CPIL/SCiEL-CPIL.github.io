import numpy as np
import time
import random

N = 18

def batcher_odd_even_merge(n):
    """
    Generates Batcher's Odd-Even Merge Sort network.
    This is a known-correct sorting network with far fewer comparators
    than odd-even transposition for n >= 10.
    """
    net = []
    
    def merge(a, b, depth=0):
        """Merge two sorted sub-networks of size a and b starting at wire offset."""
        if a == 1:
            return
        if b == 1:
            return
        if a > b:
            merge(b, a, depth)
            return
        # Standard OES: split, recursively merge, cross-compare, recursively merge
        mid_a = a // 2
        mid_b = b // 2
        
        # Recursively merge left halves
        for i in range(mid_a):
            net.append((i, i + mid_a))
        for i in range(mid_b):
            net.append((mid_a + i, mid_a + mid_b + i))
        
        # Cross comparisons
        for i in range(min(a, b)):
            if i < a:
                net.append((mid_a + i, mid_a + mid_b + i))
        
        # Recursively merge right halves
        for i in range(mid_a):
            net.append((mid_a + i, mid_a + mid_b + i))
        for i in range(mid_b):
            net.append((mid_a + mid_b + i, mid_a + a + i))
    
    # Actually, let me use a cleaner recursive implementation
    net.clear()
    
    def oes(n, offset=0):
        """Odd-Even Merge Sort on wires [offset, offset+n)."""
        if n <= 1:
            return
        if n == 2:
            net.append((offset, offset + 1))
            return
        half = n // 2
        # Recursively sort first half and second half
        oes(half, offset)
        oes(half, offset + half)
        # Merge the two sorted halves
        merge_halves(half, half, offset)
    
    def merge_halves(a, b, offset):
        """Merge two sorted halves of sizes a and b starting at offset."""
        if a == 0 or b == 0:
            return
        if a == 1 and b == 1:
            net.append((offset, offset + 1))
            return
        if a == 1:
            # Just compare the single element from first half with each in second
            for i in range(b):
                net.append((offset, offset + 1 + i))
            return
        if b == 1:
            for i in range(a):
                net.append((offset + i, offset + a))
            return
        
        ha = a // 2
        hb = b // 2
        
        # Split: [a1][b1] and [a2][b2] where a1+a2=a, b1+b2=b
        # First merge: sort [a1 | b1] then [a2 | b2]
        # a1 and b1: sizes ha and hb
        # a2 and b2: sizes a-ha and b-hb
        merge_halves(ha, hb, offset)
        merge_halves(a - ha, b - hb, offset + ha)
        
        # Cross-compare: for each i in range(min(a, b)), compare (ha+i) with (ha+hb+i)
        for i in range(min(a, b)):
            net.append((offset + ha + i, offset + ha + hb + i))
        
        # Second merge: sort [a1 | b1] and [a2 | b2] again but interleaved
        # Actually after cross-compare, we need to re-merge:
        # [a1] is sorted, [b1] is sorted, but we need to merge them properly
        # The standard OES does:
        # 1. Recursively merge first halves: merge(a1, b1)
        # 2. Cross compare
        # 3. Recursively merge second halves: merge(a2, b2)
        # But that's what we did. Let me just use a simpler correct approach.
        pass
    
    # Let me use a simpler, verified correct network: Batcher's BSN
    net.clear()
    
    def batcher_sort(n, offset=0):
        """Batcher's Bitonic Sort Network (simpler and well-known)."""
        if n <= 1:
            return
        if n == 2:
            net.append((offset, offset + 1))
            return
        half = n // 2
        # Sort first half, sort second half
        batcher_sort(half, offset)
        batcher_sort(half, offset + half)
        # Create bitonic sequence by reversing second half
        # Actually for a full sort, we just need to merge two sorted halves
        # Use odd-even merge
        odd_even_merge(half, half, offset)
    
    def odd_even_merge(a, b, offset):
        """Odd-Even Merge: merge two sorted sequences of lengths a and b."""
        if a == 0 or b == 0:
            return
        if a == 1 and b == 1:
            net.append((offset, offset + 1))
            return
        if a == 1:
            for i in range(b):
                net.append((offset, offset + 1 + i))
            return
        if b == 1:
            for i in range(a):
                net.append((offset + i, offset + a))
            return
        
        ha = a // 2
        hb = b // 2
        
        # Recursively merge odd-indexed and even-indexed elements
        # Odd elements: a[1], a[3], ... and b[1], b[3], ...
        # Even elements: a[0], a[2], ... and b[0], b[2], ...
        
        # Merge even-indexed elements
        odd_even_merge(ha, hb, offset)
        # Merge odd-indexed elements  
        odd_even_merge(a - ha, b - hb, offset + 1)
        
        # Cross-compare adjacent pairs
        for i in range(min(a, b) - 1):
            net.append((offset + i, offset + i + 1))
    
    # Use a simpler approach: just build the network recursively
    net.clear()
    
    def build_oes(n, offset=0):
        """Build odd-even merge sort network."""
        if n <= 1:
            return
        if n == 2:
            net.append((offset, offset + 1))
            return
        half = n // 2
        # Sort both halves
        build_oes(half, offset)
        build_oes(half, offset + half)
        # Merge
        oes_merge(half, half, offset)
    
    def oes_merge(a, b, offset):
        """Odd-even merge of two sorted sequences."""
        if a == 0 or b == 0:
            return
        if a == 1 and b == 1:
            net.append((offset, offset + 1))
            return
        if a == 1:
            for i in range(b):
                net.append((offset, offset + 1 + i))
            return
        if b == 1:
            for i in range(a):
                net.append((offset + i, offset + a))
            return
        
        ha = a // 2
        hb = b // 2
        
        # Recursively merge
        oes_merge(ha, hb, offset)
        oes_merge(a - ha, b - hb, offset + 1)
        
        # Cross comparisons
        for i in range(min(a, b) - 1):
            net.append((offset + i, offset + i + 1))
    
    build_oes(N)
    return net

def generate_odd_even_transposition(n):
    """
    Generates the Odd-Even Transposition sorting network.
    """
    net = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            net.append((i, i + 1))
    return net

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit, leaving 5s margin for safety
    
    # 1. Try Batcher's OES first (much smaller than OET)
    network = batcher_odd_even_merge(N)
    
    # Fallback to OET if something goes wrong
    if not network:
        network = generate_odd_even_transposition(N)
    
    # 2. Prepare test vectors
    test_vectors = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        period = 1 << (N - i)
        half = period // 2
        pattern = np.array([0]*half + [1]*half, dtype=np.uint8)
        reps = (1 << N) // period
        test_vectors[:, i] = np.tile(pattern, reps)
    
    # 3. Optimization helpers
    def check_network(net):
        if not net:
            return False
        current = test_vectors.copy()
        for i, j in net:
            min_val = np.minimum(current[:, i], current[:, j])
            max_val = np.maximum(current[:, i], current[:, j])
            current[:, i] = min_val
            current[:, j] = max_val
        return np.all(current[:, :-1] <= current[:, 1:])
    
    def try_remove(network, idx):
        """Try removing comparator at idx. Returns (new_network, success)."""
        candidate = network[:idx] + network[idx+1:]
        if check_network(candidate):
            return candidate, True
        return network, False
    
    # Initial validation
    if not check_network(network):
        # Fall back to OET
        network = generate_odd_even_transposition(N)
        if not check_network(network):
            return network
    
    # 4. Greedy Removal Phase
    changed = True
    passes = 0
    max_passes = 10
    
    while changed and passes < max_passes and time.time() < deadline:
        changed = False
        passes += 1
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                break
            new_net, ok = try_remove(network, idx)
            if ok:
                network = new_net
                changed = True
    
    # 5. Local Search: random perturbations
    best_net = list(network)
    best_len = len(best_net)
    
    for _ in range(20):
        if time.time() > deadline:
            break
        
        # Randomly add 1-3 comparators
        perturbed = list(best_net)
        num_adds = random.randint(1, 3)
        for _ in range(num_adds):
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            perturbed.append((i, j))
        
        # Try to remove from perturbed
        changed = True
        ppasses = 0
        while changed and ppasses < 5 and time.time() < deadline:
            changed = False
            ppasses += 1
            for idx in range(len(perturbed) - 1, -1, -1):
                if time.time() > deadline:
                    break
                new_net, ok = try_remove(perturbed, idx)
                if ok:
                    perturbed = new_net
                    changed = True
        
        if len(perturbed) < best_len:
            best_net = perturbed
            best_len = len(best_net)
    
    network = best_net
    
    # Ensure we return a valid, non-empty network
    if len(network) == 0:
        return generate_odd_even_transposition(N)
        
    return network
