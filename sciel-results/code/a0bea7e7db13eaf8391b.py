"""Two Sidon sets with disjoint difference sets – Fast bitset-based search."""

import time
import random

N = 256
MAX_SIZE = 64
DEADLINE = 55.0  # Safety margin before 90s limit

# Precompute bitsets for sums and diffs? No, sums go up to 512, diffs up to 255.
# We can use integers as bitsets.
# Sum bitset: bit s is set if sum s is used. Range 2..512.
# Diff bitset: bit d is set if diff d is used. Range -255..255.
# We can map diff d to index d + 255. Range 0..510.

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_a, best_b = [], []
    best_size = 0
    
    # Helper to check if adding x to set S (with sum_bits, diff_bits) is valid
    # Given existing elements list S, sum_bits (int), diff_bits (int)
    # other_diff_bits (int)
    # Returns True if valid
    
    def can_add(x, S, sum_bits, diff_bits, other_diff_bits):
        # Check sum collisions: x+s for s in S, and x+x
        # If (x+s) collides with an existing sum, invalid.
        # Existing sums are in sum_bits.
        # New sums are x+s for s in S, and 2x.
        # Collision if (sum_bits >> (x+s)) & 1 or (sum_bits >> (2x)) & 1
        # Also need to check if new sums collide with each other?
        # If x+s1 == x+s2 => s1==s2, unique.
        # If x+s == 2x => s=x, but x not in S yet.
        # So we just need to check against existing sum_bits.
        
        for s in S:
            if (sum_bits >> (x + s)) & 1:
                return False
        if (sum_bits >> (2 * x)) & 1:
            return False
            
        # Check diff collisions with other set
        # New diffs: x-s for s in S, s-x for s in S.
        # Collision if (other_diff_bits >> (x-s+255)) & 1 or (other_diff_bits >> (s-x+255)) & 1
        for s in S:
            d1 = x - s + 255
            d2 = s - x + 255
            if (other_diff_bits >> d1) & 1 or (other_diff_bits >> d2) & 1:
                return False
                
        return True

    def greedy_build(order_a, order_b):
        S_a = []
        S_b = []
        sum_bits_a = 0
        sum_bits_b = 0
        diff_bits_a = 0
        diff_bits_b = 0
        
        # Interleaved greedy? Or sequential?
        # Let's try sequential A then B first, but fast.
        
        for x in order_a:
            if len(S_a) >= MAX_SIZE:
                break
            if can_add(x, S_a, sum_bits_a, diff_bits_a, diff_bits_b):
                S_a.append(x)
                # Update bits
                for s in S_a[:-1]:
                    sum_bits_a |= (1 << (x + s))
                    diff_bits_a |= (1 << (x - s + 255))
                    diff_bits_a |= (1 << (s - x + 255))
                sum_bits_a |= (1 << (2 * x))
                
        for x in order_b:
            if len(S_b) >= MAX_SIZE:
                break
            if can_add(x, S_b, sum_bits_b, diff_bits_b, diff_bits_a):
                S_b.append(x)
                for s in S_b[:-1]:
                    sum_bits_b |= (1 << (x + s))
                    diff_bits_b |= (1 << (x - s + 255))
                    diff_bits_b |= (1 << (s - x + 255))
                sum_bits_b |= (1 << (2 * x))
                
        return S_a, S_b

    def local_improve(S_a, S_b):
        # Rebuild bits
        sum_bits_a = 0
        sum_bits_b = 0
        diff_bits_a = 0
        diff_bits_b = 0
        
        for a in S_a:
            for b in S_a:
                if a <= b:
                    sum_bits_a |= (1 << (a + b))
            for b in S_a:
                if a != b:
                    diff_bits_a |= (1 << (a - b + 255))
                    
        for a in S_b:
            for b in S_b:
                if a <= b:
                    sum_bits_b |= (1 << (a + b))
            for b in S_b:
                if a != b:
                    diff_bits_b |= (1 << (a - b + 255))
        
        improved = True
        while improved:
            improved = False
            # Try to add to A
            for x in range(1, N + 1):
                if x in S_a:
                    continue
                if can_add(x, S_a, sum_bits_a, diff_bits_a, diff_bits_b):
                    S_a.append(x)
                    for s in S_a[:-1]:
                        sum_bits_a |= (1 << (x + s))
                        diff_bits_a |= (1 << (x - s + 255))
                        diff_bits_a |= (1 << (s - x + 255))
                    sum_bits_a |= (1 << (2 * x))
                    improved = True
                    break
            if improved:
                continue
                
            # Try to add to B
            for x in range(1, N + 1):
                if x in S_b:
                    continue
                if can_add(x, S_b, sum_bits_b, diff_bits_b, diff_bits_a):
                    S_b.append(x)
                    for s in S_b[:-1]:
                        sum_bits_b |= (1 << (x + s))
                        diff_bits_b |= (1 << (x - s + 255))
                        diff_bits_b |= (1 << (s - x + 255))
                    sum_bits_b |= (1 << (2 * x))
                    improved = True
                    break
            if improved:
                continue
                
            # Try swap in A: remove a, add x
            # This is complex to update bits efficiently without full rebuild.
            # Given time, let's stick to additions which are faster to implement correctly.
            # If no additions, stop.
            
        return S_a, S_b

    random.seed(42)
    
    while time.time() < deadline - 2.0:
        order_a = list(range(1, N + 1))
        random.shuffle(order_a)
        order_b = list(range(1, N + 1))
        random.shuffle(order_b)
        
        a, b = greedy_build(order_a, order_b)
        current_size = len(a) + len(b)
        
        if current_size > best_size:
            best_size = current_size
            best_a, best_b = a, b
            
        if time.time() < deadline - 3.0:
            a_imp, b_imp = local_improve(a, b)
            imp_size = len(a_imp) + len(b_imp)
            if imp_size > best_size:
                best_size = imp_size
                best_a, best_b = a_imp, b_imp
                
    if not best_a:
        best_a = [1]
    if not best_b:
        best_b = [2]
        
    return [best_a, best_b]
