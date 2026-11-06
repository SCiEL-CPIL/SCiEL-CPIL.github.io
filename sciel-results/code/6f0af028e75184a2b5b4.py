import numpy as np
import random
import time

def construct():
    n = 256
    # Masks are 8 uint64s (512 bits)
    # Sum indices: 2..512 -> 0..510 (shift by 2? or just use index directly if we allocate 513 bits? 
    # Max sum 512. Index 512 needs bit 512. 
    # 512 bits is 8 * 64. Index 511 is the highest bit in the last uint64.
    # So we need indices 0..511.
    # Sum $s$ maps to bit $s-2$? 
    # Min sum 2. Max sum 512.
    # Range 2..512 has 511 values.
    # Let's map sum $S$ to bit $S-2$.
    # Bit 0 corresponds to sum 2.
    # Bit 510 corresponds to sum 512.
    # This fits in 511 bits.
    
    # Diff indices: -255..255.
    # Map diff $d$ to bit $d+255$.
    # Min diff -255 -> bit 0.
    # Max diff 255 -> bit 510.
    # This fits in 511 bits.
    
    # We use 8 uint64s to be safe.
    
    def bit_set(mask, idx):
        mask[idx // 64] |= (1 << (idx % 64))
        
    def bit_check(mask, idx):
        return (mask[idx // 64] & (1 << (idx % 64))) != 0
        
    def mask_and(m1, m2):
        return m1 & m2
        
    def mask_or(m1, m2):
        return m1 | m2

    def try_add(x, S, sum_mask, diff_mask_self, diff_mask_other):
        """
        Check if x can be added to S.
        S: list of elements
        sum_mask, diff_mask_self, diff_mask_other: np.array of 8 uint64
        Returns (new_S, new_sum_mask, new_diff_mask) or None
        """
        # Calculate new sums and diffs
        # We need to check collisions with existing masks.
        # New sums: 2*x, x+s for s in S
        # New diffs: x-s, s-x for s in S
        
        # To check collision, we can construct the mask of new bits and AND with existing.
        # Constructing a temporary mask for each x is O(|S|) but with NumPy ops.
        # Alternatively, just iterate and check bits.
        
        # Let's try iterating and checking bits directly on the NumPy array.
        # This avoids creating temporary arrays.
        
        # Check 2*x
        sum_idx = 2 * x - 2
        if bit_check(sum_mask, sum_idx):
            return None
            
        # Check diffs for 2*x? No, diff is 0. We only care about non-zero diffs for disjointness?
        # "nonzero difference sets must be disjoint".
        # So diff 0 is ignored for the disjointness constraint between A and B.
        # But within A, diff 0 is unique (only one pair a,a gives 0? No, a-a=0 for all a. 
        # Sidon condition: pair sums unique. 
        # Difference condition: The set of non-zero differences of A must be disjoint from the set of non-zero differences of B.
        # It does NOT say A's non-zero differences must be unique. 
        # Wait, "Sidon set" definition: all pair sums are unique.
        # This implies all non-zero differences are unique in magnitude?
        # If $a+b = c+d$, then $a-c = d-b$.
        # If we have distinct pairs $(a,b)$ and $(c,d)$ with same sum, it's a collision.
        # Standard Sidon: $x_1+x_2 = x_3+x_4 \implies \{x_1,x_2\} = \{x_3,x_4\}$.
        # This is equivalent to: all non-zero differences $x_i - x_j$ are distinct in absolute value?
        # No. $x_1 - x_2 = x_3 - x_4 \implies x_1 + x_4 = x_2 + x_3$.
        # So yes, unique pair sums implies unique non-zero differences (up to sign).
        # So within a set, non-zero diffs must be unique.
        
        # So for A:
        # 1. Sum $2x$ must not be in $sum\_mask\_A$.
        # 2. For each $s \in A$:
        #    a. Sum $x+s$ must not be in $sum\_mask\_A$.
        #    b. Diff $x-s$ (and $s-x$) must not be in $diff\_mask\_A$ (non-zero).
        #    c. Diff $x-s$ (and $s-x$) must not be in $diff\_mask\_B$ (non-zero).
        
        for s in S:
            # Sum check
            sum_idx = x + s - 2
            if bit_check(sum_mask, sum_idx):
                return None
                
            # Diff checks
            d = x - s
            if d == 0:
                continue # x == s, but x is new, s is in S, so x != s if we check membership first?
                # We should check if x is already in S.
                # If x is in S, we shouldn't add it again.
                # The problem says "subsets", so no duplicates.
                # So x must not be in S.
                
            d_idx = d + 255
            if bit_check(diff_mask_self, d_idx):
                return None
            if bit_check(diff_mask_other, d_idx):
                return None
                
            d_idx_neg = -d + 255
            if bit_check(diff_mask_self, d_idx_neg):
                return None
            if bit_check(diff_mask_other, d_idx_neg):
                return None

        # If we reach here, it's safe to add.
        # Update masks
        new_sum_mask = sum_mask.copy()
        new_diff_mask = diff_mask_self.copy()
        
        # Set bits for 2*x
        sum_idx = 2 * x - 2
        new_sum_mask[sum_idx // 64] |= (1 << (sum_idx % 64))
        
        for s in S:
            sum_idx = x + s - 2
            new_sum_mask[sum_idx // 64] |= (1 << (sum_idx % 64))
            
            d = x - s
            d_idx = d + 255
            new_diff_mask[d_idx // 64] |= (1 << (d_idx % 64))
            d_idx_neg = -d + 255
            new_diff_mask[d_idx_neg // 64] |= (1 << (d_idx_neg % 64))
            
        return S + [x], new_sum_mask, new_diff_mask

    def remove(x, S, sum_mask, diff_mask_self):
        """
        Remove x from S.
        Returns (new_S, new_sum_mask, new_diff_mask)
        """
        new_S = [s for s in S if s != x]
        new_sum_mask = sum_mask.copy()
        new_diff_mask = diff_mask_self.copy()
        
        # Clear bits for 2*x
        sum_idx = 2 * x - 2
        new_sum_mask[sum_idx // 64] &= ~(1 << (sum_idx % 64))
        
        for s in new_S:
            sum_idx = x + s - 2
            new_sum_mask[sum_idx // 64] &= ~(1 << (sum_idx % 64))
            
            d = x - s
            d_idx = d + 255
            new_diff_mask[d_idx // 64] &= ~(1 << (d_idx % 64))
            d_idx_neg = -d + 255
            new_diff_mask[d_idx_neg // 64] &= ~(1 << (d_idx_neg % 64))
            
        return new_S, new_sum_mask, new_diff_mask

    best_size = 0
    best_A = []
    best_B = []
    
    start_time = time.time()
    max_time = 55.0
    
    candidates = list(range(1, n + 1))
    
    while time.time() - start_time < max_time:
        random.shuffle(candidates)
        
        A = []
        B = []
        sum_mask_A = np.zeros(8, dtype=np.uint64)
        diff_mask_A = np.zeros(8, dtype=np.uint64)
        sum_mask_B = np.zeros(8, dtype=np.uint64)
        diff_mask_B = np.zeros(8, dtype=np.uint64)
        
        # Greedy Phase
        for x in candidates:
            # Try A
            if x not in A:
                res = try_add(x, A, sum_mask_A, diff_mask_A, diff_mask_B)
                if res:
                    A, sum_mask_A, diff_mask_A = res
                    
            # Try B
            if x not in B:
                res = try_add(x, B, sum_mask_B, diff_mask_B, diff_mask_A)
                if res:
                    B, sum_mask_B, diff_mask_B = res
                    
        total_size = len(A) + len(B)
        
        # Local Search Phase
        # Try to improve by kicking out 1 and adding 2
        # Or just try to add more from the remaining candidates?
        # The greedy pass already went through all candidates.
        # We can try to swap elements to free up space.
        
        # Simple local search: try to replace one element in A with two from the pool?
        # This is complex. Let's try a simpler "restart with bias".
        # Or just keep restarting.
        
        if total_size > best_size:
            best_size = total_size
            best_A = A[:]
            best_B = B[:]
            
    return [best_A, best_B]
