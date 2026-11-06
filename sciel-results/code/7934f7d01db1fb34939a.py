import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # All 4-subsets of [1,16]
    # Represent as bitmasks
    all_sets = []
    mask_to_idx = {}
    idx_to_mask = {}
    
    # Generate all 4-combinations of 16 elements
    # Elements are 0..15 corresponding to 1..16
    for c in combinations(range(16), 4):
        mask = 0
        for x in c:
            mask |= (1 << x)
        idx = len(all_sets)
        all_sets.append(mask)
        mask_to_idx[mask] = idx
        idx_to_mask[idx] = mask
        
    n_masks = len(all_sets) # 1820
    full_mask = (1 << 16) - 1
    
    def popcount(x):
        return bin(x).count("1")
        
    def get_bits(mask):
        bits = []
        m = mask
        i = 0
        while m:
            if m & 1:
                bits.append(i)
            m >>= 1
            i += 1
        return bits

    def get_forbidden_indices(a_mask, b_mask):
        """
        Returns list of indices of masks C such that A, B, C form a sunflower.
        Condition: A&b == A&C == B&C
        Let U = A&B.
        C must contain U.
        C must not contain (A^B) \ U? No.
        A^B is symmetric difference.
        C & (A^B) must be 0?
        If C & (A^B) != 0, then C shares an element with A but not B, or B but not A.
        Let x in C & (A \ B). Then x in A, x not in B.
        A&C contains x. B&C does not contain x.
        So A&C != B&C.
        Thus C & (A \ B) must be 0 and C & (B \ A) must be 0.
        So C & (A ^ B) must be 0.
        
        So C is a subset of [16] \ (A^B).
        And C must contain U = A&B.
        
        Let V = [16] \ (A^B).
        We need 4-subsets of V that contain U.
        """
        u = a_mask & b_mask
        d = a_mask ^ b_mask
        v = full_mask ^ d # Elements not in symmetric difference
        
        # We need to choose 4 - popcount(u) elements from v \ u
        k_u = popcount(u)
        if k_u > 4:
            return []
            
        target_size = 4 - k_u
        v_minus_u = v ^ u # Elements in V but not in U
        
        if target_size == 0:
            # Only one choice: U itself (if |U|==4)
            # But A!=B so |U|<4. If |U|==4, A=B, impossible.
            c_mask = u
            if c_mask in mask_to_idx:
                return [mask_to_idx[c_mask]]
            return []
            
        bits = get_bits(v_minus_u)
        forbidden = []
        # Generate combinations of bits of size target_size
        for comb in combinations(bits, target_size):
            comb_mask = 0
            for b in comb:
                comb_mask |= (1 << b)
            c_mask = u | comb_mask
            if c_mask in mask_to_idx:
                forbidden.append(mask_to_idx[c_mask])
        return forbidden

    best_size = 0
    best_family = []
    
    random.seed(42)
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        shuffled_indices = list(range(n_masks))
        random.shuffle(shuffled_indices)
        
        blocked = [False] * n_masks
        family = []
        
        for idx in shuffled_indices:
            if blocked[idx]:
                continue
            
            # Check if adding idx creates a sunflower with any pair in family
            # This is handled by the 'blocked' array logic if we update it correctly.
            # However, we must ensure that idx is not blocked by pairs formed *before* it was added.
            # The 'blocked' array is updated when a new set is added.
            # So if idx is not blocked, it is safe to add.
            
            family.append(idx)
            
            # Update blocked array for all pairs involving the new set
            a_mask = idx_to_mask[idx]
            for j in range(len(family) - 1): # All previous sets
                b_idx = family[j]
                b_mask = idx_to_mask[b_idx]
                
                forb_ids = get_forbidden_indices(a_mask, b_mask)
                for fid in forb_ids:
                    blocked[fid] = True
            
            # Optional: Early exit if family size is small and time is up? No.
            # If we hit max possible, break?
            if len(family) >= 250:
                break
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
        # If we found a very large family, we can try to optimize further or break
        if best_size >= 245:
             # Try a few more restarts to see if we can hit 250
             pass

    # Convert best_family indices back to sorted lists of 1..16
    result = []
    for idx in best_family:
        mask = idx_to_mask[idx]
        s = []
        for i in range(16):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result
