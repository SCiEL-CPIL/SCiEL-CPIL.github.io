import time
import random
from itertools import combinations

def construct():
    n = 256
    start_time = time.time()
    deadline = 85.0  # 5s margin for 90s limit
    
    # Initial valid set: [171, 256]
    # Size 86
    base_S = list(range(171, n + 1))
    best_S = set(base_S)
    best_size = len(base_S)
    
    candidates = list(range(1, 171))
    
    # Helper to compute sums set for a given list S
    # Sums of b+c for b <= c in S
    def compute_sums(S_list):
        sums = set()
        L = len(S_list)
        for i in range(L):
            b = S_list[i]
            for j in range(i, L):
                c = S_list[j]
                sums.add(b + c)
        return sums
    
    # Initial sums for base_S
    base_sums = compute_sums(base_S)
    
    # DP-based approach: track residues modulo small integers
    # For each small modulus m (2, 3, 4, ..., 16), track which residues
    # are present in the current set. This allows quick conflict detection.
    
    def compute_residue_mask(S_list, m):
        """Return a bitmask of residues present modulo m"""
        mask = 0
        for x in S_list:
            mask |= (1 << (x % m))
        return mask
    
    def check_conflict_with_residues(x, current_S, base_sums, residue_masks):
        """
        Check if adding x would create a conflict.
        x must not divide any sum b+c where b,c are in current_S and b <= c.
        
        We use residue masks for small moduli to quickly rule out conflicts.
        """
        # Quick check: if x is large, few multiples to check
        # x divides b+c means b+c ≡ 0 (mod x)
        
        # For small x, use residue-based pruning
        # If x <= 16, we can check if any pair in current_S has sum ≡ 0 mod x
        
        if x <= 16:
            # Check if there exist b, c in current_S with b+c ≡ 0 (mod x)
            # This is equivalent to: there exist residues r1, r2 in current_S mod x
            # such that r1 + r2 ≡ 0 (mod x)
            # i.e., r2 ≡ -r1 (mod x)
            
            residues = set()
            for y in current_S:
                residues.add(y % x)
            
            for r in residues:
                target = (-r) % x
                if target in residues:
                    # Potential conflict: need to verify with actual sums
                    # But this is a necessary condition, not sufficient
                    # Still, if no such pair exists, x is definitely safe
                    pass
            
            # More precise check: iterate through multiples of x in sum range
            min_sum = 2 * min(current_S)
            max_sum = 2 * max(current_S)
            start_mult = (min_sum + x - 1) // x * x
            
            for m in range(start_mult, max_sum + 1, x):
                if m in base_sums:
                    return False
        
        # For larger x, direct check is fast enough
        min_sum = 2 * min(current_S)
        max_sum = 2 * max(current_S)
        start_mult = (min_sum + x - 1) // x * x
        
        for m in range(start_mult, max_sum + 1, x):
            if m in base_sums:
                return False
        
        return True
    
    # Greedy search with random restarts and DP-inspired pruning
    num_restarts = 0
    while time.time() < deadline:
        # Random permutation of candidates
        perm = candidates[:]
        random.shuffle(perm)
        
        current_S = base_S[:]
        current_sums = set(base_sums) # Copy the set
        
        # Precompute residue masks for small moduli
        residue_masks = {}
        for m in range(2, 17):
            residue_masks[m] = compute_residue_mask(current_S, m)
        
        for x in perm:
            # Check if x can be added
            can_add = check_conflict_with_residues(x, current_S, current_sums, residue_masks)
            
            if can_add:
                # Add x to current_S
                old_S = current_S[:]
                current_S.append(x)
                current_S.sort()
                
                # Update sums: new sums are x + y for all y in old_S
                for y in old_S:
                    current_sums.add(x + y)
                
                # Update residue masks
                for m in range(2, 17):
                    residue_masks[m] |= (1 << (x % m))
                
                # Update best if size increased
                if len(current_S) > best_size:
                    best_size = len(current_S)
                    best_S = set(current_S)
        
        num_restarts += 1
    
    return list(best_S)
