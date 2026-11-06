import random

def construct():
    best_A = []
    best_B = []
    best_size = 0
    
    # Bitmask helpers
    # Range for sums: 2 to 512. Size 511.
    # Range for diffs: -255 to 255. Offset by 256. Index 0 is -256, 256 is 0, 511 is 255.
    # We only care about nonzero diffs.
    
    def try_construct():
        A = []
        B = []
        sums_A_mask = 0
        sums_B_mask = 0
        A_mask = 0
        B_mask = 0
        diffs_A_mask = 0
        diffs_B_mask = 0
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check A
            can_add_A = True
            
            # Sidon check for A:
            # 1. (x + a) in sums_A for any a in A
            # (A_mask << x) has bits at a+x.
            if (sums_A_mask & (A_mask << x)) != 0:
                can_add_A = False
            else:
                # 2. 2x in sums_A
                if (sums_A_mask & (1 << (2 * x))) != 0:
                    can_add_A = False
            
            # Diff check for A against B:
            # Need (x - a) not in diffs_B for any a in A
            if can_add_A and A:
                for a in A:
                    d = x - a
                    # d is in [-255, 255]
                    # Check bit at d + 256 in diffs_B_mask
                    if (diffs_B_mask >> (d + 256)) & 1:
                        can_add_A = False
                        break
            
            # Check B
            can_add_B = True
            
            # Sidon check for B
            if (sums_B_mask & (B_mask << x)) != 0:
                can_add_B = False
            else:
                if (sums_B_mask & (1 << (2 * x))) != 0:
                    can_add_B = False
            
            # Diff check for B against A
            if can_add_B and B:
                for b in B:
                    d = x - b
                    if (diffs_A_mask >> (d + 256)) & 1:
                        can_add_B = False
                        break
            
            # Decision
            if can_add_A and can_add_B:
                # Balance
                if len(A) <= len(B):
                    # Add to A
                    A.append(x)
                    A_mask |= (1 << x)
                    # Update sums_A
                    # New sums: x+a for a in A[:-1], and 2x
                    # We can update mask by ORing (A_mask_old << x) and (1 << 2x)
                    # But A_mask is updated. We need old A_mask.
                    # Let's do it before updating A_mask?
                    # We checked using old A_mask.
                    # So:
                    new_sums = (A_mask << x) | (1 << (2 * x))
                    sums_A_mask |= new_sums
                    # Update diffs_A
                    for a in A[:-1]:
                        d1 = x - a
                        d2 = a - x
                        diffs_A_mask |= (1 << (d1 + 256))
                        diffs_A_mask |= (1 << (d2 + 256))
                else:
                    # Add to B
                    B.append(x)
                    B_mask |= (1 << x)
                    new_sums = (B_mask << x) | (1 << (2 * x))
                    sums_B_mask |= new_sums
                    for b in B[:-1]:
                        d1 = x - b
                        d2 = b - x
                        diffs_B_mask |= (1 << (d1 + 256))
                        diffs_B_mask |= (1 << (d2 + 256))
            elif can_add_A:
                A.append(x)
                A_mask |= (1 << x)
                new_sums = (A_mask << x) | (1 << (2 * x))
                sums_A_mask |= new_sums
                for a in A[:-1]:
                    d1 = x - a
                    d2 = a - x
                    diffs_A_mask |= (1 << (d1 + 256))
                    diffs_A_mask |= (1 << (d2 + 256))
            elif can_add_B:
                B.append(x)
                B_mask |= (1 << x)
                new_sums = (B_mask << x) | (1 << (2 * x))
                sums_B_mask |= new_sums
                for b in B[:-1]:
                    d1 = x - b
                    d2 = b - x
                    diffs_B_mask |= (1 << (d1 + 256))
                    diffs_B_mask |= (1 << (d2 + 256))
        
        return len(A) + len(B), A, B

    # Run multiple restarts
    # Estimate time per restart.
    # 256 candidates.
    # Inner loops are small.
    # Should be fast.
    
    num_restarts = 5000
    for i in range(num_restarts):
        size, A, B = try_construct()
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
            if best_size >= 32:
                # Can't do better than 32 theoretically?
                # Let's break early if we hit 32 to save time?
                # Or continue to see if we can get lucky?
                # No, 32 is the bound.
                break

    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
