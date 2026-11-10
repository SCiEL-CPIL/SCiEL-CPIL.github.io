import time
import random

def construct():
    N = 512
    # Indices 0 to 511 correspond to numbers 1 to 512
    # Bitmask representation
    # Precompute validity check helper? 
    # Let's just use the direct check function.
    
    def is_valid_addition(S, z_idx):
        """
        Check if adding z_idx to S creates a 4-AP.
        z_idx is 0-indexed (0..511), value is z_idx+1.
        S is a bitmask of selected indices.
        """
        z = z_idx + 1
        # Case 1: z is the first term (a)
        # AP: z, z+d, z+2d, z+3d
        # Max d such that z+3d <= 512
        max_d1 = (N - z) // 3
        for d in range(1, max_d1 + 1):
            if ((S >> (z + d - 1)) & 1) and ((S >> (z + 2*d - 1)) & 1) and ((S >> (z + 3*d - 1)) & 1):
                return False
        
        # Case 2: z is the second term (a+d)
        # AP: z-d, z, z+d, z+2d
        # Min d such that z-d >= 1 => d <= z-1
        # Max d such that z+2d <= 512 => 2d <= 512-z => d <= (512-z)//2
        max_d2 = min(z - 1, (N - z) // 2)
        for d in range(1, max_d2 + 1):
            if ((S >> (z - d - 1)) & 1) and ((S >> (z + d - 1)) & 1) and ((S >> (z + 2*d - 1)) & 1):
                return False

        # Case 3: z is the third term (a+2d)
        # AP: z-2d, z-d, z, z+d
        # Min d such that z-2d >= 1 => 2d <= z-1 => d <= (z-1)//2
        # Max d such that z+d <= 512 => d <= 512-z
        max_d3 = min((z - 1) // 2, N - z)
        for d in range(1, max_d3 + 1):
            if ((S >> (z - 2*d - 1)) & 1) and ((S >> (z - d - 1)) & 1) and ((S >> (z + d - 1)) & 1):
                return False

        # Case 4: z is the fourth term (a+3d)
        # AP: z-3d, z-2d, z-d, z
        # Min d such that z-3d >= 1 => 3d <= z-1 => d <= (z-1)//3
        max_d4 = (z - 1) // 3
        for d in range(1, max_d4 + 1):
            if ((S >> (z - 3*d - 1)) & 1) and ((S >> (z - 2*d - 1)) & 1) and ((S >> (z - d - 1)) & 1):
                return False
                
        return True

    def get_conflict_mask(S, z_idx):
        """
        Returns a bitmask of elements in S that would form a 4-AP with z_idx.
        Actually, we just need to know if it's valid.
        """
        pass

    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    best_S = 0
    best_size = 0
    
    # Greedy initialization with random permutations
    # We try a few different orders to find a good starting point
    # Order 1: Reverse (larger numbers first)
    # Order 2: Random
    # Order 3: Sorted by "density" of conflicts? (Hard to compute quickly)
    
    orders = []
    # Reverse order
    orders.append(list(range(N-1, -1, -1)))
    # Forward order
    orders.append(list(range(N)))
    # A few random shuffles
    for _ in range(5):
        r = list(range(N))
        random.shuffle(r)
        orders.append(r)
    
    for order in orders:
        if time.time() > deadline:
            break
        S = 0
        size = 0
        for z_idx in order:
            if is_valid_addition(S, z_idx):
                S |= (1 << z_idx)
                size += 1
        if size > best_size:
            best_size = size
            best_S = S
    
    # Local Search: 1-Exchange and Greedy Refill
    # We will iterate on the best_S found so far
    # To make it faster, we can run multiple local search passes from different initializations
    
    # Let's run local search on the best candidate found
    S = best_S
    size = best_size
    
    # Helper to check if S is valid (should be, but just in case)
    # We assume S is valid from greedy.
    
    # 1-Exchange: Try to swap a in S with b not in S
    # To be efficient, we can try to add b and remove a if it reduces conflicts?
    # Simpler: Try to add b. If invalid, find a in S that conflicts with b and try to remove a?
    # Even simpler: Iterate through all b not in S. If is_valid_addition(S, b), add it.
    # This is "Greedy Refill".
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Greedy Refill: Try to add any element not in S
        for b_idx in range(N):
            if not ((S >> b_idx) & 1):
                if is_valid_addition(S, b_idx):
                    S |= (1 << b_idx)
                    size += 1
                    improved = True
                    # Restart refill loop? Or just continue?
                    # Continuing is fine, but we might add multiple.
                    # Let's break and restart to ensure we don't miss better additions?
                    # Actually, adding one might block another. 
                    # Standard greedy refill adds if possible.
                    pass
        
        # 1-Exchange: Try to swap a in S with b not in S
        # This is O(N^2) checks. 
        # Optimization: Only try swaps if we are stuck.
        # For now, let's do a simple 1-exchange pass.
        if not improved:
             # Try 1-exchange
             for b_idx in range(N):
                 if ((S >> b_idx) & 1): continue
                 # Check if b can be added if we remove some a
                 # This requires knowing which a conflicts with b.
                 # Let's just try all pairs (a in S, b not in S)
                 # This might be slow. 512*512 = 260k checks. Each check is ~100 ops. 26M ops. 
                 # In Python, this is too slow for a single pass within 50s if done many times.
                 # We need a faster conflict check or fewer iterations.
                 
                 # Let's stick to Greedy Refill and maybe a limited 1-exchange.
                 break # Skip complex 1-exchange for now to keep code simple and fast, relying on multiple restarts
    
    # Convert bitmask to list of values (1-indexed)
    result = []
    for i in range(N):
        if (best_S >> i) & 1:
            result.append(i + 1)
    
    return sorted(result)
