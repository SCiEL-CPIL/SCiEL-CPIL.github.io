import random
import time

# EVOLVE-BLOCK-START

def construct():
    N = 128
    deadline = time.time() + 55.0  # Safety margin
    
    # Precompute conflict masks
    # conflict_mask[i][j] is a bitmask of k (0-indexed) such that:
    # i < j < k (values i+1 < j+1 < k+1)
    # and (i+1)+(j+1) <= N, (i+1)+(k+1) <= N, (j+1)+(k+1) <= N
    # Note: We only care about k > j.
    # We store masks for 0-indexed positions.
    
    conflict_masks = [[0] * N for _ in range(N)]
    for i in range(N):
        vi = i + 1
        for j in range(i + 1, N):
            vj = j + 1
            sum_ij = vi + vj
            if sum_ij > N:
                break # Since k > j, sum_ik and sum_jk will be even larger
            # Find all k > j such that vi+vk <= N and vj+vk <= N
            # vi < vj, so vi+vk < vj+vk.
            # Condition: vk <= N - vj
            max_vk = N - vj
            # k is 0-indexed, so vk = k+1.
            # k+1 <= max_vk => k <= max_vk - 1
            max_k = max_vk - 1
            if max_k > j:
                # Set bits for k from j+1 to max_k
                # mask = ((1 << (max_k - j)) - 1) << (j + 1)
                # But we need to be careful with large shifts.
                # N=128, so 128 bits. Python ints handle this fine.
                mask = 0
                for k in range(j + 1, max_k + 1):
                    mask |= (1 << k)
                conflict_masks[i][j] = mask
            else:
                conflict_masks[i][j] = 0

    def is_valid(mask):
        # mask is an integer representing the set
        # Check for forbidden triples
        for i in range(N):
            if not (mask >> i) & 1:
                continue
            vi = i + 1
            for j in range(i + 1, N):
                if not (mask >> j) & 1:
                    continue
                vj = j + 1
                sum_ij = vi + vj
                if sum_ij > N:
                    break
                # Check if sum_ij is in mask
                if not (mask >> (sum_ij - 1)) & 1:
                    continue
                
                # Get potential k's
                # conflict_masks[i][j] has bits set for k > j where sums are in range
                potential_k_mask = conflict_masks[i][j] & mask
                if potential_k_mask == 0:
                    continue
                
                # Iterate through set bits in potential_k_mask
                pk = potential_k_mask
                while pk:
                    # Get lowest set bit
                    lsb = pk & (-pk)
                    k = lsb.bit_length() - 1
                    pk ^= lsb
                    
                    vk = k + 1
                    sum_ik = vi + vk
                    sum_jk = vj + vk
                    
                    # Check if sums are in mask
                    if (mask >> (sum_ik - 1)) & 1 and (mask >> (sum_jk - 1)) & 1:
                        return False
        return True

    def count_bits(mask):
        return bin(mask).count("1")

    best_size = 0
    best_mask = 0
    
    # Heuristic 1: Upper half
    mask_upper = 0
    for i in range(N // 2, N):
        mask_upper |= (1 << i)
    if is_valid(mask_upper):
        best_size = count_bits(mask_upper)
        best_mask = mask_upper

    # Heuristic 2: Randomized Greedy + Local Search
    # We run several restarts
    max_restarts = 200
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Randomize start: mix of upper and lower
        # Or just random shuffle of all elements
        candidates = list(range(N))
        random.shuffle(candidates)
        
        current_mask = 0
        # Greedy construction
        for idx in candidates:
            new_mask = current_mask | (1 << idx)
            if is_valid(new_mask):
                current_mask = new_mask
        
        # Local Search: Try to add missing elements
        # Then try swaps
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # Try adding
            for i in range(N):
                if not (current_mask >> i) & 1:
                    new_mask = current_mask | (1 << i)
                    if is_valid(new_mask):
                        current_mask = new_mask
                        improved = True
                        break # Restart adding loop after change
            
            # If no additions, try swaps: remove one, add another
            if not improved:
                for i in range(N):
                    if (current_mask >> i) & 1:
                        for j in range(N):
                            if not (current_mask >> j) & 1:
                                new_mask = (current_mask ^ (1 << i)) | (1 << j)
                                if is_valid(new_mask):
                                    current_mask = new_mask
                                    improved = True
                                    break
                        if improved:
                            break

        if count_bits(current_mask) > best_size:
            best_size = count_bits(current_mask)
            best_mask = current_mask

    # Convert best_mask to list of ints
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    
    return result

# EVOLVE-BLOCK-END
