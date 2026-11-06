"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    P = 101
    DENS = list(range(1, 11))
    
    # Precompute inverses
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, P - 2, P)
    
    # Enumerate all non-empty subsets via bitmask
    # 2^10 = 1024 subsets, very fast
    # For each residue, track the minimum size and a valid subset of that size
    min_size = [11] * P  # 11 = infinity (max possible is 10)
    best_subset = [None] * P
    
    for mask in range(1, 1 << 10):
        if time.time() > deadline:
            break
        size = bin(mask).count('1')
        
        residue = 0
        for i in range(10):
            if mask & (1 << i):
                residue = (residue + inv[i + 1]) % P
        
        if size < min_size[residue]:
            min_size[residue] = size
            best_subset[residue] = [i + 1 for i in range(10) if mask & (1 << i)]
    
    # Build result
    result = []
    for r in range(P):
        if best_subset[r] is not None:
            result.append(best_subset[r])
        else:
            # Fallback: should not happen, but find any subset
            for mask in range(1, 1 << 10):
                residue = 0
                for i in range(10):
                    if mask & (1 << i):
                        residue = (residue + inv[i + 1]) % P
                if residue == r:
                    result.append([i + 1 for i in range(10) if mask & (1 << i)])
                    break
            else:
                result.append([1])  # Should never reach here
    
    return result

# EVOLVE-BLOCK-END
