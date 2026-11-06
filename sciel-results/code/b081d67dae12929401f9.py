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
    
    # For each residue, find the minimum-size subset that sums to it
    # Use bitmask enumeration for speed
    best = [None] * P
    min_size = [11] * P  # 11 = infinity (max possible is 10)
    
    # Enumerate all non-empty subsets via bitmask
    # 2^10 = 1024 subsets, very fast
    for mask in range(1, 1 << 10):
        if time.time() > deadline:
            break
        size = bin(mask).count('1')
        
        # Quick skip: if size >= min_size for all residues we could still improve, 
        # but let's just compute all since 1023 is tiny
        residue = 0
        for i in range(10):
            if mask & (1 << i):
                residue = (residue + inv[i + 1]) % P
        
        if size < min_size[residue]:
            min_size[residue] = size
            best[residue] = [i + 1 for i in range(10) if mask & (1 << i)]
    
    # Build result
    result = []
    for r in range(P):
        if best[r] is not None:
            result.append(best[r])
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
