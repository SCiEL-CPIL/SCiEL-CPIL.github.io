"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [pow(x, -1, 101) for x in range(1, 11)]
    
    # Precompute all non-empty subsets: (size, residue, mask)
    all_subsets = []
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        size = len(terms)
        residue = sum(inv[x - 1] for x in terms) % 101
        all_subsets.append((size, residue, mask))
    
    # Group subsets by residue
    by_residue = [[] for _ in range(101)]
    for size, residue, mask in all_subsets:
        by_residue[residue].append((size, mask))
    
    # For each residue, find the minimum size needed
    min_sizes = [11] * 101  # 11 means not found yet (max possible is 10)
    for r in range(101):
        if by_residue[r]:
            min_sizes[r] = min(s for s, m in by_residue[r])
    
    # The global maximum of minimum sizes
    max_min_size = max(min_sizes)
    
    # Now, for each residue, pick a subset of size at most max_min_size
    # To minimize the largest row length, we want to use subsets of size <= max_min_size
    # But we also want to minimize the actual maximum, so let's try to find the minimum possible max
    
    # Actually, the minimum possible max row length is max_min_size, since each residue 
    # needs at least min_sizes[r] elements, and max over r of min_sizes[r] = max_min_size
    
    # So we just need to pick, for each residue, a subset of size exactly min_sizes[r]
    # (or any size <= max_min_size, but using min_sizes[r] is optimal)
    
    # Build the result: for each residue, pick a subset of minimum size
    result = []
    for r in range(101):
        # Find a subset of size min_sizes[r] for residue r
        best_mask = None
        for size, mask in by_residue[r]:
            if size == min_sizes[r]:
                best_mask = mask
                break
        if best_mask is None:
            # Should not happen, but fallback
            best_mask = by_residue[r][0][1]
        
        terms = [x for x in range(1, 11) if (best_mask >> (x - 1)) & 1]
        result.append(terms)
    
    return result
# EVOLVE-BLOCK-END
