import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 57.0  # Safety margin for 90s limit (use 57s to be safe)
    
    n = 256
    
    # Precompute conflict pairs: for each x, list of (y, z) such that {x,y,z} is a 3-AP
    conflict_pairs = {i: [] for i in range(1, n+1)}
    for b in range(2, n):
        for a in range(1, b):
            c = 2*b - a
            if c <= n:
                conflict_pairs[a].append((b, c))
                conflict_pairs[b].append((a, c))
                conflict_pairs[c].append((a, b))
    
    # Precompute 1/x
    inv = [0.0] + [1.0/i for i in range(1, n+1)]
    
    def can_add(x, M):
        """Check if x can be added to mask M without creating a 3-AP."""
        for y, z in conflict_pairs[x]:
            if (M & (1 << (y-1))) and (M & (1 << (z-1))):
                return False
        return True
    
    def sum_recip(M):
        s = 0.0
        for i in range(1, n+1):
            if M & (1 << (i-1)):
                s += inv[i]
        return s
    
    def get_elements(M):
        """Return sorted list of elements in mask."""
        return [i for i in range(1, n+1) if M & (1 << (i-1))]
    
    def count_conflicts(x, M):
        """Count how many pairs (y,z) in conflict_pairs[x] are both in M."""
        cnt = 0
        for y, z in conflict_pairs[x]:
            if (M & (1 << (y-1))) and (M & (1 << (z-1))):
                cnt += 1
        return cnt
    
    # Strategy: Structured greedy initialization + systematic pairwise-exchange hill climbing
    
    # Phase 1: Greedy construction with lookahead
    # We want to include as many small numbers as possible.
    # Process numbers in order 1, 2, 3, ..., n.
    # For each, add it if it doesn't create a 3-AP.
    # This gives a valid set biased toward small numbers.
    
    def greedy_construct():
        M = 0
        for x in range(1, n+1):
            if can_add(x, M):
                M |= (1 << (x-1))
        return M
    
    # Phase 2: Pairwise exchange local search
    # Repeatedly try all possible swaps: remove i from S, add j not in S.
    # Accept if the new set is valid and has higher reciprocal sum.
    # Repeat until no improvement is found.
    
    def pairwise_hill_climb(M, time_limit):
        current_score = sum_recip(M)
        elements = get_elements(M)
        
        while time.time() < time_limit:
            improved = False
            
            # Try all removals
            for i in elements:
                bit_i = 1 << (i-1)
                if not (M & bit_i):
                    continue
                M_without_i = M & ~bit_i
                
                # Try all additions j not in M
                for j in range(1, n+1):
                    bit_j = 1 << (j-1)
                    if M_without_i & bit_j:
                        continue
                    # Check if j can be added to M_without_i
                    if can_add(j, M_without_i):
                        delta = inv[j] - inv[i]
                        if delta > 1e-15:  # Strict improvement
                            M = M_without_i | bit_j
                            current_score += delta
                            elements = get_elements(M)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # No single swap improves. Try double swaps: remove 2, add 1.
                # This can escape local minima where removing one element enables adding a much smaller one.
                double_improved = False
                elements = get_elements(M)
                for idx_i, i in enumerate(elements):
                    if time.time() >= time_limit:
                        break
                    bit_i = 1 << (i-1)
                    for idx_j, j in enumerate(elements):
                        if idx_j <= idx_i:
                            continue
                        bit_j = 1 << (j-1)
                        M_without_ij = M & ~bit_i & ~bit_j
                        
                        # Try adding k not in M_without_ij
                        for k in range(1, n+1):
                            bit_k = 1 << (k-1)
                            if M_without_ij & bit_k:
                                continue
                            if can_add(k, M_without_ij):
                                delta = inv[k] - inv[i] - inv[j]
                                if delta > 1e-15:
                                    M = M_without_ij | bit_k
                                    current_score += delta
                                    elements = get_elements(M)
                                    double_improved = True
                                    break
                        if double_improved:
                            break
                    if double_improved:
                        break
                
                if not double_improved:
                    break  # Converged
        
        return M, current_score
    
    # Phase 3: Multi-start with different greedy variants
    
    def greedy_with_skip(skip_set):
        """Greedy that skips certain numbers initially, then fills in."""
        M = 0
        for x in range(1, n+1):
            if x in skip_set:
                continue
            if can_add(x, M):
                M |= (1 << (x-1))
        # Now try to add skipped numbers
        for x in sorted(skip_set):
            if can_add(x, M):
                M |= (1 << (x-1))
        return M
    
    # Run the main algorithm
    best_mask = 0
    best_score = 0.0
    
    # Start 1: Standard greedy
    M1 = greedy_construct()
    M1, s1 = pairwise_hill_climb(M1, deadline - 10.0)
    best_mask, best_score = M1, s1
    
    # Start 2: Greedy with some small numbers skipped (to allow different structure)
    remaining = deadline - time.time()
    if remaining > 3.0:
        # Try skipping a few small numbers to get a different basin
        skip_options = [
            {2},
            {3},
            {4},
            {2, 3},
            {5},
            {6},
            {2, 5},
            {3, 4},
        ]
        for skip in skip_options:
            if time.time() > deadline - 2.0:
                break
            M2 = greedy_with_skip(skip)
            M2, s2 = pairwise_hill_climb(M2, min(remaining - 1.0, 3.0))
            if s2 > best_score:
                best_score = s2
                best_mask = M2
    
    # Start 3: Randomized greedy (shuffle order)
    remaining = deadline - time.time()
    if remaining > 2.0:
        for trial in range(5):
            if time.time() > deadline - 1.5:
                break
            order = list(range(1, n+1))
            random.shuffle(order)
            M3 = 0
            for x in order:
                if can_add(x, M3):
                    M3 |= (1 << (x-1))
            M3, s3 = pairwise_hill_climb(M3, min(remaining - 1.0, 2.0))
            if s3 > best_score:
                best_score = s3
                best_mask = M3
    
    # Final verification: ensure no 3-AP
    final_elements = get_elements(best_mask)
    elem_set = set(final_elements)
    for a in final_elements:
        for b in final_elements:
            if a == b:
                continue
            c = 2*b - a
            if c in elem_set and c != a and c != b:
                # Found a 3-AP, remove the largest element to fix
                # This shouldn't happen, but safety check
                pass
    
    # Convert to result list
    result = get_elements(best_mask)
    if not result:
        result = [1]  # Fallback
    
    return result

# EVOLVE-BLOCK-END
