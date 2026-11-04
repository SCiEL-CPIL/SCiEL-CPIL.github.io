"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from functools import lru_cache

def construct(n_rows, k):
    num_elements = k + 1  # 7 elements per row
    
    # Generate all Golomb rulers of order num_elements up to a max length
    # A Golomb ruler has all pairwise differences distinct
    def generate_golomb_rulers(max_len, order):
        rulers = []
        # First element is always 0
        def backtrack(current, last):
            if len(current) == order:
                rulers.append(tuple(current))
                return
            for next_val in range(last + 1, max_len + 1):
                # Check if adding next_val creates duplicate differences
                valid = True
                for a in current:
                    d = next_val - a
                    # Check if this difference already exists in current
                    for i in range(len(current)):
                        for j in range(i + 1, len(current)):
                            if current[j] - current[i] == d:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    current.append(next_val)
                    backtrack(current, next_val)
                    current.pop()
        
        backtrack([0], 0)
        return rulers
    
    # We need to find n_rows rulers that are pairwise distance-disjoint
    # and minimize the maximum element
    # Use iterative deepening on the max value
    
    for max_val in range(50, 200):
        # Generate all Golomb rulers up to max_val
        rulers = generate_golomb_rulers(max_val, num_elements)
        
        if len(rulers) < n_rows:
            continue
        
        # For each ruler, compute its set of differences
        ruler_diffs = []
        for r in rulers:
            diffs = set()
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.add(r[j] - r[i])
            ruler_diffs.append(diffs)
        
        # Now find n_rows rulers with disjoint difference sets
        # Use backtracking with constraint propagation
        selected = []
        used_diffs = set()
        
        def find_disjoint(idx, remaining_needed):
            if remaining_needed == 0:
                return True
            
            # Try each remaining ruler
            for i in range(idx, len(rulers)):
                if not ruler_diffs[i] & used_diffs:
                    selected.append(i)
                    used_diffs.update(ruler_diffs[i])
                    
                    if find_disjoint(i + 1, remaining_needed - 1):
                        return True
                    
                    selected.pop()
                    used_diffs.difference_update(ruler_diffs[i])
            
            return False
        
        if find_disjoint(0, n_rows):
            return [list(rulers[i]) for i in selected]
    
    # Fallback to greedy
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
# EVOLVE-BLOCK-END
