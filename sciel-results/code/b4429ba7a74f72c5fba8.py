import itertools
import time

def construct():
        # Precomputation
        points = range(12)
        all_5sets = list(itertools.combinations(points, 5))
        n_5sets = len(all_5sets)
        # Map 5set to index
        idx_5set = {s: i for i, s in enumerate(all_5sets)}
        
        all_6sets = list(itertools.combinations(points, 6))
        n_6sets = len(all_6sets)
        
        # Precompute masks for each 6set
        block_masks = [0] * n_6sets
        # For each 5set, list of blocks covering it
        blocks_for_5set = [[] for _ in range(n_5sets)]
        
        for b_idx, b in enumerate(all_6sets):
            mask = 0
            for s in itertools.combinations(b, 5):
                s_idx = idx_5set[s]
                mask |= (1 << s_idx)
                blocks_for_5set[s_idx].append(b_idx)
            block_masks[b_idx] = mask
        
        full_mask = (1 << n_5sets) - 1
        solution = []
        
        # Backtracking
        def backtrack(covered_mask, blocks_used):
            if covered_mask == full_mask:
                solution.extend(blocks_used)
                return True
            
            # Find first uncovered 5set
            # lowest set bit in ~covered_mask
            inv = ~covered_mask & full_mask
            # Find index of lowest set bit
            # In Python, (x & -x).bit_length() - 1 gives index of lowest set bit
            if inv == 0:
                return False # Should not happen if logic is correct
            
            # Get index of the uncovered 5set
            # We need the index of the lowest set bit in inv
            # inv & -inv gives the lowest set bit as a number
            low_bit = inv & -inv
            s_idx = low_bit.bit_length() - 1
            
            # Try all blocks covering this 5set
            for b_idx in blocks_for_5set[s_idx]:
                b_mask = block_masks[b_idx]
                if (b_mask & covered_mask) == 0:
                    # Disjoint
                    if backtrack(covered_mask | b_mask, blocks_used + [all_6sets[b_idx]]):
                        return True
            
            return False
        
        backtrack(0, [])
        
        return solution
