import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 12
    m = 40
    
    def count_pairs(family_masks):
        """
        Count unordered pairs where one mask is a strict subset of the other.
        family_masks: list of integers
        """
        count = 0
        # Optimization: sort by bit count? 
        # If a has fewer bits than b, a can only be subset of b.
        # If a has same bits, they are not strict subsets of each other (unless equal, but distinct).
        # So we only need to check pairs where bit_count(a) < bit_count(b).
        
        # Precompute bit counts
        bc = [bin(x).count('1') for x in family_masks]
        
        # Sort indices by bit count
        indices = sorted(range(m), key=lambda i: bc[i])
        
        for i_idx in range(m):
            i = indices[i_idx]
            mask_i = family_masks[i]
            ci = bc[i]
            for j_idx in range(i_idx + 1, m):
                j = indices[j_idx]
                cj = bc[j]
                if ci >= cj:
                    break # Since sorted, all subsequent will have >= bits, so no strict subset possible from i to j
                # Check if mask_i is subset of mask_j
                if (mask_i & family_masks[j]) == mask_i:
                    count += 1
        return count

    def generate_initial():
        # Start with a chain
        masks = []
        # Chain: 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095
        for k in range(n + 1):
            masks.append((1 << k) - 1)
        
        # Add random distinct masks
        while len(masks) < m:
            mask = random.getrandbits(n)
            if mask not in masks:
                masks.append(mask)
        return masks

    def propose_move(family_masks):
        """
        Propose a new family by modifying one element.
        Returns a new list of masks.
        """
        new_masks = family_masks[:]
        idx = random.randint(0, m - 1)
        move_type = random.choice(['flip', 'replace', 'shift'])
        
        old_mask = new_masks[idx]
        new_mask = -1
        
        if move_type == 'flip':
            # Flip a random bit
            bit = random.randint(0, n - 1)
            new_mask = old_mask ^ (1 << bit)
            
        elif move_type == 'replace':
            # Random new mask
            new_mask = random.getrandbits(n)
            
        elif move_type == 'shift':
            # Remove a random present bit, add a random absent bit
            present_bits = [b for b in range(n) if (old_mask >> b) & 1]
            absent_bits = [b for b in range(n) if not ((old_mask >> b) & 1)]
            
            if present_bits and absent_bits:
                rm = random.choice(present_bits)
                add = random.choice(absent_bits)
                new_mask = (old_mask & ~(1 << rm)) | (1 << add)
            elif present_bits:
                rm = random.choice(present_bits)
                new_mask = old_mask & ~(1 << rm)
            elif absent_bits:
                add = random.choice(absent_bits)
                new_mask = old_mask | (1 << add)
            else:
                return None # No move possible, should not happen for n=12, m=40

        # Check distinctness
        if new_mask == old_mask:
            return None
        # Check if new_mask exists elsewhere
        # Since we only changed one element, we just need to check against the others
        for k in range(m):
            if k != idx and new_masks[k] == new_mask:
                return None
                
        new_masks[idx] = new_mask
        return new_masks

    best_score = -1
    best_family = None
    
    # Run multiple restarts
    while time.time() < deadline:
        # Generate initial
        current_family = generate_initial()
        current_score = count_pairs(current_family)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try a batch of moves
            for _ in range(200):
                if time.time() > deadline - 5.0:
                    break
                new_family = propose_move(current_family)
                if new_family is None:
                    continue
                new_score = count_pairs(new_family)
                if new_score > current_score:
                    current_family = new_family
                    current_score = new_score
                    improved = True
                    break # Restart the batch to explore around new best
            
        if current_score > best_score:
            best_score = current_score
            best_family = current_family[:]
            
        # If we found a perfect score (max possible), stop
        # Max possible is C(40, 2) = 780
        if best_score == 780:
            break

    # Convert masks to lists of integers
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
        
    return result
