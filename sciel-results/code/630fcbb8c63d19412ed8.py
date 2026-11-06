import time
import random

def construct():
    """
    Finds the largest Sidon set (distinct pairwise sums) in [1, 284]
    containing {1, 2, 5, 11}.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s limit to be safe
    
    # Initial set
    initial_set = [1, 2, 5, 11]
    
    # Pre-compute initial sums mask
    def get_sums_mask(S):
        mask = 0
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                mask |= (1 << (S[i] + S[j]))
        return mask

    initial_mask = get_sums_mask(initial_set)
    
    # Beam Search Parameters
    beam_width = 100
    max_depth = 35
    
    # State: (S_list, sums_mask, size)
    # We store S as a tuple for hashing/comparison if needed, but list is fine for mutation in beam
    beam = [(list(initial_set), initial_mask, len(initial_set))]
    
    best_S = list(initial_set)
    best_size = len(initial_set)
    
    depth = 0
    while depth < max_depth:
        current_time = time.time()
        if current_time > deadline:
            break
            
        next_beam_candidates = []
        
        for S, mask, size in beam:
            # Try adding each candidate x
            # Optimization: We can skip x if it's in S
            S_set = set(S)
            
            # To improve diversity, we can shuffle the range or use different orders
            # But standard 1..284 is fine.
            
            for x in range(1, 285):
                if x in S_set:
                    continue
                
                # Check conflicts
                conflict = False
                new_sums_mask = mask
                for y in S:
                    s = x + y
                    if mask & (1 << s):
                        conflict = True
                        break
                    new_sums_mask |= (1 << s)
                
                if conflict:
                    continue
                
                s = 2 * x
                if mask & (1 << s):
                    continue
                new_sums_mask |= (1 << s)
                
                # Valid extension
                new_S = S + [x]
                next_beam_candidates.append((new_S, new_sums_mask, size + 1))
                
                # Early break if we have too many candidates?
                # No, we need to generate all to prune properly.
                
        if not next_beam_candidates:
            break
            
        # Prune next_beam_candidates to beam_width
        # Sort by size descending
        next_beam_candidates.sort(key=lambda x: -x[2])
        
        # If we have more than beam_width, we need to select a diverse subset
        if len(next_beam_candidates) > beam_width:
            # Strategy: Keep top K by size, but ensure diversity
            # Simple approach: Take the first beam_width
            # Better approach: Stratified sampling
            # Let's just take the top beam_width for now, 
            # but add a small random component to avoid determinism
            # Shuffle the list of candidates that have the same max size?
            
            # Identify max size in candidates
            max_size_in_batch = next_beam_candidates[0][2]
            
            # Partition into groups by size
            groups = {}
            for state in next_beam_candidates:
                s = state[2]
                if s not in groups:
                    groups[s] = []
                groups[s].append(state)
            
            selected = []
            # Take from largest size down
            for s in sorted(groups.keys(), reverse=True):
                if len(selected) >= beam_width:
                    break
                group = groups[s]
                # If group is larger than remaining slots, sample
                if len(group) > beam_width - len(selected):
                    # Shuffle and take
                    random.shuffle(group)
                    selected.extend(group[:beam_width - len(selected)])
                else:
                    selected.extend(group)
            
            beam = selected
        else:
            beam = next_beam_candidates
            
        # Update best
        for S, mask, size in beam:
            if size > best_size:
                best_size = size
                best_S = S
                
        depth += 1
        
    # Final check: ensure the returned set is valid and contains required elements
    # (It should, by construction, but good to be safe)
    # The problem requires distinct integers. Our construction ensures this.
    
    return best_S

# EVOLVE-BLOCK-END
